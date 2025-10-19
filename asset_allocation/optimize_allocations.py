"""
Asset Allocation Optimization Module

This module implements a transaction-aware optimization algorithm for allocating assets
to stablecoin pools. It uses CVXPY with GUROBI solver to maximize daily yield while
accounting for gas fees, conversion costs, and various constraints.

Key Features:
- Transaction-level modeling (withdrawals, conversions, allocations)
- Multi-token pool support with even distribution requirements
- Rebalancing logic considering existing allocations
- Gas fee and conversion cost optimization
- Comprehensive constraint system
"""

import logging
import cvxpy as cp
import pandas as pd
import numpy as np
import json
from datetime import datetime, date, timezone
from uuid import uuid4
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from database.db_utils import get_db_connection

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def fetch_pool_data(engine) -> pd.DataFrame:
    """
    Fetches approved pools with forecasted APY and metadata.
    
    Returns:
        DataFrame with columns: pool_id, symbol, chain, protocol, forecasted_apy
    """
    query = """
    SELECT
        pdm.pool_id,
        p.symbol,
        p.chain,
        p.protocol,
        pdm.forecasted_apy
    FROM pool_daily_metrics pdm
    JOIN pools p ON pdm.pool_id = p.pool_id
    WHERE pdm.date = CURRENT_DATE 
      AND pdm.is_filtered_out = FALSE
      AND pdm.forecasted_apy IS NOT NULL
      AND pdm.forecasted_apy > 0;
    """
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} approved pools")
    return df


def fetch_token_prices(engine, tokens: List[str]) -> Dict[str, float]:
    """
    Fetches latest closing prices for given tokens.
    
    Args:
        tokens: List of token symbols
        
    Returns:
        Dictionary mapping token symbol to USD price
    """
    if not tokens:
        return {}
    
    tokens_str = "','".join(tokens)
    query = f"""
    WITH ranked_ohlcv AS (
        SELECT
            raw_json_data->>'symbol' as symbol,
            (raw_json_data->>'close')::float as close_price,
            (raw_json_data->>'timestamp')::timestamp as ts,
            ROW_NUMBER() OVER(
                PARTITION BY raw_json_data->>'symbol' 
                ORDER BY (raw_json_data->>'timestamp')::timestamp DESC
            ) as rn
        FROM raw_coinmarketcap_ohlcv
        WHERE raw_json_data->>'symbol' IN ('{tokens_str}')
    )
    SELECT symbol, close_price
    FROM ranked_ohlcv
    WHERE rn = 1;
    """
    df = pd.read_sql(query, engine)
    prices = dict(zip(df['symbol'], df['close_price']))
    logger.info(f"Loaded prices for {len(prices)} tokens")
    return prices


def fetch_gas_fee_data(engine) -> Tuple[float, float]:
    """
    Fetches forecasted gas fee and ETH price.
    
    Returns:
        Tuple of (forecasted_max_gas_gwei, eth_price_usd)
    """
    gas_query = """
    SELECT forecasted_max_gas_gwei
    FROM gas_fees_daily
    WHERE date = CURRENT_DATE
    ORDER BY date DESC
    LIMIT 1;
    """
    gas_df = pd.read_sql(gas_query, engine)
    gas_gwei = gas_df['forecasted_max_gas_gwei'].iloc[0] if not gas_df.empty else 50.0
    
    # Fetch ETH price
    eth_price_query = """
    WITH ranked_eth AS (
        SELECT
            (raw_json_data->>'close')::float as close_price,
            ROW_NUMBER() OVER(ORDER BY (raw_json_data->>'timestamp')::timestamp DESC) as rn
        FROM raw_coinmarketcap_ohlcv
        WHERE raw_json_data->>'symbol' = 'ETH'
    )
    SELECT close_price
    FROM ranked_eth
    WHERE rn = 1;
    """
    eth_df = pd.read_sql(eth_price_query, engine)
    eth_price = eth_df['close_price'].iloc[0] if not eth_df.empty else 3000.0
    
    gas_fee_usd = gas_gwei * 1e-9 * eth_price
    logger.info(f"Gas fee: {gas_gwei:.2f} Gwei, ETH price: ${eth_price:.2f}, Gas fee USD: ${gas_fee_usd:.6f}")
    
    return gas_gwei, eth_price


def fetch_current_balances(engine) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """
    Fetches current token balances from cold wallet and allocated positions.
    
    Returns:
        Tuple of (cold_wallet_balances, current_allocations)
        - cold_wallet_balances: {token_symbol: amount}
        - current_allocations: {(pool_id, token_symbol): amount}
    """
    query = """
    SELECT
        token_symbol,
        unallocated_balance,
        allocated_balance,
        pool_id
    FROM daily_balances
    WHERE date = CURRENT_DATE - INTERVAL '1 day';
    """
    df = pd.read_sql(query, engine)
    
    cold_wallet = {}
    allocations = {}
    
    for _, row in df.iterrows():
        token = row['token_symbol']
        
        # Unallocated balance in cold wallet
        if pd.notna(row['unallocated_balance']) and row['unallocated_balance'] > 0:
            cold_wallet[token] = cold_wallet.get(token, 0) + float(row['unallocated_balance'])
        
        # Allocated balance to pools
        if pd.notna(row['allocated_balance']) and row['allocated_balance'] > 0 and pd.notna(row['pool_id']):
            key = (row['pool_id'], token)
            allocations[key] = allocations.get(key, 0) + float(row['allocated_balance'])
    
    logger.info(f"Cold wallet: {len(cold_wallet)} tokens, Total allocated positions: {len(allocations)}")
    return cold_wallet, allocations


def fetch_allocation_parameters(engine) -> Dict:
    """Fetches the latest allocation parameters."""
    query = """
    SELECT *
    FROM allocation_parameters
    ORDER BY timestamp DESC
    LIMIT 1;
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        logger.warning("No allocation parameters found, using defaults")
        return {
            'max_alloc_percentage': 0.20,
            'conversion_rate': 0.0004,
            'min_transaction_value': 50.0
        }
    
    params = df.iloc[0].to_dict()
    logger.info(f"Loaded allocation parameters: max_alloc={params.get('max_alloc_percentage')}")
    return params


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_pool_tokens(symbol: str) -> List[str]:
    """
    Extracts tokens from pool symbol.
    
    Args:
        symbol: Pool symbol (e.g., "DAI-USDC-USDT" or "GTUSDC")
        
    Returns:
        List of token symbols in uppercase
    """
    return [t.upper().strip() for t in symbol.split('-')]


def calculate_aum(cold_wallet: Dict[str, float], 
                  current_allocations: Dict[Tuple[str, str], float],
                  token_prices: Dict[str, float]) -> float:
    """
    Calculates total Assets Under Management in USD.
    
    Args:
        cold_wallet: Unallocated token balances
        current_allocations: Allocated positions
        token_prices: Token prices in USD
        
    Returns:
        Total AUM in USD
    """
    total_usd = 0.0
    
    # Cold wallet value
    for token, amount in cold_wallet.items():
        price = token_prices.get(token, 1.0)  # Default to $1 for stablecoins
        total_usd += amount * price
    
    # Allocated positions value
    for (pool_id, token), amount in current_allocations.items():
        price = token_prices.get(token, 1.0)
        total_usd += amount * price
    
    logger.info(f"Total AUM: ${total_usd:,.2f}")
    return total_usd


def build_token_universe(pools_df: pd.DataFrame, 
                         cold_wallet: Dict[str, float],
                         current_allocations: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Builds the complete set of tokens needed for optimization.
    
    Returns:
        Sorted list of unique token symbols
    """
    tokens = set()
    
    # Tokens from pools
    for symbol in pools_df['symbol']:
        tokens.update(parse_pool_tokens(symbol))
    
    # Tokens in cold wallet
    tokens.update(cold_wallet.keys())
    
    # Tokens in current allocations
    for (pool_id, token) in current_allocations.keys():
        tokens.add(token)
    
    token_list = sorted(list(tokens))
    logger.info(f"Token universe: {len(token_list)} tokens - {token_list}")
    return token_list


# ============================================================================
# OPTIMIZATION MODEL
# ============================================================================

class AllocationOptimizer:
    """
    Transaction-aware portfolio optimization for stablecoin pools.
    """
    
    def __init__(self, pools_df: pd.DataFrame, token_prices: Dict[str, float],
                 cold_wallet: Dict[str, float], current_allocations: Dict[Tuple[str, str], float],
                 gas_fee_usd: float, alloc_params: Dict):
        """
        Initialize the optimizer.
        
        Args:
            pools_df: DataFrame of approved pools
            token_prices: Token price dictionary
            cold_wallet: Current cold wallet balances
            current_allocations: Current pool allocations
            gas_fee_usd: Gas fee per transaction in USD
            alloc_params: Allocation parameters
        """
        self.pools_df = pools_df
        self.token_prices = token_prices
        self.cold_wallet = cold_wallet
        self.current_allocations = current_allocations
        self.gas_fee_usd = gas_fee_usd
        self.alloc_params = alloc_params
        
        # Build indices
        self.tokens = build_token_universe(pools_df, cold_wallet, current_allocations)
        self.pools = pools_df['pool_id'].tolist()
        
        self.n_tokens = len(self.tokens)
        self.n_pools = len(self.pools)
        
        self.token_idx = {t: i for i, t in enumerate(self.tokens)}
        self.pool_idx = {p: i for i, p in enumerate(self.pools)}
        
        # Build pool-token mapping
        self.pool_tokens = {}  # pool_id -> list of token symbols
        for idx, row in pools_df.iterrows():
            pool_id = row['pool_id']
            tokens = parse_pool_tokens(row['symbol'])
            self.pool_tokens[pool_id] = tokens
        
        # Calculate AUM
        self.total_aum = calculate_aum(cold_wallet, current_allocations, token_prices)
        
        # Constants
        self.conversion_rate = alloc_params.get('conversion_rate', 0.0004)
        self.min_transaction_value = alloc_params.get('min_transaction_value', 50.0)
        self.max_alloc_percentage = alloc_params.get('max_alloc_percentage', 0.20)
        
        logger.info(f"Optimizer initialized: {self.n_pools} pools, {self.n_tokens} tokens, AUM=${self.total_aum:,.2f}")
    
    def build_model(self) -> cp.Problem:
        """
        Constructs the CVXPY optimization model.
        
        Returns:
            CVXPY Problem instance
        """
        logger.info("Building optimization model...")
        
        # ====================================================================
        # DECISION VARIABLES
        # ====================================================================
        
        # Final allocation to each (pool, token) - the target state
        # Shape: (n_pools, n_tokens)
        self.alloc = cp.Variable((self.n_pools, self.n_tokens), nonneg=True)
        
        # Withdrawal from each (pool, token)
        self.withdraw = cp.Variable((self.n_pools, self.n_tokens), nonneg=True)
        
        # Conversion between tokens: convert[i,j] = amount of token i converted to token j
        # Shape: (n_tokens, n_tokens)
        self.convert = cp.Variable((self.n_tokens, self.n_tokens), nonneg=True)
        
        # Final cold wallet balance for each token
        self.final_cold_wallet = cp.Variable(self.n_tokens, nonneg=True)
        
        # ====================================================================
        # INITIAL STATE VECTORS
        # ====================================================================
        
        # Current allocation matrix
        current_alloc_matrix = np.zeros((self.n_pools, self.n_tokens))
        for (pool_id, token), amount in self.current_allocations.items():
            if pool_id in self.pool_idx and token in self.token_idx:
                i = self.pool_idx[pool_id]
                j = self.token_idx[token]
                current_alloc_matrix[i, j] = amount
        
        # Current cold wallet vector
        cold_wallet_vector = np.zeros(self.n_tokens)
        for token, amount in self.cold_wallet.items():
            if token in self.token_idx:
                j = self.token_idx[token]
                cold_wallet_vector[j] = amount
        
        # Token prices vector
        price_vector = np.array([self.token_prices.get(t, 1.0) for t in self.tokens])
        
        # ====================================================================
        # OBJECTIVE FUNCTION
        # ====================================================================
        
        # Daily yield = sum over all pools of (allocation * daily_apy * price)
        # daily_apy = forecasted_apy / 365
        daily_apy_matrix = np.zeros((self.n_pools, self.n_tokens))
        
        for idx, row in self.pools_df.iterrows():
            pool_id = row['pool_id']
            i = self.pool_idx[pool_id]
            daily_apy = row['forecasted_apy'] / 365.0
            
            # Apply APY to all tokens in this pool
            for token in self.pool_tokens[pool_id]:
                if token in self.token_idx:
                    j = self.token_idx[token]
                    daily_apy_matrix[i, j] = daily_apy
        
        # Total yield in USD
        yield_usd = cp.sum(cp.multiply(
            cp.multiply(self.alloc, daily_apy_matrix),
            price_vector
        ))
        
        # Transaction costs
        # 1. Gas costs - we approximate by using the sum of all transaction amounts
        # This creates a continuous approximation rather than counting discrete transactions
        total_gas_cost = (cp.sum(self.withdraw) + cp.sum(self.alloc) + cp.sum(self.convert)) * self.gas_fee_usd / self.total_aum
        
        # 2. Conversion costs = sum of all conversions * conversion_rate * price
        conversion_cost = cp.sum(cp.multiply(
            cp.sum(self.convert, axis=1),
            price_vector
        )) * self.conversion_rate
        
        # Net objective: maximize daily yield minus costs
        objective = cp.Maximize(yield_usd - conversion_cost - total_gas_cost * self.total_aum)
        
        # ====================================================================
        # CONSTRAINTS
        # ====================================================================
        
        constraints = []
        
        # 1. Token balance conservation
        # For each token: initial_cold_wallet + withdrawals + conversions_in 
        #                 = final_cold_wallet + allocations + conversions_out
        for j in range(self.n_tokens):
            token_in = (
                cold_wallet_vector[j] +  # Initial cold wallet
                cp.sum(self.withdraw[:, j]) +  # Withdrawals from pools
                cp.sum(self.convert[:, j])  # Conversions into this token
            )
            token_out = (
                self.final_cold_wallet[j] +  # Final cold wallet
                cp.sum(self.alloc[:, j]) +  # Allocations to pools
                cp.sum(self.convert[j, :])  # Conversions out of this token
            )
            constraints.append(token_in == token_out)
        
        # 2. Withdrawal constraints: can only withdraw what's currently allocated
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                constraints.append(self.withdraw[i, j] <= current_alloc_matrix[i, j])
        
        # 3. No self-conversion
        for j in range(self.n_tokens):
            constraints.append(self.convert[j, j] == 0)
        
        # 4. Multi-token pool even distribution
        for pool_id, tokens in self.pool_tokens.items():
            if len(tokens) > 1:  # Multi-token pool
                i = self.pool_idx[pool_id]
                token_indices = [self.token_idx[t] for t in tokens if t in self.token_idx]
                
                if len(token_indices) > 1:
                    # All tokens in this pool must have equal USD value
                    for k in range(len(token_indices) - 1):
                        j1 = token_indices[k]
                        j2 = token_indices[k + 1]
                        # alloc[i, j1] * price[j1] == alloc[i, j2] * price[j2]
                        constraints.append(
                            self.alloc[i, j1] * price_vector[j1] == 
                            self.alloc[i, j2] * price_vector[j2]
                        )
        
        # 5. Pool allocation limits
        # Maximum allocation per pool as percentage of total AUM
        for i in range(self.n_pools):
            pool_total_usd = cp.sum(cp.multiply(self.alloc[i, :], price_vector))
            constraints.append(pool_total_usd <= self.max_alloc_percentage * self.total_aum)
        
        # 6. Total allocated amount <= total AUM
        total_allocated_usd = cp.sum(cp.multiply(self.alloc, price_vector))
        constraints.append(total_allocated_usd <= self.total_aum)
        
        # 7. Non-negativity (already enforced by variable definition, but adding explicitly)
        constraints.append(self.alloc >= 0)
        constraints.append(self.withdraw >= 0)
        constraints.append(self.convert >= 0)
        constraints.append(self.final_cold_wallet >= 0)
        
        logger.info(f"Model built with {len(constraints)} constraints")
        
        problem = cp.Problem(objective, constraints)
        return problem
    
    def solve(self, solver=cp.GUROBI, verbose=True) -> bool:
        """
        Solves the optimization problem.
        
        Args:
            solver: CVXPY solver to use
            verbose: Whether to print solver output
            
        Returns:
            True if optimal solution found, False otherwise
        """
        problem = self.build_model()
        
        logger.info(f"Solving with {solver}...")
        try:
            problem.solve(solver=solver, verbose=verbose)
        except Exception as e:
            logger.error(f"Solver error: {e}")
            logger.info("Attempting with ECOS solver as fallback...")
            try:
                problem.solve(solver=cp.ECOS, verbose=verbose)
            except Exception as e2:
                logger.error(f"ECOS solver also failed: {e2}")
                return False
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            logger.info(f"✓ Optimization successful: {problem.status}")
            logger.info(f"  Objective value: ${problem.value:,.4f} daily yield")
            return True
        else:
            logger.error(f"✗ Optimization failed: {problem.status}")
            return False
    
    def extract_results(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Extracts results from solved optimization model.
        
        Returns:
            Tuple of (allocations_df, transactions_list)
        """
        if self.alloc.value is None:
            logger.error("No solution available to extract")
            return pd.DataFrame(), []
        
        allocations = []
        transactions = []
        transaction_seq = 1
        
        price_vector = np.array([self.token_prices.get(t, 1.0) for t in self.tokens])
        
        # ====================================================================
        # STEP 1: WITHDRAWALS
        # ====================================================================
        
        for i in range(self.n_pools):
            pool_id = self.pools[i]
            for j in range(self.n_tokens):
                token = self.tokens[j]
                amount = self.withdraw.value[i, j] if self.withdraw.value is not None else 0
                
                if amount > 0.01:  # Threshold for meaningful transactions
                    transactions.append({
                        'seq': transaction_seq,
                        'type': 'WITHDRAWAL',
                        'from_location': pool_id,
                        'to_location': 'cold_wallet',
                        'token': token,
                        'amount': amount,
                        'amount_usd': amount * price_vector[j],
                        'gas_cost_usd': self.gas_fee_usd
                    })
                    transaction_seq += 1
        
        # ====================================================================
        # STEP 2: CONVERSIONS
        # ====================================================================
        
        for i in range(self.n_tokens):
            from_token = self.tokens[i]
            for j in range(self.n_tokens):
                to_token = self.tokens[j]
                if i != j:
                    amount = self.convert.value[i, j] if self.convert.value is not None else 0
                    
                    if amount > 0.01:
                        transactions.append({
                            'seq': transaction_seq,
                            'type': 'CONVERSION',
                            'from_location': 'cold_wallet',
                            'to_location': 'cold_wallet',
                            'from_token': from_token,
                            'to_token': to_token,
                            'amount': amount,
                            'amount_usd': amount * price_vector[i],
                            'conversion_fee_usd': amount * price_vector[i] * self.conversion_rate,
                            'gas_cost_usd': self.gas_fee_usd
                        })
                        transaction_seq += 1
        
        # ====================================================================
        # STEP 3: ALLOCATIONS
        # ====================================================================
        
        for i in range(self.n_pools):
            pool_id = self.pools[i]
            symbol = self.pools_df[self.pools_df['pool_id'] == pool_id]['symbol'].iloc[0]
            
            for j in range(self.n_tokens):
                token = self.tokens[j]
                amount = self.alloc.value[i, j] if self.alloc.value is not None else 0
                
                if amount > 0.01:
                    amount_usd = amount * price_vector[j]
                    
                    allocations.append({
                        'pool_id': pool_id,
                        'pool_symbol': symbol,
                        'token': token,
                        'amount': amount,
                        'amount_usd': amount_usd
                    })
                    
                    transactions.append({
                        'seq': transaction_seq,
                        'type': 'ALLOCATION',
                        'from_location': 'cold_wallet',
                        'to_location': pool_id,
                        'token': token,
                        'amount': amount,
                        'amount_usd': amount_usd,
                        'gas_cost_usd': self.gas_fee_usd
                    })
                    transaction_seq += 1
        
        allocations_df = pd.DataFrame(allocations)
        
        logger.info(f"Extracted {len(allocations)} allocations and {len(transactions)} transactions")
        
        return allocations_df, transactions


# ============================================================================
# RESULT PERSISTENCE
# ============================================================================

def store_results(engine, run_id: str, allocations_df: pd.DataFrame, 
                  transactions: List[Dict], alloc_params: Dict):
    """
    Stores optimization results to database.
    
    Args:
        engine: Database engine
        run_id: Unique run identifier
        allocations_df: Final allocations DataFrame
        transactions: List of transaction dictionaries
        alloc_params: Allocation parameters
    """
    conn = engine.raw_connection()
    cursor = conn.cursor()
    
    try:
        # Store allocation parameters snapshot
        cursor.execute("""
            INSERT INTO allocation_parameters (
                run_id, timestamp, max_alloc_percentage, conversion_rate
            ) VALUES (%s, %s, %s, %s);
        """, (
            run_id,
            datetime.now(timezone.utc),
            alloc_params.get('max_alloc_percentage'),
            alloc_params.get('conversion_rate')
        ))
        
        # Store transaction sequence
        for txn in transactions:
            cursor.execute("""
                INSERT INTO asset_allocations (
                    run_id, step_number, operation, from_asset, to_asset, 
                    amount, pool_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (
                run_id,
                txn['seq'],
                txn['type'],
                txn.get('from_token', txn.get('token')),
                txn.get('to_token', txn.get('token')),
                txn['amount'],
                txn.get('to_location') if txn['type'] == 'ALLOCATION' else None
            ))
        
        conn.commit()
        logger.info(f"✓ Results stored with run_id: {run_id}")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing results: {e}")
        raise
    finally:
        cursor.close()


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def optimize_allocations():
    """
    Main orchestration function for asset allocation optimization.
    """
    logger.info("=" * 80)
    logger.info("STABLECOIN POOL ALLOCATION OPTIMIZATION")
    logger.info("=" * 80)
    
    engine = None
    try:
        # Connect to database
        engine = get_db_connection()
        if not engine:
            logger.error("Failed to establish database connection")
            return
        
        # Load data
        logger.info("\n[1/5] Loading data...")
        pools_df = fetch_pool_data(engine)
        
        if pools_df.empty:
            logger.warning("No approved pools available. Exiting.")
            return
        
        # Build token universe
        cold_wallet, current_allocations = fetch_current_balances(engine)
        tokens = build_token_universe(pools_df, cold_wallet, current_allocations)
        
        # Fetch prices and gas fees
        token_prices = fetch_token_prices(engine, tokens + ['ETH'])
        gas_gwei, eth_price = fetch_gas_fee_data(engine)
        gas_fee_usd = gas_gwei * 1e-9 * eth_price
        
        # Fetch parameters
        alloc_params = fetch_allocation_parameters(engine)
        
        # Initialize optimizer
        logger.info("\n[2/5] Initializing optimizer...")
        optimizer = AllocationOptimizer(
            pools_df=pools_df,
            token_prices=token_prices,
            cold_wallet=cold_wallet,
            current_allocations=current_allocations,
            gas_fee_usd=gas_fee_usd,
            alloc_params=alloc_params
        )
        
        # Solve optimization
        logger.info("\n[3/5] Solving optimization problem...")
        success = optimizer.solve(solver=cp.GUROBI, verbose=True)
        
        if not success:
            logger.error("Optimization failed")
            return
        
        # Extract results
        logger.info("\n[4/5] Extracting results...")
        allocations_df, transactions = optimizer.extract_results()
        
        # Print results
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"\nFinal Allocations ({len(allocations_df)} positions):")
        logger.info("\n" + allocations_df.to_string(index=False))
        
        logger.info(f"\n\nTransaction Sequence ({len(transactions)} transactions):")
        for txn in transactions:
            logger.info(f"  {txn['seq']:3d}. {txn['type']:12s} | {txn.get('token', txn.get('from_token', ''))} | "
                       f"${txn['amount_usd']:10,.2f}")
        
        # Store results
        logger.info("\n[5/5] Storing results...")
        run_id = str(uuid4())
        store_results(engine, run_id, allocations_df, transactions, alloc_params)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✓ OPTIMIZATION COMPLETE - Run ID: {run_id}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Optimization failed with error: {e}", exc_info=True)
        raise
    
    finally:
        if engine:
            engine.dispose()


if __name__ == "__main__":
    optimize_allocations()