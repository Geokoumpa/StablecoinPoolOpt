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
import time
from datetime import datetime, date, timezone
from uuid import uuid4
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from database.db_utils import get_db_connection
from asset_allocation.data_quality_report import generate_data_quality_report

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
    Includes pools with current allocations even if they're below the APY limit.
    
    Returns:
        DataFrame with columns: pool_id, symbol, chain, protocol, forecasted_apy, forecasted_tvl, normalized_tokens
    """
    from config import MAIN_ASSET_HOLDING_ADDRESS
    
    # First, get pools with current allocations (include them regardless of APY)
    allocated_pools_query = """
    SELECT DISTINCT pdm.pool_id
    FROM daily_balances db
    JOIN pool_daily_metrics pdm ON db.pool_id = pdm.pool_id
    WHERE db.date = CURRENT_DATE 
      AND (db.wallet_address = %s OR db.wallet_address IS NULL)
      AND db.allocated_balance > 0
      AND pdm.date = CURRENT_DATE
    """
    allocated_pools_df = pd.read_sql(allocated_pools_query, engine, params=(MAIN_ASSET_HOLDING_ADDRESS,))
    allocated_pool_ids = allocated_pools_df['pool_id'].tolist()
    
    # Build query to include both approved pools AND pools with current allocations
    if allocated_pool_ids:
        pool_ids_str = "', '".join(allocated_pool_ids)
        query = f"""
        SELECT
            pdm.pool_id,
            p.symbol,
            p.chain,
            p.protocol,
            pdm.forecasted_apy,
            pdm.forecasted_tvl,
            pdm.normalized_tokens
        FROM pool_daily_metrics pdm
        JOIN pools p ON pdm.pool_id = p.pool_id
        WHERE pdm.date = CURRENT_DATE 
          AND pdm.is_filtered_out = FALSE
          AND pdm.forecasted_apy IS NOT NULL
          AND pdm.forecasted_apy > 0
          AND pdm.forecasted_tvl IS NOT NULL
          AND pdm.forecasted_tvl > 0
          OR (pdm.pool_id IN ('{pool_ids_str}') 
              AND pdm.date = CURRENT_DATE
              AND pdm.forecasted_apy IS NOT NULL)
        """
    else:
        query = """
        SELECT
            pdm.pool_id,
            p.symbol,
            p.chain,
            p.protocol,
            pdm.forecasted_apy,
            pdm.forecasted_tvl,
            pdm.normalized_tokens
        FROM pool_daily_metrics pdm
        JOIN pools p ON pdm.pool_id = p.pool_id
        WHERE pdm.date = CURRENT_DATE 
          AND pdm.is_filtered_out = FALSE
          AND pdm.forecasted_apy IS NOT NULL
          AND pdm.forecasted_apy > 0
          AND pdm.forecasted_tvl IS NOT NULL
          AND pdm.forecasted_tvl > 0;
        """
    
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} pools ({len(allocated_pool_ids)} with current allocations)")
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
    
    # Create a mapping from lowercase to original case for proper return values
    token_mapping = {token.lower(): token for token in tokens}
    tokens_lower = list(token_mapping.keys())
    tokens_str = "','".join(tokens_lower)
    query = f"""
    WITH ranked_ohlcv AS (
        SELECT
            LOWER(symbol) as symbol_lower,
            CASE 
                WHEN raw_json_data ? 'USD' THEN (raw_json_data->'USD'->>'close')::float
                WHEN raw_json_data ? 'USDT' THEN (raw_json_data->'USDT'->>'close')::float
                WHEN raw_json_data ? 'BTC' THEN (raw_json_data->'BTC'->>'close')::float
                WHEN raw_json_data ? 'ETH' THEN (raw_json_data->'ETH'->>'close')::float
                ELSE NULL
            END as close_price,
            data_timestamp as ts,
            ROW_NUMBER() OVER(
                PARTITION BY LOWER(symbol) 
                ORDER BY data_timestamp DESC
            ) as rn
        FROM raw_coinmarketcap_ohlcv
        WHERE LOWER(symbol) IN ('{tokens_str}')
    )
    SELECT symbol_lower, close_price
    FROM ranked_ohlcv
    WHERE rn = 1;
    """
    df = pd.read_sql(query, engine)
    # Filter out NULL prices and create dictionary
    prices = {}
    for _, row in df.iterrows():
        if pd.notna(row['close_price']):
            # Map back to the original token name using our lowercase mapping
            original_token = token_mapping.get(row['symbol_lower'])
            if original_token:
                prices[original_token] = row['close_price']
    logger.info(f"Loaded prices for {len(prices)} tokens")
    return prices


def fetch_gas_fee_data(engine) -> Tuple[float, float, float, float, float]:
    """
    Fetches forecasted gas fee components and ETH price.
    
    Returns:
        Tuple of (eth_price_usd, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units)
    """
    # Fetch ETH price
    eth_price_query = """
    WITH ranked_eth AS (
        SELECT
            (raw_json_data->'USD'->>'close')::float as close_price,
            ROW_NUMBER() OVER(ORDER BY (raw_json_data->'USD'->>'timestamp')::timestamp DESC) as rn
        FROM raw_coinmarketcap_ohlcv
        WHERE symbol = 'ETH'
    )
    SELECT close_price
    FROM ranked_eth
    WHERE rn = 1;
    """
    eth_df = pd.read_sql(eth_price_query, engine)
    eth_price = eth_df['close_price'].iloc[0] if not eth_df.empty and pd.notna(eth_df['close_price'].iloc[0]) else 3000.0
    
    # Gas fee components based on requirements
    base_fee_transfer_gwei = 10.0  # Base fee for transfer/deposit
    base_fee_swap_gwei = 30.0       # Base fee for swap
    priority_fee_gwei = 10.0        # Priority fee
    min_gas_units = 21000          # Minimum gas units
    
    logger.info(f"ETH price: ${eth_price:.2f}")
    logger.info(f"Gas fee components - Base transfer: {base_fee_transfer_gwei} Gwei, Base swap: {base_fee_swap_gwei} Gwei, Priority: {priority_fee_gwei} Gwei, Min gas units: {min_gas_units}")
    
    return eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units


def calculate_gas_fee_usd(gas_units: float, base_fee_gwei: float, priority_fee_gwei: float, eth_price_usd: float) -> float:
    """
    Calculate gas fee in USD based on the formula: Gas fee = Gas units * (base fee + priority fee)
    
    Args:
        gas_units: Gas units (limit) for the transaction
        base_fee_gwei: Base fee in Gwei
        priority_fee_gwei: Priority fee in Gwei
        eth_price_usd: ETH price in USD
        
    Returns:
        Gas fee in USD
    """
    total_fee_gwei = gas_units * (base_fee_gwei + priority_fee_gwei)
    gas_fee_usd = total_fee_gwei * 1e-9 * eth_price_usd
    return gas_fee_usd


def calculate_transaction_gas_fees(eth_price_usd: float, base_fee_transfer_gwei: float, 
                                   base_fee_swap_gwei: float, priority_fee_gwei: float, 
                                   min_gas_units: float) -> Dict[str, float]:
    """
    Calculate gas fees for different transaction types.
    
    Args:
        eth_price_usd: ETH price in USD
        base_fee_transfer_gwei: Base fee for transfer/deposit in Gwei
        base_fee_swap_gwei: Base fee for swap in Gwei
        priority_fee_gwei: Priority fee in Gwei
        min_gas_units: Minimum gas units
        
    Returns:
        Dictionary with gas fees for different transaction types in USD
    """
    # Pool allocation/withdrawal gas fee (using transfer base fee)
    pool_transaction_gas_fee_usd = calculate_gas_fee_usd(
        min_gas_units, base_fee_transfer_gwei, priority_fee_gwei, eth_price_usd
    )
    
    # Token swap gas fee (using swap base fee)
    token_swap_gas_fee_usd = calculate_gas_fee_usd(
        min_gas_units, base_fee_swap_gwei, priority_fee_gwei, eth_price_usd
    )
    
    gas_fees = {
        'allocation': pool_transaction_gas_fee_usd,      # Allocating to pools
        'withdrawal': pool_transaction_gas_fee_usd,      # Withdrawing from pools
        'conversion': token_swap_gas_fee_usd,            # Token swaps/conversions
        'transfer': pool_transaction_gas_fee_usd,        # General transfers
        'deposit': pool_transaction_gas_fee_usd          # Deposits to pools
    }
    
    logger.info(f"Transaction gas fees - Pool Allocation/Withdrawal: ${pool_transaction_gas_fee_usd:.6f}, Token Swap/Conversion: ${token_swap_gas_fee_usd:.6f}")
    
    return gas_fees


def fetch_current_balances(engine) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """
    Fetches current token balances from warm wallet and allocated positions.
    Only queries today's data.
    
    Returns:
        Tuple of (warm_wallet_balances, current_allocations)
        - warm_wallet_balances: {token_symbol: amount}
        - current_allocations: {(pool_id, token_symbol): amount}
    """
    from config import MAIN_ASSET_HOLDING_ADDRESS
    
    if not MAIN_ASSET_HOLDING_ADDRESS:
        logger.error("MAIN_ASSET_HOLDING_ADDRESS not configured")
        return {}, {}
    
    warm_wallet = {}
    allocations = {}
    
    # Query today's data only
    # Include records with NULL wallet_address as they may contain the allocated assets
    query = """
    SELECT
        token_symbol,
        unallocated_balance,
        allocated_balance,
        pool_id
    FROM daily_balances
    WHERE date = CURRENT_DATE AND (wallet_address = %s OR wallet_address IS NULL);
    """
    
    try:
        df = pd.read_sql(query, engine, params=(MAIN_ASSET_HOLDING_ADDRESS,))
        
        if df.empty:
            logger.info(f"No balance data found for wallet {MAIN_ASSET_HOLDING_ADDRESS} today")
            return {}, {}
            
        logger.info(f"Using today's data for wallet {MAIN_ASSET_HOLDING_ADDRESS}")
        
    except Exception as e:
        logger.error(f"Error fetching balance data: {e}")
        return {}, {}
    
    # Process the data
    for _, row in df.iterrows():
        token = row['token_symbol']
        
        # Unallocated balance in warm wallet (handle NULL values properly)
        if pd.notna(row['unallocated_balance']) and row['unallocated_balance'] > 0:
            warm_wallet[token] = warm_wallet.get(token, 0) + float(row['unallocated_balance'])
        
        # Allocated balance to pools (handle NULL values properly)
        if pd.notna(row['allocated_balance']) and row['allocated_balance'] > 0 and pd.notna(row['pool_id']):
            key = (row['pool_id'], token)
            allocations[key] = allocations.get(key, 0) + float(row['allocated_balance'])
    
    logger.info(f"Warm wallet: {len(warm_wallet)} tokens, Total allocated positions: {len(allocations)}")
    return warm_wallet, allocations


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
    logger.info(f"Loaded allocation parameters: max_alloc={params.get('max_alloc_percentage')}, tvl_limit={params.get('tvl_limit_percentage')}")
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


def parse_pool_tokens_with_mapping(symbol: str, normalized_tokens_json: str = None) -> List[str]:
    """
    Extracts tokens from pool symbol and applies normalized mappings if available.
    
    Args:
        symbol: Pool symbol (e.g., "DAI-USDC-USDT" or "GTUSDC")
        normalized_tokens_json: JSON string containing token mappings from filter_pools_pre
        
    Returns:
        List of token symbols in uppercase, with partial matches replaced by approved tokens
    """
    tokens = [t.upper().strip() for t in symbol.split('-')]
    
    # Apply normalized mappings if available
    if normalized_tokens_json:
        try:
            token_mappings = json.loads(normalized_tokens_json)
            # Replace tokens with their mapped approved tokens
            normalized_tokens = []
            for token in tokens:
                # Check if this token has a mapping (case-insensitive)
                mapped_token = None
                for original, approved in token_mappings.items():
                    if token.lower() == original.lower():
                        mapped_token = approved.upper()
                        break
                normalized_tokens.append(mapped_token if mapped_token else token)
            return normalized_tokens
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse normalized_tokens_json: {e}")
    
    return tokens


def calculate_aum(warm_wallet: Dict[str, float], 
                  current_allocations: Dict[Tuple[str, str], float],
                  token_prices: Dict[str, float]) -> float:
    """
    Calculates total Assets Under Management in USD.
    
    Args:
        warm_wallet: Unallocated token balances
        current_allocations: Allocated positions
        token_prices: Token prices in USD
        
    Returns:
        Total AUM in USD
    """
    total_usd = 0.0
    
    # Warm wallet value
    for token, amount in warm_wallet.items():
        price = token_prices.get(token, 1.0)  # Default to $1 for stablecoins
        total_usd += amount * price
    
    # Allocated positions value
    for (pool_id, token), amount in current_allocations.items():
        price = token_prices.get(token, 1.0)
        total_usd += amount * price
    
    logger.info(f"Total AUM: ${total_usd:,.2f}")
    return total_usd


def build_token_universe(pools_df: pd.DataFrame, 
                         warm_wallet: Dict[str, float],
                         current_allocations: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Builds the complete set of tokens needed for optimization.
    
    Returns:
        Sorted list of unique token symbols
    """
    tokens = set()
    
    # Tokens from pools (using normalized mappings if available)
    for _, row in pools_df.iterrows():
        symbol = row['symbol']
        normalized_tokens_json = row.get('normalized_tokens')
        pool_tokens = parse_pool_tokens_with_mapping(symbol, normalized_tokens_json)
        tokens.update(pool_tokens)
    
    # Tokens in warm wallet
    tokens.update(warm_wallet.keys())
    
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
                 warm_wallet: Dict[str, float], current_allocations: Dict[Tuple[str, str], float],
                 gas_fees: Dict[str, float], alloc_params: Dict):
        """
        Initialize the optimizer.
        
        Args:
            pools_df: DataFrame of approved pools
            token_prices: Token price dictionary
            warm_wallet: Current warm wallet balances
            current_allocations: Current pool allocations
            gas_fees: Dictionary of gas fees for different transaction types in USD
            alloc_params: Allocation parameters
        """
        self.pools_df = pools_df
        self.token_prices = token_prices
        self.warm_wallet = warm_wallet
        self.current_allocations = current_allocations
        self.gas_fees = gas_fees
        self.alloc_params = alloc_params
        
        # Extract specific gas fees for convenience
        self.allocation_gas_fee = gas_fees['allocation']
        self.withdrawal_gas_fee = gas_fees['withdrawal']
        self.conversion_gas_fee = gas_fees['conversion']
        
        # Build indices
        self.tokens = build_token_universe(pools_df, warm_wallet, current_allocations)
        self.pools = pools_df['pool_id'].tolist()
        
        self.n_tokens = len(self.tokens)
        self.n_pools = len(self.pools)
        
        self.token_idx = {t: i for i, t in enumerate(self.tokens)}
        self.pool_idx = {p: i for i, p in enumerate(self.pools)}
        
        # Build pool-token mapping
        self.pool_tokens = {}  # pool_id -> list of token symbols
        self.pool_tvl = {}     # pool_id -> forecasted_tvl
        for idx, row in pools_df.iterrows():
            pool_id = row['pool_id']
            normalized_tokens_json = row.get('normalized_tokens')
            tokens = parse_pool_tokens_with_mapping(row['symbol'], normalized_tokens_json)
            self.pool_tokens[pool_id] = tokens
            self.pool_tvl[pool_id] = row['forecasted_tvl']
        
        # Calculate AUM
        self.total_aum = calculate_aum(warm_wallet, current_allocations, token_prices)
        
        # Constants
        self.conversion_rate = alloc_params.get('conversion_rate', 0.0004)
        self.min_transaction_value = alloc_params.get('min_transaction_value', 50.0)
        self.max_alloc_percentage = alloc_params.get('max_alloc_percentage', 0.20)
        self.tvl_limit_percentage = alloc_params.get('tvl_limit_percentage', 0.05) or 0.05
        
        logger.info(f"Optimizer initialized: {self.n_pools} pools, {self.n_tokens} tokens, AUM=${self.total_aum:,.2f}")
        logger.info(f"Parameters: max_alloc={self.max_alloc_percentage:.1%}, tvl_limit={self.tvl_limit_percentage:.1%}, conversion_rate={self.conversion_rate:.4%}")
    
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
        
        # Final warm wallet balance for each token
        self.final_warm_wallet = cp.Variable(self.n_tokens, nonneg=True)
        
        # Binary variables to track whether conversion is needed for each allocation
        # needs_conversion[i,j] = 1 if we need to convert token j for allocation to pool i
        self.needs_conversion = cp.Variable((self.n_pools, self.n_tokens), boolean=True)
        
        # Binary variables to track if there's an allocation to each pool
        # has_allocation[i] = 1 if we're allocating any amount to pool i
        self.has_allocation = cp.Variable(self.n_pools, boolean=True)
        
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
        
        # Current warm wallet vector
        warm_wallet_vector = np.zeros(self.n_tokens)
        for token, amount in self.warm_wallet.items():
            if token in self.token_idx:
                j = self.token_idx[token]
                warm_wallet_vector[j] = amount
        
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
            # Daily APY is directly calculated from the stored percentage
            # Stored APY is already a percentage (e.g., 1.2 means 1.2%), so convert to daily rate
            daily_apy = row['forecasted_apy'] / 100.0 / 365.0
            
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
        
        # Transaction costs with correct formulas
        # Create binary variables for withdrawals (is_withdrawal[i,j] = 1 if withdraw[i,j] > 0)
        self.is_withdrawal = cp.Variable((self.n_pools, self.n_tokens), boolean=True)
        
        # Create binary variables for conversions (is_conversion[i,j] = 1 if convert[i,j] > 0)
        self.is_conversion = cp.Variable((self.n_tokens, self.n_tokens), boolean=True)
        
        # 1. Withdrawal costs: ONLY gas_fee (no conversion costs for withdrawals)
        withdrawal_gas_costs = cp.sum(cp.multiply(
            self.is_withdrawal,
            self.withdrawal_gas_fee
        ))
        
        # 2. Allocation costs: ONLY gas_fee (no conversion costs for allocations)
        # Gas costs for allocations (only allocation gas fee, no conversion involved)
        allocation_gas_costs = cp.sum(cp.multiply(
            self.needs_conversion,  # Binary tracking if allocation exists
            self.allocation_gas_fee
        ))
        
        # 3. Conversion costs: amount * conversion_rate + gas_fee for each actual conversion
        conversion_conversion_costs = cp.sum(cp.multiply(
            cp.multiply(self.convert, price_vector),
            self.conversion_rate
        ))
        
        conversion_gas_costs = cp.sum(cp.multiply(
            self.is_conversion,
            self.conversion_gas_fee
        ))
        
        # Total transaction costs
        self.total_transaction_costs = (
            withdrawal_gas_costs +
            allocation_gas_costs +
            conversion_conversion_costs + 
            conversion_gas_costs
        )
        
        # IMPROVED OBJECTIVE FUNCTION:
        # Maximize NET yield improvement: yield from new allocations - yield lost from withdrawals
        # This incentivizes reallocation from lower-APY to higher-APY pools
        
        # Calculate yield lost from withdrawals
        withdrawal_yield_loss = cp.sum(cp.multiply(
            cp.multiply(self.withdraw, daily_apy_matrix),
            price_vector
        ))
        
        # Net yield improvement = new yield - yield lost from withdrawals
        net_yield_improvement = yield_usd - withdrawal_yield_loss
        
        # Maximize net yield improvement
        objective = cp.Maximize(net_yield_improvement)
        
        # ====================================================================
        # CONSTRAINTS
        # ====================================================================
        
        constraints = []
        
        # 1. Token balance conservation
        # For each token: initial_warm_wallet + withdrawals + conversions_in 
        #                 = final_warm_wallet + allocations + conversions_out
        for j in range(self.n_tokens):
            token_in = (
                warm_wallet_vector[j] +  # Initial warm wallet
                cp.sum(self.withdraw[:, j]) +  # Withdrawals from pools
                cp.sum(self.convert[:, j])  # Conversions into this token
            )
            token_out = (
                self.final_warm_wallet[j] +  # Final warm wallet
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
            
            # TVL limit constraint: allocation cannot exceed tvl_limit_percentage of pool's forecasted TVL
            pool_id = self.pools[i]
            pool_forecasted_tvl = self.pool_tvl.get(pool_id, 0)
            if pool_forecasted_tvl > 0:
                constraints.append(pool_total_usd <= self.tvl_limit_percentage * pool_forecasted_tvl)
        
        # 6. AUM CONSERVATION CONSTRAINT (IMPROVED):
        # Total allocated amount + transaction costs + final warm wallet <= total AUM
        # This ensures proper accounting where costs are actually deducted from available capital
        total_allocated_usd = cp.sum(cp.multiply(self.alloc, price_vector))
        total_final_warm_wallet_usd = cp.sum(cp.multiply(self.final_warm_wallet, price_vector))
        
        # The sum of allocations, costs, and remaining warm wallet must equal initial AUM
        constraints.append(
            total_allocated_usd + self.total_transaction_costs + total_final_warm_wallet_usd <= self.total_aum
        )
        
        # Alternative constraint to minimize unallocated funds (optional - can be enabled/disabled)
        # This pushes the optimizer to allocate as much as possible after accounting for costs
        # constraints.append(total_final_warm_wallet_usd <= 0.01 * self.total_aum)  # Allow max 1% unallocated
        
        # 7. Link conversion needs to available balances
        # For each allocation, determine if conversion is needed based on available warm wallet balance
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                # If we need to allocate more than we have in warm wallet, we need conversion
                # This is a simplified approach - in practice we'd need to track the sequence more carefully
                # needs_conversion[i,j] should be 1 if alloc[i,j] > warm_wallet[j] after accounting for withdrawals
                constraints.append(
                    self.needs_conversion[i, j] >= 
                    (self.alloc[i, j] - warm_wallet_vector[j] - cp.sum(self.withdraw[:, j])) / self.total_aum
                )
                # Ensure needs_conversion is binary (0 or 1)
                constraints.append(self.needs_conversion[i, j] <= 1)
        
        # 8. Link has_allocation to actual allocations
        for i in range(self.n_pools):
            # has_allocation[i] should be 1 if any token is allocated to pool i
            total_pool_allocation = cp.sum(self.alloc[i, :])
            # Big-M formulation - if total allocation > 0, then has_allocation = 1
            big_M = self.total_aum  # Upper bound on any allocation
            constraints.append(total_pool_allocation <= big_M * self.has_allocation[i])
            constraints.append(total_pool_allocation >= 0.01 * self.has_allocation[i])  # Small threshold to avoid numerical issues
        
        # 9. Link binary variables to transaction amounts
        # Withdrawal binary variables
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                # Big-M formulation for withdrawal binary variables
                big_M_withdraw = current_alloc_matrix[i, j] + self.total_aum  # Upper bound
                constraints.append(self.withdraw[i, j] <= big_M_withdraw * self.is_withdrawal[i, j])
                constraints.append(self.withdraw[i, j] >= 0.01 * self.is_withdrawal[i, j])  # Small threshold
        
        # Conversion binary variables
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                if i != j:  # Skip self-conversions
                    # Big-M formulation for conversion binary variables
                    big_M_convert = self.total_aum  # Upper bound
                    constraints.append(self.convert[i, j] <= big_M_convert * self.is_conversion[i, j])
                    constraints.append(self.convert[i, j] >= 0.01 * self.is_conversion[i, j])  # Small threshold
        
        # 10. Non-negativity (already enforced by variable definition, but adding explicitly)
        constraints.append(self.alloc >= 0)
        constraints.append(self.withdraw >= 0)
        constraints.append(self.convert >= 0)
        constraints.append(self.final_warm_wallet >= 0)
        
        logger.info(f"Model built with {len(constraints)} constraints")
        
        problem = cp.Problem(objective, constraints)
        return problem
    
    def solve(self, solver=cp.HIGHS, verbose=True) -> bool:
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
                    # Calculate withdrawal cost: ONLY gas_fee (no conversion costs)
                    withdrawal_cost = self.withdrawal_gas_fee
                    transactions.append({
                        'seq': transaction_seq,
                        'type': 'WITHDRAWAL',
                        'from_location': pool_id,
                        'to_location': 'warm_wallet',
                        'token': token,
                        'amount': amount,
                        'amount_usd': amount * price_vector[j],
                        'gas_cost_usd': self.withdrawal_gas_fee,
                        'conversion_cost_usd': 0.0,  # No conversion cost for withdrawals
                        'total_cost_usd': withdrawal_cost
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
                        # Calculate conversion cost: amount * conversion_rate + conversion_gas_fee
                        conversion_cost = amount * price_vector[i] * self.conversion_rate + self.conversion_gas_fee
                        transactions.append({
                            'seq': transaction_seq,
                            'type': 'CONVERSION',
                            'from_location': 'warm_wallet',
                            'to_location': 'warm_wallet',
                            'from_token': from_token,
                            'to_token': to_token,
                            'amount': amount,
                            'amount_usd': amount * price_vector[i],
                            'conversion_cost_usd': amount * price_vector[i] * self.conversion_rate,
                            'gas_cost_usd': self.conversion_gas_fee,
                            'total_cost_usd': conversion_cost
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
                    
                    # Check if conversion is needed for this allocation
                    needs_conversion = self.needs_conversion.value[i, j] if self.needs_conversion.value is not None else 0
                    
                    # Calculate allocation cost: ONLY gas_fee (no conversion costs)
                    # Allocations don't incur conversion costs - only the separate CONVERSION transactions do
                    gas_cost = self.allocation_gas_fee
                    total_cost = gas_cost
                    
                    allocations.append({
                        'pool_id': pool_id,
                        'pool_symbol': symbol,
                        'token': token,
                        'amount': amount,
                        'amount_usd': amount_usd,
                        'needs_conversion': bool(needs_conversion > 0.5)
                    })
                    
                    transactions.append({
                        'seq': transaction_seq,
                        'type': 'ALLOCATION',
                        'from_location': 'warm_wallet',
                        'to_location': pool_id,
                        'token': token,
                        'amount': amount,
                        'amount_usd': amount_usd,
                        'conversion_cost_usd': 0.0,  # No conversion cost for allocations
                        'gas_cost_usd': gas_cost,
                        'total_cost_usd': total_cost,
                        'needs_conversion': bool(needs_conversion > 0.5)
                    })
                    transaction_seq += 1
        
        allocations_df = pd.DataFrame(allocations)
        
        logger.info(f"Extracted {len(allocations)} allocations and {len(transactions)} transactions")
        
        return allocations_df, transactions
    
    def format_results(self) -> Dict:
        """
        Formats optimization results to match the requirements in optimization.md.
        
        Returns:
            Dictionary with final_allocations, unallocated_tokens, and transactions
        """
        if self.alloc.value is None:
            logger.error("No solution available to format")
            return {
                "final_allocations": {},
                "unallocated_tokens": {},
                "transactions": []
            }
        
        # Get allocations and transactions
        allocations_df, transactions = self.extract_results()
        
        # Initialize result structure
        final_allocations = {}
        unallocated_tokens = {}
        price_vector = np.array([self.token_prices.get(t, 1.0) for t in self.tokens])
        
        # ====================================================================
        # PROCESS FINAL ALLOCATIONS
        # ====================================================================
        
        for _, allocation in allocations_df.iterrows():
            pool_id = allocation['pool_id']
            pool_symbol = allocation['pool_symbol']
            token = allocation['token']
            amount = allocation['amount']
            amount_usd = allocation['amount_usd']
            
            # Initialize pool if not exists
            if pool_id not in final_allocations:
                final_allocations[pool_id] = {
                    "pool_symbol": pool_symbol,
                    "tokens": {}
                }
            
            # Add token to pool
            final_allocations[pool_id]["tokens"][token] = {
                "amount": amount,
                "amount_usd": amount_usd
            }
        
        # ====================================================================
        # PROCESS UNALLOCATED TOKENS (final warm wallet balances)
        # ====================================================================
        
        for j in range(self.n_tokens):
            token = self.tokens[j]
            amount = self.final_warm_wallet.value[j] if self.final_warm_wallet.value is not None else 0
            
            if amount > 0.01:  # Only include meaningful amounts
                amount_usd = amount * price_vector[j]
                unallocated_tokens[token] = {
                    "amount": amount,
                    "amount_usd": amount_usd
                }
        
        # ====================================================================
        # FORMAT TRANSACTIONS
        # ====================================================================
        
        # Ensure transactions are properly formatted with all required fields
        formatted_transactions = []
        for txn in transactions:
            formatted_txn = {
                "seq": txn["seq"],
                "type": txn["type"],
                "from_location": txn["from_location"],
                "to_location": txn["to_location"],
                "amount": txn["amount"],
                "amount_usd": txn["amount_usd"],
                "gas_cost_usd": txn["gas_cost_usd"]
            }
            
            # Add token field for non-conversion transactions
            if "token" in txn:
                formatted_txn["token"] = txn["token"]
            
            # Add conversion-specific fields
            if txn["type"] == "CONVERSION":
                formatted_txn["from_token"] = txn["from_token"]
                formatted_txn["to_token"] = txn["to_token"]
                formatted_txn["conversion_cost_usd"] = txn.get("conversion_cost_usd", 0)
            
            # Add cost breakdown fields for all transaction types
            if "conversion_cost_usd" in txn:
                formatted_txn["conversion_cost_usd"] = txn["conversion_cost_usd"]
            if "total_cost_usd" in txn:
                formatted_txn["total_cost_usd"] = txn["total_cost_usd"]
            if "needs_conversion" in txn:
                formatted_txn["needs_conversion"] = txn["needs_conversion"]
            
            formatted_transactions.append(formatted_txn)
        
        result = {
            "final_allocations": final_allocations,
            "unallocated_tokens": unallocated_tokens,
            "transactions": formatted_transactions
        }
        
        logger.info(f"Formatted results: {len(final_allocations)} pools, "
                   f"{len(unallocated_tokens)} unallocated tokens, "
                   f"{len(formatted_transactions)} transactions")
        
        return result


# ============================================================================
# RESULT PERSISTENCE
# ============================================================================

def delete_todays_allocations(engine):
    """
    Deletes all asset allocations for the current date to ensure only one set exists per day.
    
    Args:
        engine: Database engine
    """
    conn = engine.raw_connection()
    cursor = conn.cursor()
    
    try:
        # Delete all allocations for today's date
        cursor.execute("""
            DELETE FROM asset_allocations 
            WHERE DATE(timestamp) = CURRENT_DATE;
        """)
        
        deleted_rows = cursor.rowcount
        conn.commit()
        logger.info(f"Deleted {deleted_rows} existing allocation records for today")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting today's allocations: {e}")
        raise
    finally:
        cursor.close()


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
        # Convert NumPy types to native Python types to avoid PostgreSQL errors
        max_alloc = alloc_params.get('max_alloc_percentage')
        conversion_rate = alloc_params.get('conversion_rate')
        
        max_alloc = float(max_alloc) if hasattr(max_alloc, 'dtype') else max_alloc
        conversion_rate = float(conversion_rate) if hasattr(conversion_rate, 'dtype') else conversion_rate
        
        cursor.execute("""
            INSERT INTO allocation_parameters (
                run_id, timestamp, max_alloc_percentage, conversion_rate
            ) VALUES (%s, %s, %s, %s);
        """, (
            run_id,
            datetime.now(timezone.utc),
            max_alloc,
            conversion_rate
        ))
        
        # Store transaction sequence
        for txn in transactions:
            # Convert NumPy types to native Python types to avoid PostgreSQL errors
            amount = float(txn['amount']) if hasattr(txn['amount'], 'dtype') else txn['amount']
            
            cursor.execute("""
                INSERT INTO asset_allocations (
                    run_id, step_number, operation, from_asset, to_asset, 
                    amount, pool_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (
                run_id,
                int(txn['seq']) if hasattr(txn['seq'], 'dtype') else txn['seq'],
                txn['type'],
                txn.get('from_token', txn.get('token')),
                txn.get('to_token', txn.get('token')),
                amount,
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
        
        # Generate Data Quality Report
        logger.info("\n[0/5] Generating data quality report...")
        try:
            quality_report = generate_data_quality_report()
            quality_score = quality_report.get('summary', {}).get('overall_quality_score', 0)
            quality_assessment = quality_report.get('summary', {}).get('quality_assessment', 'Unknown')
            logger.info(f"Data quality assessment: {quality_assessment} (Score: {quality_score:.1f}/100)")
            
            # Check for critical issues that might prevent optimization
            critical_issues = []
            for category, items in quality_report.get('abnormal_values', {}).items():
                critical_items = [i for i in items if i['severity'] == 'critical']
                if critical_items:
                    critical_issues.extend([f"{category}: {i['reason']}" for i in critical_items])
            
            if critical_issues:
                logger.warning(f"Found {len(critical_issues)} critical data quality issues:")
                for issue in critical_issues[:5]:  # Show first 5
                    logger.warning(f"  - {issue}")
                if len(critical_issues) > 5:
                    logger.warning(f"  ... and {len(critical_issues) - 5} more")
            
            # Check model feasibility
            model_feasibility = quality_report.get('model_feasibility', {})
            if not model_feasibility.get('constraint_feasible', True):
                logger.error("Model feasibility check failed - optimization may not succeed")
                if model_feasibility.get('aum_exceeds_capacity', False):
                    logger.error("  - Total AUM exceeds pool capacity")
                if model_feasibility.get('tokens_without_prices', 0) > 5:
                    logger.error("  - Too many tokens without price data")
            
        except Exception as e:
            logger.error(f"Failed to generate data quality report: {e}")
            logger.warning("Proceeding with optimization without quality assessment")
        
        # Load data
        logger.info("\n[1/5] Loading data...")
        pools_df = fetch_pool_data(engine)
        
        if pools_df.empty:
            logger.warning("No approved pools available. Exiting.")
            return
        
        # Build token universe
        warm_wallet, current_allocations = fetch_current_balances(engine)
        tokens = build_token_universe(pools_df, warm_wallet, current_allocations)
        
        # Fetch prices and gas fees
        token_prices = fetch_token_prices(engine, tokens + ['ETH'])
        eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units = fetch_gas_fee_data(engine)
        gas_fees = calculate_transaction_gas_fees(eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units)
        
        # Fetch parameters
        alloc_params = fetch_allocation_parameters(engine)
        
        # Initialize optimizer
        logger.info("\n[3/6] Initializing optimizer...")
        optimizer = AllocationOptimizer(
            pools_df=pools_df,
            token_prices=token_prices,
            warm_wallet=warm_wallet,
            current_allocations=current_allocations,
            gas_fees=gas_fees,
            alloc_params=alloc_params
        )
        
        # Solve optimization
        logger.info("\n[4/6] Solving optimization problem...")
        import cvxpy as cp
        success = False
        
        # Try different solvers in order of preference
        for solver_name in ['HIGHS', 'CBC', 'SCIPY']:
            try:
                solver = getattr(cp, solver_name)
                logger.info(f"\nAttempting to solve with {solver_name}...")
                start_time = time.time()
                success = optimizer.solve(solver=solver, verbose=True)
                solve_time = time.time() - start_time
                
                if success:
                    logger.info(f"✓ Solved with {solver_name} in {solve_time:.3f} seconds")
                    break
            except Exception as e:
                logger.warning(f"{solver_name} solver failed: {e}")
                continue
        
        if not success:
            logger.error("✗ Could not solve optimization problem with available solvers")
            logger.error("Optimization failed")
            return
        
        # Extract and format results
        logger.info("\n[5/6] Extracting and formatting results...")
        formatted_results = optimizer.format_results()
        
        # Calculate yield improvement
        logger.info("\nYIELD IMPROVEMENT ANALYSIS:")
        logger.info("-" * 50)
        
        # Calculate current daily yield from existing allocations
        current_daily_yield = 0.0
        for (pool_id, token), amount in current_allocations.items():
            # Get APY for this pool
            pool_data = pools_df[pools_df['pool_id'] == pool_id]
            if not pool_data.empty:
                apy = pool_data['forecasted_apy'].iloc[0]
                token_price = token_prices.get(token, 1.0)
                usd_value = amount * token_price
                current_daily_yield += usd_value * apy / 100 / 365
        
        # Calculate optimized daily yield from new allocations
        optimized_daily_yield = 0.0
        for pool_id, pool_data in formatted_results["final_allocations"].items():
            pool_info = pools_df[pools_df['pool_id'] == pool_id]
            if not pool_info.empty:
                apy = pool_info['forecasted_apy'].iloc[0]
                for token_data in pool_data["tokens"].values():
                    optimized_daily_yield += token_data['amount_usd'] * apy / 100 / 365
        
        # Calculate improvements
        daily_improvement = optimized_daily_yield - current_daily_yield
        annual_improvement = daily_improvement * 365
        improvement_percentage = (daily_improvement / current_daily_yield * 100) if current_daily_yield > 0 else 0
        
        logger.info(f"Current daily yield: ${current_daily_yield:.2f}")
        logger.info(f"Optimized daily yield: ${optimized_daily_yield:.2f}")
        logger.info(f"Daily improvement: ${daily_improvement:.2f} ({improvement_percentage:+.1f}%)")
        logger.info(f"Annualized improvement: ${annual_improvement:,.2f}")
        
        # Print results in the required format
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 80)
        
        # Print final allocations
        logger.info("\nFINAL ALLOCATIONS:")
        for pool_id, pool_data in formatted_results["final_allocations"].items():
            logger.info(f"\nPool: {pool_id} ({pool_data['pool_symbol']})")
            for token, token_data in pool_data["tokens"].items():
                logger.info(f"  {token}: {token_data['amount']:,.2f} (${token_data['amount_usd']:,.2f})")
        
        # Print unallocated tokens
        logger.info("\nUNALLOCATED TOKENS (in warm wallet):")
        for token, token_data in formatted_results["unallocated_tokens"].items():
            logger.info(f"  {token}: {token_data['amount']:,.2f} (${token_data['amount_usd']:,.2f})")
        
        # Print transaction sequence
        logger.info("\nTRANSACTION SEQUENCE:")
        for txn in formatted_results["transactions"]:
            if txn["type"] == "CONVERSION":
                logger.info(f"  {txn['seq']:3d}. {txn['type']:12s} | {txn['from_token']} → {txn['to_token']} | "
                           f"${txn['amount_usd']:10,.2f} | Gas: ${txn['gas_cost_usd']:6.4f} | "
                           f"Conv: ${txn.get('conversion_cost_usd', 0):.4f} | Total: ${txn.get('total_cost_usd', 0):.4f}")
            elif txn["type"] == "ALLOCATION":
                conv_flag = " (conv)" if txn.get('needs_conversion', False) else ""
                # Get pool name for allocation
                pool_id = txn.get('to_location', '')
                pool_name = ''
                if pool_id and pool_id in optimizer.pools_df['pool_id'].values:
                    pool_name = optimizer.pools_df[optimizer.pools_df['pool_id'] == pool_id]['symbol'].iloc[0]
                    pool_name = f" ({pool_name})"
                logger.info(f"  {txn['seq']:3d}. {txn['type']:12s} | {txn.get('token', '')}{conv_flag} → Pool {pool_id}{pool_name} | "
                           f"${txn['amount_usd']:10,.2f} | Gas: ${txn['gas_cost_usd']:6.4f} | "
                           f"Conv: ${txn.get('conversion_cost_usd', 0):.4f} | Total: ${txn.get('total_cost_usd', 0):.4f}")
            else:  # WITHDRAWAL
                # Get pool name for withdrawal
                pool_id = txn.get('from_location', '')
                pool_name = ''
                if pool_id and pool_id in optimizer.pools_df['pool_id'].values:
                    pool_name = optimizer.pools_df[optimizer.pools_df['pool_id'] == pool_id]['symbol'].iloc[0]
                    pool_name = f" ({pool_name})"
                logger.info(f"  {txn['seq']:3d}. {txn['type']:12s} | Pool {pool_id}{pool_name} → {txn.get('token', '')} | "
                           f"${txn['amount_usd']:10,.2f} | Gas: ${txn['gas_cost_usd']:6.4f} | "
                           f"Conv: ${txn.get('conversion_cost_usd', 0):.4f} | Total: ${txn.get('total_cost_usd', 0):.4f}")
        
        # Store results
        logger.info("\n[6/6] Storing results...")
        
        # First delete any existing allocations for today to ensure only one set exists
        logger.info("Deleting any existing allocations for today...")
        delete_todays_allocations(engine)
        
        run_id = str(uuid4())
        
        # Convert back to original format for storage
        allocations_df = pd.DataFrame([
            {
                'pool_id': pool_id,
                'pool_symbol': pool_data['pool_symbol'],
                'token': token,
                'amount': token_data['amount'],
                'amount_usd': token_data['amount_usd']
            }
            for pool_id, pool_data in formatted_results["final_allocations"].items()
            for token, token_data in pool_data["tokens"].items()
        ])
        
        store_results(engine, run_id, allocations_df, formatted_results["transactions"], alloc_params)
        
        # Save results to JSON file (DISABLED)
        # results_file = f"optimization_results_{run_id}.json"
        # with open(results_file, 'w') as f:
        #     json.dump(formatted_results, f, indent=2, default=str)
        # logger.info(f"Results saved to {results_file}")
        logger.info("Results JSON file saving is disabled")
        
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