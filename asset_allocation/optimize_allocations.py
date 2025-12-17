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
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
from uuid import uuid4
from typing import Dict, List, Tuple, TYPE_CHECKING


from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.daily_balance_repository import DailyBalanceRepository
from database.repositories.raw_data_repository import RawDataRepository
from database.repositories.parameter_repository import ParameterRepository
from database.repositories.allocation_repository import AllocationRepository
from asset_allocation.data_quality_report import generate_data_quality_report

# Lazy import for heavy optimization library - imported inside functions to reduce cold start time
# Type hints for lazy-loaded modules (for IDE support without runtime import)
if TYPE_CHECKING:
    import cvxpy as cp

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

def fetch_pool_data() -> pd.DataFrame:
    """
    Fetches approved pools with forecasted APY and metadata.
    Includes pools with current allocations even if they're below the APY limit or inactive.
    Only includes active pools (is_active = TRUE) for new allocations.
    
    Returns:
        DataFrame with columns: pool_id, symbol, chain, protocol, forecasted_apy, forecasted_tvl, underlying_tokens
    """
    from config import MAIN_ASSET_HOLDING_ADDRESS
    
    metrics_repo = PoolMetricsRepository()
    balance_repo = DailyBalanceRepository()
    
    # First, get pools with current allocations
    allocated_pool_ids = []
    if MAIN_ASSET_HOLDING_ADDRESS:
        allocated_pool_ids = balance_repo.get_allocated_pool_ids(MAIN_ASSET_HOLDING_ADDRESS, date.today())
    
    # Fetch pool candidates using repository
    rows = metrics_repo.get_pool_candidates_for_optimization(date.today(), allocated_pool_ids)
    
    if not rows:
        logger.warning(f"No pools found for optimization (Allocated pools: {len(allocated_pool_ids)})")
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=[
        'pool_id', 'symbol', 'chain', 'protocol', 'forecasted_apy', 'forecasted_tvl', 'underlying_tokens'
    ])
    
    logger.info(f"Loaded {len(df)} pools ({len(allocated_pool_ids)} with current allocations)")
    
    # Normalize underlying tokens using approved tokens mapping
    from database.repositories.token_repository import TokenRepository
    token_repo = TokenRepository()
    approved_tokens = token_repo.get_approved_tokens()
    
    # Create mapping: address (lowercase) -> symbol
    addr_to_symbol = {}
    for t in approved_tokens:
        if t.token_address:
            addr_to_symbol[t.token_address.lower()] = t.token_symbol
            
    # Normalize tokens in dataframe
    normalized_tokens_list = []
    for _, row in df.iterrows():
        raw_tokens = row.get('underlying_tokens')
        pool_tokens = []
        
        # Parse if string
        if isinstance(raw_tokens, str):
            try:
                pool_tokens = json.loads(raw_tokens)
            except:
                pool_tokens = []
        elif isinstance(raw_tokens, list):
            pool_tokens = raw_tokens
            
        # Normalize
        normalized_pool_tokens = []
        if pool_tokens:
            for t in pool_tokens:
                t_lower = t.lower() if isinstance(t, str) else str(t).lower()
                # Check if it's an address in our map
                if t_lower in addr_to_symbol:
                    normalized_pool_tokens.append(addr_to_symbol[t_lower])
                else:
                    # Keep original if no map found (might already be a symbol)
                    normalized_pool_tokens.append(t)
        
        normalized_tokens_list.append(normalized_pool_tokens)
        
    df['underlying_tokens'] = normalized_tokens_list
    
    return df


def fetch_token_prices(tokens: List[str]) -> Dict[str, float]:
    """
    Fetches latest closing prices for given tokens.
    
    Args:
        tokens: List of token symbols
        
    Returns:
        Dictionary mapping token symbol to USD price
    """
    repo = RawDataRepository()
    prices = repo.get_latest_prices(tokens)
    logger.info(f"Loaded prices for {len(prices)} tokens")
    return prices


def fetch_gas_fee_data() -> Tuple[float, float, float, float, float]:
    """
    Fetches forecasted gas fee components and ETH price.
    
    Returns:
        Tuple of (eth_price_usd, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units)
    """
    repo = RawDataRepository()
    
    # Fetch ETH price
    eth_prices = repo.get_latest_prices(['ETH'])
    eth_price = eth_prices.get('ETH', 3000.0)
    
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
    """
    total_fee_gwei = gas_units * (base_fee_gwei + priority_fee_gwei)
    gas_fee_usd = total_fee_gwei * 1e-9 * eth_price_usd
    return gas_fee_usd


def calculate_transaction_gas_fees(eth_price_usd: float, base_fee_transfer_gwei: float, 
                                   base_fee_swap_gwei: float, priority_fee_gwei: float, 
                                   min_gas_units: float) -> Dict[str, float]:
    """
    Calculate gas fees for different transaction types.
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


def fetch_current_balances() -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """
    Fetches current token balances from warm wallet and allocated positions.
    Only queries today's data.
    
    Returns:
        Tuple of (warm_wallet_balances, current_allocations)
        - warm_wallet_balances: {token_symbol: amount}
        - current_allocations: {(pool_id, token_symbol): amount}
    """
    from config import MAIN_ASSET_HOLDING_ADDRESS
    
    repo = DailyBalanceRepository()
    
    if not MAIN_ASSET_HOLDING_ADDRESS:
        logger.error("MAIN_ASSET_HOLDING_ADDRESS not configured")
        return {}, {}
    
    warm_wallet = {}
    allocations = {}
    
    try:
        # Query today's data only
        balances = repo.get_current_balances(MAIN_ASSET_HOLDING_ADDRESS, date.today())
        
        if not balances:
            logger.info(f"No balance data found for wallet {MAIN_ASSET_HOLDING_ADDRESS} today")
            return {}, {}
            
        logger.info(f"Using today's data for wallet {MAIN_ASSET_HOLDING_ADDRESS}")
        
    except Exception as e:
        logger.error(f"Error fetching balance data: {e}")
        return {}, {}
    
    # Process the data
    for row in balances:
        token = row.token_symbol
        
        # Unallocated balance in warm wallet
        if row.unallocated_balance and row.unallocated_balance > 0:
            warm_wallet[token] = warm_wallet.get(token, 0) + float(row.unallocated_balance)
        
        # Allocated balance to pools
        if row.allocated_balance and row.allocated_balance > 0 and row.pool_id:
            key = (row.pool_id, token)
            allocations[key] = allocations.get(key, 0) + float(row.allocated_balance)
    
    logger.info(f"Warm wallet: {len(warm_wallet)} tokens, Total allocated positions: {len(allocations)}")
    return warm_wallet, allocations


def fetch_default_parameters() -> Dict:
    """Fetches default parameters from default_allocation_parameters table."""
    repo = ParameterRepository()
    defaults = repo.get_all_default_parameters()
    logger.info(f"Loaded {len(defaults)} default parameters from default_allocation_parameters")
    return defaults


def fetch_allocation_parameters(custom_overrides: Dict = None) -> Dict:
    """
    Fetches the latest allocation parameters with support for custom overrides.
    """
    repo = ParameterRepository()
    
    # First, fetch default parameters
    default_params = fetch_default_parameters()
    
    # Then, fetch the latest allocation parameters
    latest_params_obj = repo.get_latest_parameters()
    
    if not latest_params_obj:
        logger.warning("No allocation parameters found, using defaults from default_allocation_parameters")
        params = {}
    else:
        # Convert SQLAlchemy object to dict
        params = {
             'run_id': latest_params_obj.run_id,
             'timestamp': latest_params_obj.timestamp,
             'max_alloc_percentage': latest_params_obj.max_alloc_percentage,
             'conversion_rate': latest_params_obj.conversion_rate,
             'tvl_limit_percentage': latest_params_obj.tvl_limit_percentage,
             'min_pools': latest_params_obj.min_pools,
             'token_marketcap_limit': latest_params_obj.token_marketcap_limit,
             'pool_tvl_limit': latest_params_obj.pool_tvl_limit,
             'pool_apy_limit': latest_params_obj.pool_apy_limit,
             'pool_pair_tvl_ratio_min': latest_params_obj.pool_pair_tvl_ratio_min,
             'pool_pair_tvl_ratio_max': latest_params_obj.pool_pair_tvl_ratio_max,
             'group1_max_pct': latest_params_obj.group1_max_pct,
             'group2_max_pct': latest_params_obj.group2_max_pct,
             'group3_max_pct': latest_params_obj.group3_max_pct,
             'position_max_pct_total_assets': latest_params_obj.position_max_pct_total_assets,
             'position_max_pct_pool_tvl': latest_params_obj.position_max_pct_pool_tvl,
             'group1_apy_delta_max': latest_params_obj.group1_apy_delta_max,
             'group1_7d_stddev_max': latest_params_obj.group1_7d_stddev_max,
             'group1_30d_stddev_max': latest_params_obj.group1_30d_stddev_max,
             'group2_apy_delta_max': latest_params_obj.group2_apy_delta_max,
             'group2_7d_stddev_max': latest_params_obj.group2_7d_stddev_max,
             'group2_30d_stddev_max': latest_params_obj.group2_30d_stddev_max,
             'group3_apy_delta_min': latest_params_obj.group3_apy_delta_min,
             'group3_7d_stddev_min': latest_params_obj.group3_7d_stddev_min,
             'group3_30d_stddev_min': latest_params_obj.group3_30d_stddev_min,
             'icebox_ohlc_l_threshold_pct': latest_params_obj.icebox_ohlc_l_threshold_pct,
             'icebox_ohlc_l_days_threshold': latest_params_obj.icebox_ohlc_l_days_threshold,
             'icebox_ohlc_c_threshold_pct': latest_params_obj.icebox_ohlc_c_threshold_pct,
             'icebox_ohlc_c_days_threshold': latest_params_obj.icebox_ohlc_c_days_threshold,
             'icebox_recovery_l_days_threshold': latest_params_obj.icebox_recovery_l_days_threshold,
             'icebox_recovery_c_days_threshold': latest_params_obj.icebox_recovery_c_days_threshold
        }
        logger.info(f"Parameter source: Latest allocation_parameters with run_id={params.get('run_id')}")

    # Fallbacks for critical parameters if missing from DB or object
    # Using defaults logic similar to original script
    defaults_map = {
        'max_alloc_percentage': 0.25,
        'conversion_rate': 0.0004,
        'tvl_limit_percentage': 0.05,
        'min_pools': 5,
        'min_transaction_value': 50.0,
        # ... others ...
    }
    
    # Merge logic: params from DB (if any) -> defaults from DB -> hardcoded defaults
    final_params = defaults_map.copy()
    
    # Update with DB defaults
    for k, v in default_params.items():
        if v is not None:
             final_params[k] = v
             
    # Update with latest params from table
    for k, v in params.items():
        if v is not None:
            final_params[k] = v
            
    # Apply custom overrides if provided
    if custom_overrides:
        logger.info(f"Applying {len(custom_overrides)} custom parameter overrides")
        for key, value in custom_overrides.items():
            if key in final_params:
                original_value = final_params[key]
                final_params[key] = value
                logger.info(f"Override applied: {key} changed from {original_value} to {value}")
            else:
                logger.warning(f"Override parameter '{key}' not found in parameters, skipping")
    
    logger.info(f"Final parameters: max_alloc={final_params.get('max_alloc_percentage')}, tvl_limit={final_params.get('tvl_limit_percentage')}")
    return final_params


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_aum(warm_wallet: Dict[str, float], 
                  current_allocations: Dict[Tuple[str, str], float],
                  token_prices: Dict[str, float]) -> float:
    """Calculates total Assets Under Management in USD."""
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
    """Builds the complete set of tokens needed for optimization."""
    tokens = set()
    
    # Tokens from pools using underlying_tokens
    for _, row in pools_df.iterrows():
        underlying_tokens = row.get('underlying_tokens')
        
        has_valid_tokens = False
        if isinstance(underlying_tokens, list):
            has_valid_tokens = len(underlying_tokens) > 0
        elif isinstance(underlying_tokens, str):
            has_valid_tokens = True
        elif underlying_tokens is not None:
            has_valid_tokens = pd.notna(underlying_tokens)
        
        if has_valid_tokens:
            try:
                if isinstance(underlying_tokens, str):
                    pool_tokens = json.loads(underlying_tokens)
                elif isinstance(underlying_tokens, list):
                    pool_tokens = underlying_tokens
                else:
                    pool_tokens = None
                
                if isinstance(pool_tokens, list) and pool_tokens:
                    tokens.update(pool_tokens)
                else:
                    logger.warning(f"No valid underlying_tokens found for pool {row.get('pool_id')}")
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse underlying_tokens for pool {row.get('pool_id')}: {underlying_tokens}")
        else:
            logger.warning(f"Pool {row.get('pool_id')} has no underlying_tokens")
    
    # Tokens in warm wallet
    tokens.update(warm_wallet.keys())
    
    # Tokens in current allocations
    for (pool_id, token) in current_allocations.keys():
        tokens.add(token)
    
    token_list = sorted(list(tokens))
    logger.info(f"Token universe: {len(token_list)} tokens - {token_list}")
    return token_list


# ============================================================================
# OPTIMIZATION MODEL (AllocationOptimizer Class)
# ============================================================================

class AllocationOptimizer:
    """
    Transaction-aware portfolio optimization for stablecoin pools.
    """
    
    def __init__(self, pools_df: pd.DataFrame, token_prices: Dict[str, float],
                 warm_wallet: Dict[str, float], current_allocations: Dict[Tuple[str, str], float],
                 gas_fees: Dict[str, float], alloc_params: Dict):
        """Initialize the optimizer."""
        # Cast numeric columns to float to avoid Decimal issues
        if not pools_df.empty:
            pools_df['forecasted_apy'] = pools_df['forecasted_apy'].astype(float)
            pools_df['forecasted_tvl'] = pools_df['forecasted_tvl'].astype(float)
            
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
        for _, row in pools_df.iterrows():
            pool_id = row['pool_id']
            underlying_tokens = row.get('underlying_tokens')
            
            # Parse underlying_tokens from JSON if needed
            if isinstance(underlying_tokens, str):
                try:
                    tokens = json.loads(underlying_tokens)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse underlying_tokens for pool {pool_id}: {underlying_tokens}")
                    tokens = []
            elif isinstance(underlying_tokens, list):
                tokens = underlying_tokens
            else:
                logger.warning(f"Pool {pool_id} has invalid underlying_tokens: {underlying_tokens}")
                tokens = []
            
            self.pool_tokens[pool_id] = tokens
            self.pool_tvl[pool_id] = row['forecasted_tvl']
        
        # Calculate AUM
        self.total_aum = calculate_aum(warm_wallet, current_allocations, token_prices)
        
        # Constants
        self.conversion_rate = float(alloc_params.get('conversion_rate', 0.0004))
        self.max_alloc_percentage = float(alloc_params.get('max_alloc_percentage', 0.25))
        self.tvl_limit_percentage = float(alloc_params.get('tvl_limit_percentage', 0.05) or 0.05)
        
        logger.info(f"Optimizer initialized: {self.n_pools} pools, {self.n_tokens} tokens, AUM=${self.total_aum:,.2f}")
        logger.info(f"Parameters: max_alloc={self.max_alloc_percentage:.1%} (${self.max_alloc_percentage * self.total_aum:,.2f} per pool), tvl_limit={self.tvl_limit_percentage:.1%}, conversion_rate={self.conversion_rate:.4%}")
    
    def build_model(self) -> "cp.Problem":
        """Constructs the CVXPY optimization model."""
        import cvxpy as cp
        self._cp = cp  # Store reference for use in solve() method
        
        logger.info("Building optimization model with incremental adjustment support...")
        
        # Variables definition
        # WE model the FINAL position and compute changes
        self.final_alloc = cp.Variable((self.n_pools, self.n_tokens), nonneg=True)
        self.convert = cp.Variable((self.n_tokens, self.n_tokens), nonneg=True)
        self.final_warm_wallet = cp.Variable(self.n_tokens, nonneg=True)
        
        # Deltas
        self.net_deposit = cp.Variable((self.n_pools, self.n_tokens), nonneg=True)
        self.net_withdraw = cp.Variable((self.n_pools, self.n_tokens), nonneg=True)
        
        # Binary indicators
        self.has_deposit = cp.Variable((self.n_pools, self.n_tokens), boolean=True)
        self.has_withdrawal = cp.Variable((self.n_pools, self.n_tokens), boolean=True)
        self.has_allocation = cp.Variable(self.n_pools, boolean=True)
        self.is_conversion = cp.Variable((self.n_tokens, self.n_tokens), boolean=True)
        
        # Vectors
        current_alloc_matrix = np.zeros((self.n_pools, self.n_tokens))
        for (pool_id, token), amount in self.current_allocations.items():
            if pool_id in self.pool_idx and token in self.token_idx:
                i = self.pool_idx[pool_id]
                j = self.token_idx[token]
                current_alloc_matrix[i, j] = amount
        
        warm_wallet_vector = np.zeros(self.n_tokens)
        for token, amount in self.warm_wallet.items():
            if token in self.token_idx:
                j = self.token_idx[token]
                warm_wallet_vector[j] = amount
        
        price_vector = np.array([self.token_prices.get(t, 1.0) for t in self.tokens])
        
        # Objective Function
        daily_apy_matrix = np.zeros((self.n_pools, self.n_tokens))
        for _, row in self.pools_df.iterrows():
            pool_id = row['pool_id']
            i = self.pool_idx[pool_id]
            daily_apy = row['forecasted_apy'] / 100.0 / 365.0
            
            for token in self.pool_tokens[pool_id]:
                if token in self.token_idx:
                    j = self.token_idx[token]
                    daily_apy_matrix[i, j] = daily_apy
        
        # Yield based on FINAL allocation
        yield_usd = cp.sum(cp.multiply(
            cp.multiply(self.final_alloc, daily_apy_matrix),
            price_vector
        ))
        
        # Transaction Costs - based on DELTAS
        withdrawal_gas_costs = cp.sum(cp.multiply(self.has_withdrawal, self.withdrawal_gas_fee))
        deposit_gas_costs = cp.sum(cp.multiply(self.has_deposit, self.allocation_gas_fee))
        conversion_conversion_costs = cp.sum(cp.multiply(cp.multiply(self.convert, price_vector), self.conversion_rate))
        conversion_gas_costs = cp.sum(cp.multiply(self.is_conversion, self.conversion_gas_fee))
        
        self.total_transaction_costs = (
            withdrawal_gas_costs + deposit_gas_costs + conversion_conversion_costs + conversion_gas_costs
        )
        
        # Objective: Maximize (Annualized Yield - Transaction Costs)
        annual_yield_usd = yield_usd * 365.0
        objective = cp.Maximize(annual_yield_usd - self.total_transaction_costs)
        
        # Constraints
        constraints = []
        
        # 1. Link final allocation to changes
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                constraints.append(
                    self.final_alloc[i, j] == current_alloc_matrix[i, j] + self.net_deposit[i, j] - self.net_withdraw[i, j]
                )
        
        # 2. Balance Conservation
        for j in range(self.n_tokens):
            in_flow = warm_wallet_vector[j] + cp.sum(self.net_withdraw[:, j]) + cp.sum(self.convert[:, j])
            out_flow = self.final_warm_wallet[j] + cp.sum(self.net_deposit[:, j]) + cp.sum(self.convert[j, :])
            constraints.append(in_flow == out_flow)
        
        # 3. Withdrawal limits (Cannot withdraw more than current alloc)
        constraints.append(self.net_withdraw <= current_alloc_matrix)
        
        # 4. No self-conversion
        for j in range(self.n_tokens):
            constraints.append(self.convert[j, j] == 0)
        
        # 5. Multi-token pool equal distribution (on FINAL allocation)
        for pool_id, tokens in self.pool_tokens.items():
            if len(tokens) > 1:
                i = self.pool_idx[pool_id]
                token_indices = [self.token_idx[t] for t in tokens if t in self.token_idx]
                if len(token_indices) > 1:
                    for k in range(len(token_indices) - 1):
                        j1 = token_indices[k]
                        j2 = token_indices[k + 1]
                        constraints.append(self.final_alloc[i, j1] * price_vector[j1] == self.final_alloc[i, j2] * price_vector[j2])
        
        # 6. Pool allocation limits
        max_pool_allocation_usd = self.max_alloc_percentage * self.total_aum * 0.999
        for i in range(self.n_pools):
            pool_total_usd = cp.sum(cp.multiply(self.final_alloc[i, :], price_vector))
            constraints.append(pool_total_usd <= max_pool_allocation_usd)
            
            pool_id = self.pools[i]
            pool_forecasted_tvl = self.pool_tvl.get(pool_id, 0)
            pool_tvl_min_limit = float(self.alloc_params.get('pool_tvl_limit', 0) or 0)
            
            if pool_forecasted_tvl < pool_tvl_min_limit:
                 # Force exit if TVL is below absolute minimum limit
                 constraints.append(pool_total_usd == 0)
            else:
                 # TVL Limit applies to FINAL allocation size
                 constraints.append(pool_total_usd <= self.tvl_limit_percentage * pool_forecasted_tvl)
        
        # 7. AUM Conservation
        total_allocated_usd = cp.sum(cp.multiply(self.final_alloc, price_vector))
        total_final_warm_wallet_usd = cp.sum(cp.multiply(self.final_warm_wallet, price_vector))
        # Allow slight slack for numerical stability
        constraints.append(total_allocated_usd + self.total_transaction_costs + total_final_warm_wallet_usd <= self.total_aum * 1.0001)
        
        # 8. Binary variables linkage
        big_M = self.total_aum
        
        # Link has_allocation
        for i in range(self.n_pools):
             pool_val = cp.sum(self.final_alloc[i, :])
             constraints.append(pool_val <= big_M * self.has_allocation[i])
             constraints.append(pool_val >= 0.001 * self.has_allocation[i])

        # Min Pools Constraint
        min_pools = self.alloc_params.get('min_pools', 0)
        if min_pools > 0:
            constraints.append(cp.sum(self.has_allocation) >= min_pools)
             
        # Link transaction indicators
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                constraints.append(self.net_withdraw[i, j] <= big_M * self.has_withdrawal[i, j])
                constraints.append(self.net_deposit[i, j] <= big_M * self.has_deposit[i, j])
                
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                if i != j:
                    big_M_convert = self.total_aum
                    constraints.append(self.convert[i, j] <= big_M_convert * self.is_conversion[i, j])
                    constraints.append(self.convert[i, j] >= 0.01 * self.is_conversion[i, j])
        
        # Link has_allocation min threshold (avoid dust)
        for i in range(self.n_pools):
            pool_val = cp.sum(self.final_alloc[i, :])
            # Logic: If has_allocation is 1, pool_val must be >= min_threshold
            # We skip strict enforcement here to avoid infeasibility, relied on post-processing cleanup if needed.
        
        # 10. Non-negativity
        constraints.append(self.final_alloc >= 0)
        constraints.append(self.net_deposit >= 0)
        constraints.append(self.net_withdraw >= 0)
        constraints.append(self.convert >= 0)
        constraints.append(self.final_warm_wallet >= 0)
        
        logger.info(f"Model built with {len(constraints)} constraints")
        return cp.Problem(objective, constraints)
    
    def solve(self, solver=None, verbose=True) -> bool:
        """Solves the optimization problem."""
        problem = self.build_model()
        cp = self._cp
        if solver is None:
            solver = cp.HIGHS
        
        logger.info(f"Solving with {solver}...")
        try:
            # Use tighter MIP gap tolerance for better solution quality
            # Default gap (~1%) can lead to suboptimal allocations worth $40-50/year
            solver_options = {}
            if solver == cp.HIGHS:
                solver_options = {
                    'mip_rel_gap': 0.0001,  # 0.01% relative gap (much tighter than default)
                    'time_limit': 300.0,     # 5 minute time limit
                }
                logger.info(f"HIGHS solver options: mip_rel_gap=0.01%, time_limit=300s")
            
            problem.solve(solver=solver, verbose=verbose, **solver_options)
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
        """Extracts results from solved optimization model."""
        if self.final_alloc.value is None:
            logger.error("No solution available to extract")
            return pd.DataFrame(), []
            
        allocations = []
        transactions = []
        transaction_seq = 1
        price_vector = np.array([self.token_prices.get(t, 1.0) for t in self.tokens])
        
        # Helper to get value safe from None
        def get_val(var, i, j):
             return var.value[i, j] if var.value is not None else 0
        
        # Step 0: Holds (Existing positions maintained)
        # We process these first so they appear at the top of the transaction log
        for (pool_id, token), current_amt in self.current_allocations.items():
            if pool_id in self.pool_idx and token in self.token_idx:
                i = self.pool_idx[pool_id]
                j = self.token_idx[token]
                
                # Check how much was withdrawn
                withdraw_amt = get_val(self.net_withdraw, i, j)
                
                # Kept amount is Current - Withdrawn
                kept_amt = current_amt - withdraw_amt
                
                if kept_amt > 0.01:
                    price = price_vector[j]
                    cost = 0.0 # Holding costs nothing
                    
                    transactions.append({
                        'seq': transaction_seq,
                        'type': 'HOLD',
                        'from_location': pool_id,
                        'to_location': pool_id,
                        'token': token,
                        'amount': kept_amt,
                        'amount_usd': kept_amt * price,
                        'gas_cost_usd': 0.0,
                        'conversion_cost_usd': 0.0,
                        'total_cost_usd': 0.0
                    })
                    transaction_seq += 1

        # Step 1: Withdrawals (from net_withdraw variable)
        for i in range(self.n_pools):
            pool_id = self.pools[i]
            for j in range(self.n_tokens):
                token = self.tokens[j]
                amount = get_val(self.net_withdraw, i, j)
                if amount > 0.01:
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
                        'conversion_cost_usd': 0.0,
                        'total_cost_usd': withdrawal_cost
                    })
                    transaction_seq += 1
        
        # Step 2: Conversions
        for i in range(self.n_tokens):
            from_token = self.tokens[i]
            for j in range(self.n_tokens):
                to_token = self.tokens[j]
                if i != j:
                    amount = get_val(self.convert, i, j)
                    if amount > 0.01:
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
        
        # Step 3: Allocations (from final_alloc variable, reporting deposits)
        # Note: We report the FINAL position in 'allocations' list, 
        # but the 'transactions' list records the DEPOSIT flow.
        
        for i in range(self.n_pools):
            pool_id = self.pools[i]
            symbol = self.pools_df[self.pools_df['pool_id'] == pool_id]['symbol'].iloc[0]
            
            for j in range(self.n_tokens):
                token = self.tokens[j]
                
                # Report Final Allocation
                final_amt = get_val(self.final_alloc, i, j)
                if final_amt > 0.01:
                    amount_usd = final_amt * price_vector[j]
                    
                    # Check if this involved a new deposit
                    deposit_amt = get_val(self.net_deposit, i, j)
                    
                    allocations.append({
                        'pool_id': pool_id,
                        'pool_symbol': symbol,
                        'token': token,
                        'amount': final_amt,
                        'amount_usd': amount_usd,
                        'needs_conversion': bool(deposit_amt > 0.01)
                    })
                    
                    # Record Transaction if there was a deposit
                    if deposit_amt > 0.01:
                        gas_cost = self.allocation_gas_fee
                        total_cost = gas_cost
                        
                        transactions.append({
                            'seq': transaction_seq,
                            'type': 'ALLOCATION',
                            'from_location': 'warm_wallet',
                            'to_location': pool_id,
                            'token': token,
                            'amount': deposit_amt,
                            'amount_usd': deposit_amt * price_vector[j],
                            'conversion_cost_usd': 0.0,
                            'gas_cost_usd': gas_cost,
                            'total_cost_usd': total_cost,
                            'needs_conversion': False 
                        })
                        transaction_seq += 1
                    
        allocations_df = pd.DataFrame(allocations)
        logger.info(f"Extracted {len(allocations)} allocations and {len(transactions)} transactions")
        return allocations_df, transactions

    def format_results(self) -> Dict:
        """Formats optimization results."""
        if self.final_alloc.value is None:
             return {"final_allocations": {}, "unallocated_tokens": {}, "transactions": []}
        
        allocations_df, transactions = self.extract_results()
        final_allocations = {}
        unallocated_tokens = {}
        price_vector = np.array([self.token_prices.get(t, 1.0) for t in self.tokens])
        
        for _, row in allocations_df.iterrows():
            pool_id = row['pool_id']
            if pool_id not in final_allocations:
                final_allocations[pool_id] = {"pool_symbol": row['pool_symbol'], "tokens": {}}
            final_allocations[pool_id]["tokens"][row['token']] = {
                "amount": row['amount'], "amount_usd": row['amount_usd']
            }
            
        for j in range(self.n_tokens):
            amount = self.final_warm_wallet.value[j] if self.final_warm_wallet.value is not None else 0
            if amount > 0.01:
                 unallocated_tokens[self.tokens[j]] = {
                     "amount": amount, "amount_usd": amount * price_vector[j]
                 }
                 
        formatted_transactions = []
        for txn in transactions:
            ftxn = {
                "seq": txn["seq"], "type": txn["type"], 
                "from_location": txn["from_location"], "to_location": txn["to_location"],
                "amount": txn["amount"], "amount_usd": txn["amount_usd"], 
                "gas_cost_usd": txn["gas_cost_usd"]
            }
            if "token" in txn: ftxn["token"] = txn["token"]
            if txn["type"] == "CONVERSION":
                ftxn["from_token"] = txn["from_token"]
                ftxn["to_token"] = txn["to_token"]
            if "conversion_cost_usd" in txn: ftxn["conversion_cost_usd"] = txn["conversion_cost_usd"]
            if "total_cost_usd" in txn: ftxn["total_cost_usd"] = txn["total_cost_usd"]
            if "needs_conversion" in txn: ftxn["needs_conversion"] = txn["needs_conversion"]
            formatted_transactions.append(ftxn)
            
        return {
            "final_allocations": final_allocations,
            "unallocated_tokens": unallocated_tokens,
            "transactions": formatted_transactions
        }

# ============================================================================
# RESULTS PERSISTENCE
# ============================================================================

def delete_todays_allocations():
    """Deletes all asset allocations for the current date."""
    repo = AllocationRepository()
    try:
        deleted = repo.delete_allocations_for_date(date.today())
        logger.info(f"Deleted {deleted} existing allocation records for today")
    except Exception as e:
        logger.error(f"Error deleting today's allocations: {e}")
        raise


def store_results(run_id: str, allocations_df: pd.DataFrame, 
                  transactions: List[Dict], alloc_params: Dict):
    """Stores optimization results to database."""
    repo = AllocationRepository()
    
    try:
        allocations_data = []
        for txn in transactions:
            amount = float(txn['amount']) if hasattr(txn['amount'], 'dtype') else txn['amount']
            step_number = int(txn['seq']) if hasattr(txn['seq'], 'dtype') else txn['seq']
            
            allocations_data.append({
                'run_id': run_id,
                'timestamp': datetime.now(),
                'step_number': step_number,
                'operation': txn['type'],
                'from_asset': txn.get('from_token', txn.get('token')),
                'to_asset': txn.get('to_token', txn.get('token')),
                'amount': amount,
                'pool_id': txn.get('to_location') if txn['type'] == 'ALLOCATION' else (txn.get('from_location') if txn['type'] in ['WITHDRAWAL', 'HOLD'] else None)
            })
            
        repo.bulk_insert_allocations(allocations_data)
        logger.info(f"✓ Results stored with run_id: {run_id}")
        
    except Exception as e:
        logger.error(f"Error storing results: {e}")
        raise


def update_allocation_parameters_with_results(run_id: str, transactions: List[Dict], 
                                              pools_df: pd.DataFrame, allocations_df: pd.DataFrame):
    """Updates allocation_parameters table with optimization results."""
    repo = ParameterRepository()
    
    try:
        total_costs = sum(float(txn.get('total_cost_usd', 0)) for txn in transactions)
        
        projected_apy = 0.0
        if not allocations_df.empty:
             merged = allocations_df.merge(pools_df[['pool_id', 'forecasted_apy']], on='pool_id', how='left')
             total_amt = merged['amount_usd'].sum()
             if total_amt > 0:
                 projected_apy = (merged['amount_usd'] * merged['forecasted_apy']).sum() / total_amt
                 
        repo.update_run_results(
            run_id=run_id,
            projected_apy=float(projected_apy),
            transaction_costs=float(total_costs),
            transaction_sequence=json.dumps(transactions, default=str)
        )
        logger.info(f"✓ Updated allocation_parameters with results")
        
    except Exception as e:
        logger.error(f"Error updating parameters: {e}")
        raise


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def optimize_allocations(custom_overrides: Dict = None):
    """Main orchestration function for asset allocation optimization."""
    logger.info("=" * 80)
    logger.info("STABLECOIN POOL ALLOCATION OPTIMIZATION")
    logger.info("=" * 80)
    
    # 0. Data Quality Report (optional, kept light)
    try:
        quality_report = generate_data_quality_report()
        score = quality_report.get('summary', {}).get('overall_quality_score', 0)
        logger.info(f"Data quality score: {score:.1f}/100")
    except Exception as e:
        logger.warning(f"Data quality report skipped: {e}")

    # 1. Load Data
    logger.info("\n[1/5] Loading data...")
    pools_df = fetch_pool_data()
    if pools_df.empty:
        logger.warning("No approved pools available. Exiting.")
        return
        
    warm_wallet, current_allocations = fetch_current_balances()
    tokens = build_token_universe(pools_df, warm_wallet, current_allocations)
    token_prices = fetch_token_prices(tokens + ['ETH'])
    eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units = fetch_gas_fee_data()
    gas_fees = calculate_transaction_gas_fees(eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units)
    
    # 2. Parameters
    alloc_params = fetch_allocation_parameters(custom_overrides)
    run_id = alloc_params.get('run_id') # Usually None unless existing run is updated?? Needs handling.
    # Note: If fetching latest parameters, run_id might be from previous run or a new draft. 
    # Usually we generate a new run_id or use one passed in. 
    # For now, if no run_id, generate one.
    if not run_id:
        run_id = str(uuid4())
        logger.info(f"Generated new run_id: {run_id}")
    
    # 3. Initialize Optimizer
    logger.info("\n[3/6] Initializing optimizer...")
    optimizer = AllocationOptimizer(pools_df, token_prices, warm_wallet, current_allocations, gas_fees, alloc_params)
    
    # 4. Solve
    logger.info("\n[4/6] Solving optimization problem...")
    import cvxpy as cp
    success = False
    
    for solver_name in ['HIGHS', 'CBC', 'SCIPY', 'OSQP', 'ECOS', 'CLARABEL']:
        try:
             if not hasattr(cp, solver_name):
                 continue
                 
             solver = getattr(cp, solver_name)
             logger.info(f"Attempting solver: {solver_name}")
             if optimizer.solve(solver=solver):
                 success = True
                 logger.info(f"Solver {solver_name} succeeded")
                 break
             else:
                 logger.warning(f"Solver {solver_name} failed to find optimal solution")
        except Exception as e:
             logger.warning(f"Solver {solver_name} raised exception: {e}")
             continue
             
    if not success:
        logger.error("All solvers failed to find an optimal solution")
        return
        
    # 5. Extract Results
    logger.info("\n[5/6] Extracting results...")
    optimizer.format_results()
    
    # 6. Store Results
    logger.info("\n[6/6] Storing results...")
    delete_todays_allocations()
    
    allocations_df, transactions = optimizer.extract_results()
    store_results(run_id, allocations_df, transactions, alloc_params)
    update_allocation_parameters_with_results(run_id, transactions, pools_df, allocations_df)
    
    logger.info("Optimization completed successfully.")

if __name__ == "__main__":
    optimize_allocations()