"""
Asset Allocation Optimization Module (OR-Tools Implementation)

This module implements a transaction-aware optimization algorithm for allocating assets
to stablecoin pools. It uses Google OR-Tools (SCIP solver) to maximize daily yield while
accounting for gas fees, conversion costs, and various constraints.
"""

import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
from uuid import uuid4
from typing import Dict, List, Tuple, TYPE_CHECKING, Optional

from ortools.linear_solver import pywraplp

from database.repositories.pool_metrics_repository import PoolMetricsRepository
from database.repositories.daily_balance_repository import DailyBalanceRepository
from database.repositories.raw_data_repository import RawDataRepository
from database.repositories.parameter_repository import ParameterRepository
from database.repositories.allocation_repository import AllocationRepository
from asset_allocation.data_quality_report import generate_data_quality_report

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# ============================================================================
# DATA LOADING FUNCTIONS (Reused from optimize_allocations.py)
# ============================================================================

def fetch_pool_data() -> pd.DataFrame:
    """
    Fetches approved pools with forecasted APY and metadata.
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
                if t_lower in addr_to_symbol:
                    normalized_pool_tokens.append(addr_to_symbol[t_lower])
                else:
                    normalized_pool_tokens.append(t)
        
        normalized_tokens_list.append(normalized_pool_tokens)
        
    df['underlying_tokens'] = normalized_tokens_list
    return df

def fetch_token_prices(tokens: List[str]) -> Dict[str, float]:
    repo = RawDataRepository()
    prices = repo.get_latest_prices(tokens)
    logger.info(f"Loaded prices for {len(prices)} tokens")
    return prices

def fetch_gas_fee_data() -> Tuple[float, float, float, float, float]:
    repo = RawDataRepository()
    eth_prices = repo.get_latest_prices(['ETH'])
    eth_price = eth_prices.get('ETH', 3000.0)
    
    base_fee_transfer_gwei = 10.0
    base_fee_swap_gwei = 30.0
    priority_fee_gwei = 10.0
    min_gas_units = 21000
    
    return eth_price, base_fee_transfer_gwei, base_fee_swap_gwei, priority_fee_gwei, min_gas_units

def calculate_gas_fee_usd(gas_units: float, base_fee_gwei: float, priority_fee_gwei: float, eth_price_usd: float) -> float:
    total_fee_gwei = gas_units * (base_fee_gwei + priority_fee_gwei)
    gas_fee_usd = total_fee_gwei * 1e-9 * eth_price_usd
    return gas_fee_usd

def calculate_transaction_gas_fees(eth_price_usd: float, base_fee_transfer_gwei: float, 
                                   base_fee_swap_gwei: float, priority_fee_gwei: float, 
                                   min_gas_units: float) -> Dict[str, float]:
    pool_transaction_gas_fee_usd = calculate_gas_fee_usd(
        min_gas_units, base_fee_transfer_gwei, priority_fee_gwei, eth_price_usd
    )
    token_swap_gas_fee_usd = calculate_gas_fee_usd(
        min_gas_units, base_fee_swap_gwei, priority_fee_gwei, eth_price_usd
    )
    
    gas_fees = {
        'allocation': pool_transaction_gas_fee_usd,
        'withdrawal': pool_transaction_gas_fee_usd,
        'conversion': token_swap_gas_fee_usd,
        'transfer': pool_transaction_gas_fee_usd,
        'deposit': pool_transaction_gas_fee_usd
    }
    logger.info(f"Transaction gas fees - Pool: ${pool_transaction_gas_fee_usd:.6f}, Swap: ${token_swap_gas_fee_usd:.6f}")
    return gas_fees

def fetch_current_balances() -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    from config import MAIN_ASSET_HOLDING_ADDRESS
    repo = DailyBalanceRepository()
    
    if not MAIN_ASSET_HOLDING_ADDRESS:
        return {}, {}
    
    warm_wallet = {}
    allocations = {}
    
    try:
        balances = repo.get_current_balances(MAIN_ASSET_HOLDING_ADDRESS, date.today())
        if not balances:
            return {}, {}
    except Exception:
        return {}, {}
    
    for row in balances:
        token = row.token_symbol
        if row.unallocated_balance and row.unallocated_balance > 0:
            warm_wallet[token] = warm_wallet.get(token, 0) + float(row.unallocated_balance)
        if row.allocated_balance and row.allocated_balance > 0 and row.pool_id:
            key = (row.pool_id, token)
            allocations[key] = allocations.get(key, 0) + float(row.allocated_balance)
    
    return warm_wallet, allocations

def fetch_allocation_parameters(custom_overrides: Dict = None) -> Dict:
    repo = ParameterRepository()
    default_params = repo.get_all_default_parameters()
    latest_params_obj = repo.get_latest_parameters()
    
    if not latest_params_obj:
        params = {}
    else:
        # Simplified for brevity - in real implementation ensuring all keys are mapped as in original
        params = {
             'run_id': latest_params_obj.run_id,
             'max_alloc_percentage': latest_params_obj.max_alloc_percentage,
             'conversion_rate': latest_params_obj.conversion_rate,
             'tvl_limit_percentage': latest_params_obj.tvl_limit_percentage,
             'min_pools': latest_params_obj.min_pools,
             'pool_tvl_limit': latest_params_obj.pool_tvl_limit,
             'pool_apy_limit': latest_params_obj.pool_apy_limit,
        }

    defaults_map = {
        'max_alloc_percentage': 0.25,
        'conversion_rate': 0.0004,
        'tvl_limit_percentage': 0.05,
        'min_pools': 4,
        'optimization_horizon_days': 30, # Default to 30 days if not specified
    }
    
    final_params = defaults_map.copy()
    for k, v in default_params.items():
        if v is not None: final_params[k] = v
    for k, v in params.items():
        if v is not None: final_params[k] = v
            
    if custom_overrides:
        for key, value in custom_overrides.items():
            if key in final_params:
                final_params[key] = value
                
    return final_params

def calculate_aum(warm_wallet: Dict[str, float], 
                  current_allocations: Dict[Tuple[str, str], float],
                  token_prices: Dict[str, float]) -> float:
    total_usd = 0.0
    for token, amount in warm_wallet.items():
        price = token_prices.get(token, 1.0)
        total_usd += amount * price
    for (pool_id, token), amount in current_allocations.items():
        price = token_prices.get(token, 1.0)
        total_usd += amount * price
    return total_usd

def build_token_universe(pools_df: pd.DataFrame, 
                         warm_wallet: Dict[str, float],
                         current_allocations: Dict[Tuple[str, str], float]) -> List[str]:
    tokens = set()
    for _, row in pools_df.iterrows():
        underlying = row.get('underlying_tokens')
        if isinstance(underlying, list):
            tokens.update(underlying)
    tokens.update(warm_wallet.keys())
    for (pool_id, token) in current_allocations.keys():
        tokens.add(token)
    return sorted(list(tokens))

# ============================================================================
# OPTIMIZATION MODEL (AllocationOptimizer Class - OR-Tools)
# ============================================================================

class AllocationOptimizer:
    """
    Transaction-aware portfolio optimization using Google OR-Tools.
    """
    
    def __init__(self, pools_df: pd.DataFrame, token_prices: Dict[str, float],
                 warm_wallet: Dict[str, float], current_allocations: Dict[Tuple[str, str], float],
                 gas_fees: Dict[str, float], alloc_params: Dict):
        
        # Cast numeric columns
        if not pools_df.empty:
            pools_df['forecasted_apy'] = pools_df['forecasted_apy'].astype(float)
            pools_df['forecasted_tvl'] = pools_df['forecasted_tvl'].astype(float)
            
        self.pools_df = pools_df
        self.token_prices = token_prices
        self.warm_wallet = warm_wallet
        self.current_allocations = current_allocations
        self.gas_fees = gas_fees
        self.alloc_params = alloc_params
        
        # Optimization Horizon (default 30 days)
        self.optimization_horizon_days = float(alloc_params.get('optimization_horizon_days', 30))
        logger.info(f"Optimization Horizon: {self.optimization_horizon_days} days")

        self.allocation_gas_fee = gas_fees['allocation']
        self.withdrawal_gas_fee = gas_fees['withdrawal']
        self.conversion_gas_fee = gas_fees['conversion']
        
        # Indices
        self.tokens = build_token_universe(pools_df, warm_wallet, current_allocations)
        self.pools = pools_df['pool_id'].tolist()
        
        self.n_tokens = len(self.tokens)
        self.n_pools = len(self.pools)
        
        self.token_idx = {t: i for i, t in enumerate(self.tokens)}
        self.pool_idx = {p: i for i, p in enumerate(self.pools)}
        
        # Pool helpers
        self.pool_tokens = {}
        self.pool_tvl = {}
        for _, row in pools_df.iterrows():
            pool_id = row['pool_id']
            # underlying_tokens here is already normalized list from fetch_pool_data
            self.pool_tokens[pool_id] = row.get('underlying_tokens', [])
            self.pool_tvl[pool_id] = row['forecasted_tvl']
            
        self.total_aum = calculate_aum(warm_wallet, current_allocations, token_prices)
        logger.info(f"DEBUG: Total AUM calculated: ${self.total_aum:,.2f}")
        
        # Constants
        self.conversion_rate = float(alloc_params.get('conversion_rate', 0.0004))
        self.max_alloc_percentage = float(alloc_params.get('max_alloc_percentage', 0.25))
        self.tvl_limit_percentage = float(alloc_params.get('tvl_limit_percentage', 0.05) or 0.05)
        
        # Solver variables holder
        self.solver = None
        self.vars = {} # Dictionary to hold created variables
        
    def build_model(self):
        """Constructs the OR-Tools optimization model."""
        # Create the linear solver with the SCIP backend.
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        if not self.solver:
            logger.error("SCIP solver unavailable.")
            return

        infinity = self.solver.infinity()
        
        # --- Variables ---
        
        # Continuous variables (Non-negative)
        # x[i,j]: Final allocation
        self.x = {}
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                self.x[i, j] = self.solver.NumVar(0.0, infinity, f'x_{i}_{j}')

        # w[i,j]: Withdrawal amount
        self.w = {}
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                self.w[i, j] = self.solver.NumVar(0.0, infinity, f'w_{i}_{j}')

        # d[i,j]: Deposit amount
        self.d = {}
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                self.d[i, j] = self.solver.NumVar(0.0, infinity, f'd_{i}_{j}')
        
        # c[token_from, token_to]: Conversion amount
        self.c = {}
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                if i != j:
                    self.c[i, j] = self.solver.NumVar(0.0, infinity, f'c_{i}_{j}')
        
        # y[j]: Final warm wallet
        self.y = {}
        for j in range(self.n_tokens):
            self.y[j] = self.solver.NumVar(0.0, infinity, f'y_{j}')

        # Binary Variables
        # h[i]: Has allocation in pool i
        self.h = {}
        for i in range(self.n_pools):
            self.h[i] = self.solver.BoolVar(f'h_{i}')
            
        # has_w[i, j]: Has withdrawal
        self.has_w = {}
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                self.has_w[i, j] = self.solver.BoolVar(f'has_w_{i}_{j}')

        # has_d[i, j]: Has deposit
        self.has_d = {}
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                self.has_d[i, j] = self.solver.BoolVar(f'has_d_{i}_{j}')

        # is_conv[i, j]: Is converting
        self.is_conv = {}
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                if i != j:
                    self.is_conv[i, j] = self.solver.BoolVar(f'is_conv_{i}_{j}')

        # --- Constants & Helpers ---
        
        price_vector = np.array([self.token_prices.get(t, 1.0) for t in self.tokens])
        
        # Precompute current allocation matrix
        current_alloc_matrix = np.zeros((self.n_pools, self.n_tokens))
        for (pool_id, token), amount in self.current_allocations.items():
            if pool_id in self.pool_idx and token in self.token_idx:
                current_alloc_matrix[self.pool_idx[pool_id], self.token_idx[token]] = amount

        warm_wallet_vector = np.zeros(self.n_tokens)
        for token, amount in self.warm_wallet.items():
            if token in self.token_idx:
                warm_wallet_vector[self.token_idx[token]] = amount
                
        big_M = self.total_aum * 1.1 # Slightly larger than AUM

        # --- Constraints ---

        # 1. Flow Conservation (Pools)
        # x[i,j] = CurrentAlloc[i,j] + d[i,j] - w[i,j]
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                self.solver.Add(
                    self.x[i, j] == current_alloc_matrix[i, j] + self.d[i, j] - self.w[i, j]
                )

        # 2. Flow Conservation (Warm Wallet)
        # Inflow = Outflow
        for j in range(self.n_tokens):
            # Inflow: Initial Wallet + Withdrawals (of token j from all pools) + Received Conversions (k -> j)
            inflow = warm_wallet_vector[j]
            for i in range(self.n_pools):
                inflow += self.w[i, j]
            for k in range(self.n_tokens):
                if k != j:
                    inflow += self.c[k, j]
            
            # Outflow: Final Wallet + Deposits (of token j to all pools) + Sent Conversions (j -> k)
            outflow = self.y[j]
            for i in range(self.n_pools):
                outflow += self.d[i, j]
            for k in range(self.n_tokens):
                if j != k:
                    outflow += self.c[j, k]
            
            self.solver.Add(inflow == outflow)

        # 3. Withdrawal Limits
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                self.solver.Add(self.w[i, j] <= current_alloc_matrix[i, j])

        # 4. No Self-Conversion (Implicit by loop ranges)

        # 5. Equal Token Distribution (Multi-token Pools)
        for pool_id, tokens in self.pool_tokens.items():
            if len(tokens) > 1:
                i = self.pool_idx[pool_id]
                token_indices = [self.token_idx[t] for t in tokens if t in self.token_idx]
                if len(token_indices) > 1:
                    for k in range(len(token_indices) - 1):
                        j1 = token_indices[k]
                        j2 = token_indices[k + 1]
                        # x[i, j1] * Price[j1] == x[i, j2] * Price[j2]
                        self.solver.Add(
                            self.x[i, j1] * price_vector[j1] == self.x[i, j2] * price_vector[j2]
                        )

        # 6. Pool Allocation Limits
        max_pool_allocation_usd = self.max_alloc_percentage * self.total_aum
        
        for i in range(self.n_pools):
            # Pool Value V_i
            pool_val_expr = self.solver.Sum([self.x[i, j] * price_vector[j] for j in range(self.n_tokens)])
            
            # Max Allocation
            self.solver.Add(pool_val_expr <= max_pool_allocation_usd)
            
            pool_id = self.pools[i]
            pool_forecasted_tvl = self.pool_tvl.get(pool_id, 0)
            pool_tvl_min_limit = float(self.alloc_params.get('pool_tvl_limit', 0) or 0)
            
            # TVL Limit
            if pool_forecasted_tvl < pool_tvl_min_limit:
                self.solver.Add(pool_val_expr == 0)
            else:
                self.solver.Add(pool_val_expr <= self.tvl_limit_percentage * pool_forecasted_tvl)
                
        # 7. AUM Conservation (Budget)
        # Sum(Allocated Value) + Sum(Wallet Value) + Costs <= Initial AUM
        # Note: Costs is an expression we define below, need to integrate it.
        # OR-Tools doesn't easily allow defining 'Objective' as a variable to use in constraints 
        # unless we explicitly make a variable for cost.
        # Let's define constraint using the expression directly.
        
        total_allocated_usd = self.solver.Sum([
            self.x[i, j] * price_vector[j] 
            for i in range(self.n_pools) for j in range(self.n_tokens)
        ])
        
        final_wallet_usd = self.solver.Sum([
            self.y[j] * price_vector[j]
            for j in range(self.n_tokens)
        ])
        
        # Transaction Costs Expression
        withdrawal_perm_costs = self.solver.Sum([
            self.has_w[i, j] * self.withdrawal_gas_fee
            for i in range(self.n_pools) for j in range(self.n_tokens)
        ])
        
        deposit_perm_costs = self.solver.Sum([
            self.has_d[i, j] * self.allocation_gas_fee
            for i in range(self.n_pools) for j in range(self.n_tokens)
        ])
        
        conversion_fixed_costs = self.solver.Sum([
            self.is_conv[i, j] * self.conversion_gas_fee
            for i in range(self.n_tokens) for j in range(self.n_tokens) if i != j
        ])
        
        conversion_var_costs = self.solver.Sum([
            self.c[i, j] * price_vector[i] * self.conversion_rate
            for i in range(self.n_tokens) for j in range(self.n_tokens) if i != j
        ])
        
        total_costs = withdrawal_perm_costs + deposit_perm_costs + conversion_fixed_costs + conversion_var_costs
        
        # 7. AUM Conservation (Budget)
        # We allow 1% slack to account for transaction costs that are not "burned" from token balances
        # but added to the constraint.
        self.solver.Add(total_allocated_usd + final_wallet_usd + total_costs <= self.total_aum * 1.01)

        # 8. Binary Linkage (Big-M)
        
        # Link h[i] (Pool has allocation) to x[i, j]
        for i in range(self.n_pools):
            pool_val_expr = self.solver.Sum([self.x[i, j] for j in range(self.n_tokens)])
            self.solver.Add(pool_val_expr <= big_M * self.h[i])
            self.solver.Add(pool_val_expr >= 0.001 * self.h[i]) # Minimum dust threshold if active

        # Link has_w to w
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                self.solver.Add(self.w[i, j] <= big_M * self.has_w[i, j])
                self.solver.Add(self.w[i, j] >= 0.01 * self.has_w[i, j]) # Min withdrawal size
                
        # Link has_d to d
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                self.solver.Add(self.d[i, j] <= big_M * self.has_d[i, j])
                self.solver.Add(self.d[i, j] >= 0.01 * self.has_d[i, j]) # Min deposit size

        # Link is_conv to c
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                if i != j:
                    self.solver.Add(self.c[i, j] <= big_M * self.is_conv[i, j])
                    self.solver.Add(self.c[i, j] >= 0.01 * self.is_conv[i, j]) # Min conversion

        # 9. Min Pools
        min_pools = self.alloc_params.get('min_pools', 0)
        if min_pools > 0:
            self.solver.Add(self.solver.Sum([self.h[i] for i in range(self.n_pools)]) >= min_pools)

        # --- Objective Function ---
        # Maximize Annual Yield - Total Costs
        
        # Annual Yield -> Horizon Yield
        # maximize (DailyYield * Horizon) - TotalCosts
        
        daily_apy_matrix = np.zeros((self.n_pools, self.n_tokens))
        for _, row in self.pools_df.iterrows():
            pool_id = row['pool_id']
            i = self.pool_idx[pool_id]
            daily_apy = row['forecasted_apy'] / 100.0 / 365.0
            for token in self.pool_tokens.get(pool_id, []):
                if token in self.token_idx:
                    j = self.token_idx[token]
                    daily_apy_matrix[i, j] = daily_apy

        # Calculate Yield over the specific Horizon
        horizon_yield = self.solver.Sum([
            self.x[i, j] * daily_apy_matrix[i, j] * price_vector[j] * self.optimization_horizon_days
            for i in range(self.n_pools) for j in range(self.n_tokens)
        ])
        
        self.solver.Maximize(horizon_yield - total_costs)
        
        # Save expressions for result extraction
        self.objective_expr = horizon_yield - total_costs
        self.total_costs_expr = total_costs
        
    def solve(self) -> bool:
        if not self.solver:
            self.build_model()
            
        if self.solver:
            self.solver.EnableOutput()
            
        logger.info(f"Solving with {self.solver.SolverVersion()}...")
        
        # Debug: Check feasibility of min_pools
        min_pools = self.alloc_params.get('min_pools', 0)
        pool_tvl_min_limit = float(self.alloc_params.get('pool_tvl_limit', 0) or 0)
        
        eligible_pools = 0
        for pool_id, tvl in self.pool_tvl.items():
            if tvl >= pool_tvl_min_limit:
                eligible_pools += 1
                
        logger.info(f"Debug: Min Pools req: {min_pools}, Eligible Pools (TVL >= {pool_tvl_min_limit}): {eligible_pools}")
        
        if eligible_pools < min_pools:
            logger.error(f"INFEASIBILITY LIKELY: Only {eligible_pools} pools meet TVL limit, but {min_pools} required.")

        # Set limits
        self.solver.SetTimeLimit(300 * 1000) # 300 seconds (ms)
        
        status = self.solver.Solve()
        
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            logger.info(f"✓ Optimization successful: {status}")
            logger.info(f"  Objective value: ${self.solver.Objective().Value():,.4f}")
            return True
        else:
            logger.error(f"✗ Optimization failed: {status}")
            return False

    def extract_results(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """Extracts results from solved optimization model."""
        if not self.solver:
            return pd.DataFrame(), []

        if not hasattr(self, 'x') or not hasattr(self, 'w') or not hasattr(self, 'd') or not hasattr(self, 'c'):
            logger.error("Solver variables not initialized. Call solve() first.")
            return pd.DataFrame(), []

        if self.n_pools == 0 or self.n_tokens == 0:
            logger.warning("No pools or tokens available for result extraction.")
            return pd.DataFrame(), []

        allocations = []
        transactions = []
        transaction_seq = 1
        price_vector = np.array([self.token_prices.get(t, 1.0) for t in self.tokens])

        # Helpers to get values safely
        def get_val(var_dict, *args):
            try:
                val = var_dict[args].solution_value()
                return val if val is not None else 0.0
            except (KeyError, AttributeError):
                return 0.0

        # 1. Holds
        for (pool_id, token), current_amt in self.current_allocations.items():
            if pool_id in self.pool_idx and token in self.token_idx:
                i = self.pool_idx[pool_id]
                j = self.token_idx[token]

                withdraw_amt = get_val(self.w, i, j)
                kept_amt = current_amt - withdraw_amt

                if kept_amt > 0.01:
                    transactions.append({
                        'seq': transaction_seq,
                        'type': 'HOLD',
                        'from_location': pool_id,
                        'to_location': pool_id,
                        'token': token,
                        'amount': float(kept_amt),
                        'amount_usd': float(kept_amt * price_vector[j]),
                        'gas_cost_usd': 0.0,
                        'conversion_cost_usd': 0.0,
                        'total_cost_usd': 0.0
                    })
                    transaction_seq += 1

        # 2. Withdrawals
        for i in range(self.n_pools):
            pool_id = self.pools[i]
            for j in range(self.n_tokens):
                amount = get_val(self.w, i, j)
                if amount > 0.01:
                    transactions.append({
                        'seq': transaction_seq,
                        'type': 'WITHDRAWAL',
                        'from_location': pool_id,
                        'to_location': 'warm_wallet',
                        'token': self.tokens[j],
                        'amount': float(amount),
                        'amount_usd': float(amount * price_vector[j]),
                        'gas_cost_usd': float(self.withdrawal_gas_fee),
                        'conversion_cost_usd': 0.0,
                        'total_cost_usd': float(self.withdrawal_gas_fee)
                    })
                    transaction_seq += 1

        # 3. Conversions
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                if i != j:
                    amount = get_val(self.c, i, j)
                    if amount > 0.01:
                        trans_cost = float(self.conversion_gas_fee)
                        conv_cost = float(amount * price_vector[i] * self.conversion_rate)
                        transactions.append({
                            'seq': transaction_seq,
                            'type': 'CONVERSION',
                            'from_location': 'warm_wallet',
                            'to_location': 'warm_wallet',
                            'from_token': self.tokens[i],
                            'to_token': self.tokens[j],
                            'amount': float(amount),
                            'amount_usd': float(amount * price_vector[i]),
                            'conversion_cost_usd': conv_cost,
                            'gas_cost_usd': trans_cost,
                            'total_cost_usd': trans_cost + conv_cost
                        })
                        transaction_seq += 1

        # 4. Allocations (Deposits) & Final State
        for i in range(self.n_pools):
            pool_id = self.pools[i]
            # Handle potential empty dataframe result safely
            pool_info = self.pools_df[self.pools_df['pool_id'] == pool_id]
            symbol = pool_info['symbol'].iloc[0] if not pool_info.empty else "UNKNOWN"

            for j in range(self.n_tokens):
                token = self.tokens[j]
                final_amt = get_val(self.x, i, j)
                deposit_amt = get_val(self.d, i, j)

                if final_amt > 0.01:
                    allocations.append({
                        'pool_id': pool_id,
                        'pool_symbol': symbol,
                        'token': token,
                        'amount': float(final_amt),
                        'amount_usd': float(final_amt * price_vector[j]),
                        'needs_conversion': bool(deposit_amt > 0.01)
                    })

                if deposit_amt > 0.01:
                    transactions.append({
                        'seq': transaction_seq,
                        'type': 'ALLOCATION',
                        'from_location': 'warm_wallet',
                        'to_location': pool_id,
                        'token': token,
                        'amount': float(deposit_amt),
                        'amount_usd': float(deposit_amt * price_vector[j]),
                        'conversion_cost_usd': 0.0,
                        'gas_cost_usd': float(self.allocation_gas_fee),
                        'total_cost_usd': float(self.allocation_gas_fee),
                        'needs_conversion': False
                    })
                    transaction_seq += 1

        return pd.DataFrame(allocations), transactions

    def format_results(self):
        """Formats optimization results match the existing output structure."""
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
            if self.y[j].solution_value() > 0.01:
                amt = self.y[j].solution_value()
                unallocated_tokens[self.tokens[j]] = {
                    "amount": amt, "amount_usd": amt * price_vector[j]
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
            formatted_transactions.append(ftxn)
            
        return {
            "final_allocations": final_allocations,
            "unallocated_tokens": unallocated_tokens,
            "transactions": formatted_transactions
        }


# ============================================================================
# MAIN ORCHESTRATION (Reused but pointing to new Optimizer)
# ============================================================================

def delete_todays_allocations():
    # ... reused ...
    repo = AllocationRepository()
    try:
        deleted = repo.delete_allocations_for_date(date.today())
        logger.info(f"Deleted {deleted} existing allocation records for today")
    except Exception as e:
        logger.error(f"Error deleting today's allocations: {e}")

def store_results(run_id: str, allocations_df: pd.DataFrame, 
                  transactions: List[Dict], alloc_params: Dict):
    # ... reused ...
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
    # ... reused ...
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

def optimize_allocations(custom_overrides: Dict = None):
    logger.info("=" * 80)
    logger.info("STABLECOIN POOL ALLOCATION OPTIMIZATION (OR-TOOLS)")
    logger.info("=" * 80)
    
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
    run_id = alloc_params.get('run_id')
    if not run_id:
        run_id = str(uuid4())
        logger.info(f"Generated new run_id: {run_id}")
    
    # 3. Initialize Optimizer
    logger.info("\n[3/6] Initializing optimizer...")
    optimizer = AllocationOptimizer(pools_df, token_prices, warm_wallet, current_allocations, gas_fees, alloc_params)
    
    # 4. Solve
    logger.info("\n[4/6] Solving optimization problem...")
    if optimizer.solve():
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
    else:
        logger.error("Optimization failed to find a solution.")

if __name__ == "__main__":
    optimize_allocations()
