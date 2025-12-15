import logging
import pandas as pd
import numpy as np
import cvxpy as cp
import json
import os
from datetime import date
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# DATA SETUP
# ============================================================================

# 1. Pools Data 
def load_pools_data():
    try:
        # Check adjacent file first
        file_path = os.path.join(os.path.dirname(__file__), 'initial_pools_data.json')
        if not os.path.exists(file_path):
             # Fallback to relative path from root
             file_path = 'asset_allocation/initial_pools_data.json'
             
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()

        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Enrich data since JSON source (metrics dump) lacks static pool info like symbol/tokens
        # We use a deterministic distribution of tokens to test the optimizer
        tokens_list = ["USDC", "USDF", "USDT", "PYUSD", "USDe", "USDS"]
        
        enriched = []
        for i, row in enumerate(data):
            p = row.copy()
            if 'symbol' not in p:
                p['symbol'] = f"Pool-{p['pool_id'][:6]}"
            if 'underlying_tokens' not in p:
                # Round robin assignment of tokens
                p['underlying_tokens'] = [tokens_list[i % len(tokens_list)]]
            
            # Ensure proper types
            if 'forecasted_apy' in p:
                p['forecasted_apy'] = float(p['forecasted_apy'])
            if 'forecasted_tvl' in p:
                p['forecasted_tvl'] = float(p['forecasted_tvl'])
                
            enriched.append(p)
            
        logger.info(f"Loaded {len(enriched)} pools from {file_path}")
        return pd.DataFrame(enriched)
    except Exception as e:
        logger.error(f"Failed to load specific data file: {e}")
        return pd.DataFrame()

POOLS_DF = load_pools_data()

# 2. Allocation Parameters
ALLOC_PARAMS = {
    "run_id": "376b479b-2952-47c9-943f-1e04898f0998",
    "tvl_limit_percentage": 0.05,
    "max_alloc_percentage": 0.25,
    "conversion_rate": 0.0004,
    "min_pools": 4,
    "min_transaction_value": 50.0
}

# 3. Prices & Fees
TOKEN_PRICES = {
    "USDC": 1.0, "USDF": 1.0, "USDT": 1.0, "USDe": 1.0, "USDS": 1.0, 
    "PYUSD": 1.0, "crvUSD": 1.0, "SUSDE": 1.0, "ETH": 3060.0
}
GAS_FEES = {
    'allocation': 1.28,
    'withdrawal': 1.28,
    'conversion': 1.28,
    'transfer': 1.28,
    'deposit': 1.28
}

# 4. Initial Balances
EXISTING_ALLOCATIONS = {
    ("9b9e6e11-f85d-438d-91d7-da194f59f5a9", "USDF"): 979.03,
    ("c743eac6-1906-4d96-a7aa-667426ec3cc7", "USDT"): 1199.00,
    ("a948bde2-6f17-4b0b-9f76-c184b11b9618", "USDe"): 960.92,
    ("7501ef09-87d1-405c-b2b2-f269b2727289", "USDS"): 16.70,
    # [TEST SCENARIO]: Existing allocation in a high-performing pool
    ("89c46065-d437-4b44-aebb-8f2fd342cf75", "USDS"): 5000.00 
}

WARM_WALLET = {
    "PYUSD": 12600.0,
    "SUSDE": 4210.0,
    "USDC": 4210.0,
    "USDF": 4137.0,
    "USDe": 1000.0,
    "USDS": 0.0
}

# ============================================================================
# OPTIMIZER CLASS (FIXED for Incremental Adjustments)
# ============================================================================

class AllocationOptimizerFixed:
    def __init__(self, pools_df, token_prices, warm_wallet, current_allocations, gas_fees, alloc_params):
        self.pools_df = pools_df
        self.token_prices = token_prices
        self.warm_wallet = warm_wallet
        self.current_allocations = current_allocations
        self.gas_fees = gas_fees
        self.alloc_params = alloc_params
        
        self.allocation_gas_fee = gas_fees['allocation']
        self.withdrawal_gas_fee = gas_fees['withdrawal']
        self.conversion_gas_fee = gas_fees['conversion']
        
        # Build Universe
        self.tokens = sorted(list(set(warm_wallet.keys()) | set([t for sublist in pools_df['underlying_tokens'] for t in sublist]) | set([k[1] for k in current_allocations.keys()])))
        self.pools = pools_df['pool_id'].tolist()
        
        self.n_tokens = len(self.tokens)
        self.n_pools = len(self.pools)
        
        self.token_idx = {t: i for i, t in enumerate(self.tokens)}
        self.pool_idx = {p: i for i, p in enumerate(self.pools)}
        
        self.pool_tokens = {row['pool_id']: row['underlying_tokens'] for _, row in pools_df.iterrows()}
        self.pool_tvl = {row['pool_id']: row['forecasted_tvl'] for _, row in pools_df.iterrows()}
        
        # Handle existing allocs not in pools_df
        existing_pool_ids = set(k[0] for k in current_allocations.keys())
        missing_pools = existing_pool_ids - set(self.pools)
        if missing_pools:
            logger.info(f"Adding {len(missing_pools)} missing allocated pools to tracking index (Withdrawal only)")
            for pid in missing_pools:
                self.pools.append(pid)
                self.pool_tokens[pid] = ["USDC"]
                self.pool_tvl[pid] = 0
                self.pool_idx[pid] = len(self.pools) - 1
        
        self.n_pools = len(self.pools)
        
        self.total_aum = sum(warm_wallet.get(t, 0) * token_prices.get(t, 1) for t in warm_wallet) + \
                         sum(amount * token_prices.get(t, 1) for (pid, t), amount in current_allocations.items())
        
        self.conversion_rate = alloc_params['conversion_rate']
        self.max_alloc_percentage = alloc_params['max_alloc_percentage']
        self.tvl_limit_percentage = alloc_params['tvl_limit_percentage']
        self.min_pools = alloc_params.get('min_pools', 0)

        logger.info(f"Initialized Optimizer: AUM=${self.total_aum:,.2f}, Min Pools={self.min_pools}")

    def solve(self):
        logger.info("Building model with incremental adjustment support...")
        
        # ============ KEY FIX: Model final allocations directly ============
        # Instead of treating allocations and withdrawals as independent,
        # we model the FINAL position and compute the delta from current position.
        # This allows the optimizer to:
        #   - Keep positions unchanged (no cost)
        #   - Make incremental increases (deposit only)
        #   - Make incremental decreases (withdrawal only)
        
        self.final_alloc = cp.Variable((self.n_pools, self.n_tokens), nonneg=True)
        self.convert = cp.Variable((self.n_tokens, self.n_tokens), nonneg=True)
        self.final_warm_wallet = cp.Variable(self.n_tokens, nonneg=True)
        
        # Compute changes from current allocations
        curr_alloc_mat = np.zeros((self.n_pools, self.n_tokens))
        for (pid, t), amt in self.current_allocations.items():
            if pid in self.pool_idx and t in self.token_idx:
                curr_alloc_mat[self.pool_idx[pid], self.token_idx[t]] = amt
        
        # Calculate net deposits and withdrawals (the delta)
        self.net_deposit = cp.Variable((self.n_pools, self.n_tokens), nonneg=True)
        self.net_withdraw = cp.Variable((self.n_pools, self.n_tokens), nonneg=True)
        
        # Binary indicators for transactions
        self.has_deposit = cp.Variable((self.n_pools, self.n_tokens), boolean=True)
        self.has_withdrawal = cp.Variable((self.n_pools, self.n_tokens), boolean=True)
        self.has_allocation = cp.Variable(self.n_pools, boolean=True)
        self.is_conversion = cp.Variable((self.n_tokens, self.n_tokens), boolean=True)

        # Vectors
        price_vector = np.array([self.token_prices.get(t, 1.0) for t in self.tokens])
        
        daily_apy_matrix = np.zeros((self.n_pools, self.n_tokens))
        for _, row in self.pools_df.iterrows():
            if row['pool_id'] in self.pool_idx:
                i = self.pool_idx[row['pool_id']]
                daily_apy = row['forecasted_apy'] / 100.0 / 365.0
                for t in self.pool_tokens[row['pool_id']]:
                    if t in self.token_idx:
                        daily_apy_matrix[i, self.token_idx[t]] = daily_apy

        # 1. Yield Calculation (Daily) - use FINAL allocations
        yield_usd = cp.sum(cp.multiply(cp.multiply(self.final_alloc, daily_apy_matrix), price_vector))
        
        # 2. Transaction Costs - only for actual changes
        withdrawal_gas_costs = cp.sum(cp.multiply(self.has_withdrawal, self.withdrawal_gas_fee))
        deposit_gas_costs = cp.sum(cp.multiply(self.has_deposit, self.allocation_gas_fee))
        
        # Conversions
        conversion_fee_costs = cp.sum(cp.multiply(cp.multiply(self.convert, price_vector), self.conversion_rate))
        conversion_gas_costs = cp.sum(cp.multiply(self.is_conversion, self.conversion_gas_fee))
        
        total_txn_costs = withdrawal_gas_costs + deposit_gas_costs + conversion_fee_costs + conversion_gas_costs
        
        # Objective: Maximize (Annualized Yield - Transaction Costs)
        annual_yield = yield_usd * 365.0
        objective_expr = annual_yield - total_txn_costs
        objective = cp.Maximize(objective_expr)
        
        constraints = []
        
        # ============ KEY FIX: Link final position to changes ============
        # final_alloc = current_alloc + net_deposit - net_withdraw
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                constraints.append(
                    self.final_alloc[i, j] == curr_alloc_mat[i, j] + self.net_deposit[i, j] - self.net_withdraw[i, j]
                )
        
        # Balance Conservation
        warm_wallet_vec = np.zeros(self.n_tokens)
        for t, amt in self.warm_wallet.items():
            if t in self.token_idx: 
                warm_wallet_vec[self.token_idx[t]] = amt
            
        for j in range(self.n_tokens):
            # Money in: wallet + withdrawals + conversions to this token
            in_flow = warm_wallet_vec[j] + cp.sum(self.net_withdraw[:, j]) + cp.sum(self.convert[:, j])
            # Money out: final wallet + deposits + conversions from this token
            out_flow = self.final_warm_wallet[j] + cp.sum(self.net_deposit[:, j]) + cp.sum(self.convert[j, :])
            constraints.append(in_flow == out_flow)
        
        # No self conversion
        for j in range(self.n_tokens):
            constraints.append(self.convert[j, j] == 0)
            
        # Max Alloc & TVL Limits
        max_usd_per_pool = self.max_alloc_percentage * self.total_aum
        for i in range(self.n_pools):
            pool_val_usd = cp.sum(cp.multiply(self.final_alloc[i, :], price_vector))
            constraints.append(pool_val_usd <= max_usd_per_pool)
            
            pid = self.pools[i]
            if pid in self.pool_tvl:
                constraints.append(pool_val_usd <= self.tvl_limit_percentage * self.pool_tvl[pid])
        
        # AUM Conservation (with small tolerance for numerical stability)
        total_alloc_usd = cp.sum(cp.multiply(self.final_alloc, price_vector))
        final_wallet_usd = cp.sum(cp.multiply(self.final_warm_wallet, price_vector))
        # Allow small slack for numerical precision
        constraints.append(total_alloc_usd + final_wallet_usd + total_txn_costs <= self.total_aum * 1.001)
        
        # --- Logic Linking Constraints ---
        big_M = self.total_aum
        
        # Link has_allocation to final position
        for i in range(self.n_pools):
            pool_val = cp.sum(self.final_alloc[i, :])
            constraints.append(pool_val <= big_M * self.has_allocation[i])
            constraints.append(pool_val >= 0.001 * self.has_allocation[i])
        
        # Min Pools Constraint
        if self.min_pools > 0:
            constraints.append(cp.sum(self.has_allocation) >= self.min_pools)
        
        # Link transaction indicators to actual changes
        # With continuous relaxation, we use soft linking
        # If amount > 0, the indicator will be pushed toward 1 by the division
        
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                # Link withdrawal indicator: if withdraw > 0, has_withdrawal ≈ 1
                # We use: has_withdrawal >= withdraw / (withdraw + epsilon)
                # This approaches 1 as withdraw increases, and 0 when withdraw = 0
                constraints.append(self.net_withdraw[i, j] <= big_M * self.has_withdrawal[i, j])
                
                # Link deposit indicator similarly
                constraints.append(self.net_deposit[i, j] <= big_M * self.has_deposit[i, j])

        # Link allocation indicator
        for i in range(self.n_pools):
            pool_val = cp.sum(self.final_alloc[i, :])
            constraints.append(pool_val <= big_M * self.has_allocation[i])
            # Minimum threshold: if pool has any significant allocation, indicator should be high
            # Using a small threshold (0.1% of max allocation)
            min_threshold = 0.001 * self.max_alloc_percentage * self.total_aum
            constraints.append(pool_val >= min_threshold * self.has_allocation[i])

        # Link conversion indicators  
        for i in range(self.n_tokens):
            for j in range(self.n_tokens):
                if i != j:
                    constraints.append(self.convert[i, j] <= big_M * self.is_conversion[i, j])
        
        # Non-negativity
        constraints.append(self.final_alloc >= 0)
        constraints.append(self.net_deposit >= 0)
        constraints.append(self.net_withdraw >= 0)
        constraints.append(self.convert >= 0)
        constraints.append(self.final_warm_wallet >= 0)
        
        # --- Solve ---
        problem = cp.Problem(objective, constraints)
        
        # Diagnostic info before solving
        logger.info(f"Problem has {len(constraints)} constraints")
        logger.info(f"Variables: final_alloc ({self.n_pools}x{self.n_tokens}), "
                   f"convert ({self.n_tokens}x{self.n_tokens}), "
                   f"warm_wallet ({self.n_tokens})")
        
        logger.info("Solving...")
        try:
            problem.solve(solver=cp.HIGHS, verbose=False)
        except Exception as e:
            logger.info(f"HIGHS failed with: {e}, trying ECOS_BB...")
            try:
                problem.solve(solver=cp.ECOS_BB, verbose=False)
            except Exception as e2:
                logger.info(f"ECOS_BB failed with: {e2}, trying GLPK_MI...")
                try:
                    problem.solve(solver=cp.GLPK_MI, verbose=False)
                except Exception as e3:
                    logger.info(f"GLPK_MI failed with: {e3}")
                    logger.info("Trying without integer constraints...")
                    # Relax integer constraints as a diagnostic
                    problem_relaxed = cp.Problem(objective, constraints)
                    problem_relaxed.solve(solver=cp.ECOS, verbose=False)
                    logger.info(f"Relaxed problem status: {problem_relaxed.status}")
            
        logger.info(f"Status: {problem.status}")
        if problem.value is not None:
            logger.info(f"Objective Value (Net Annual Yield): ${problem.value:,.2f}")
        else:
            logger.info("Objective Value: None (infeasible)")
            logger.info("\nDiagnostic: Checking basic feasibility...")
            logger.info(f"Total AUM: ${self.total_aum:,.2f}")
            logger.info(f"Current allocations (USD): ${sum(amt * self.token_prices.get(t, 1) for (pid, t), amt in self.current_allocations.items()):,.2f}")
            logger.info(f"Warm wallet (USD): ${sum(amt * self.token_prices.get(t, 1) for t, amt in self.warm_wallet.items()):,.2f}")
            logger.info(f"Min pools required: {self.min_pools}")
            logger.info(f"Available pools: {len([p for p in self.pools if self.pool_tvl.get(p, 0) > 0])}")
        
        return self.print_results(problem, curr_alloc_mat)

    def print_results(self, problem, curr_alloc_mat):
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print("Optimization FAILED.")
            return

        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS - WITH INCREMENTAL ADJUSTMENTS")
        print("="*80)
        
        # Track changes by pool
        pool_changes = {}
        
        for i in range(self.n_pools):
            for j in range(self.n_tokens):
                curr = curr_alloc_mat[i, j]
                final = self.final_alloc[i, j].value if self.final_alloc[i, j].value else 0
                deposit = self.net_deposit[i, j].value if self.net_deposit[i, j].value else 0
                withdraw = self.net_withdraw[i, j].value if self.net_withdraw[i, j].value else 0
                
                if curr > 1 or final > 1 or abs(deposit) > 1 or abs(withdraw) > 1:
                    pid = self.pools[i]
                    if pid not in pool_changes:
                        sym = next((x['symbol'] for _, x in self.pools_df.iterrows() if x['pool_id'] == pid), f"Pool-{pid[:6]}")
                        pool_changes[pid] = {'symbol': sym, 'changes': []}
                    
                    tok = self.tokens[j]
                    pool_changes[pid]['changes'].append({
                        'token': tok,
                        'current': curr,
                        'final': final,
                        'deposit': deposit,
                        'withdraw': withdraw,
                        'net_change': final - curr
                    })
        
        # Print changes grouped by action type
        print("\n📊 POSITION CHANGES:")
        print("-" * 80)
        
        kept_positions = []
        increased_positions = []
        decreased_positions = []
        new_positions = []
        closed_positions = []
        
        for pid, info in pool_changes.items():
            for change in info['changes']:
                entry = f"{info['symbol']}: {change['token']}"
                
                if change['current'] > 1 and change['final'] > 1:
                    if abs(change['net_change']) < 1:
                        kept_positions.append(f"  ✓ KEPT: {entry} = ${change['final']:.2f}")
                    elif change['net_change'] > 1:
                        increased_positions.append(
                            f"  ↗ INCREASED: {entry} from ${change['current']:.2f} to ${change['final']:.2f} "
                            f"(+${change['deposit']:.2f})"
                        )
                    else:
                        decreased_positions.append(
                            f"  ↘ DECREASED: {entry} from ${change['current']:.2f} to ${change['final']:.2f} "
                            f"(-${change['withdraw']:.2f})"
                        )
                elif change['current'] > 1 and change['final'] < 1:
                    closed_positions.append(f"  ✗ CLOSED: {entry} was ${change['current']:.2f}")
                elif change['current'] < 1 and change['final'] > 1:
                    new_positions.append(f"  ✦ NEW: {entry} = ${change['final']:.2f}")
        
        if kept_positions:
            print("\n🔒 Positions Maintained (No Action):")
            for p in kept_positions: print(p)
        
        if increased_positions:
            print("\n📈 Positions Increased (Deposit):")
            for p in increased_positions: print(p)
        
        if decreased_positions:
            print("\n📉 Positions Decreased (Withdrawal):")
            for p in decreased_positions: print(p)
        
        if new_positions:
            print("\n✨ New Positions Opened:")
            for p in new_positions: print(p)
        
        if closed_positions:
            print("\n🚪 Positions Closed:")
            for p in closed_positions: print(p)
        
        # Summary stats
        total_alloc = sum(
            self.final_alloc[i, j].value * self.token_prices[self.tokens[j]]
            for i in range(self.n_pools)
            for j in range(self.n_tokens)
            if self.final_alloc[i, j].value and self.final_alloc[i, j].value > 1
        )
        
        allocated_pools_count = sum(
            1 for i in range(self.n_pools)
            if sum(self.final_alloc[i, j].value or 0 for j in range(self.n_tokens)) > 1
        )
        
        print("\n" + "="*80)
        print(f"Total Allocated: ${total_alloc:,.2f} / ${self.total_aum:,.2f} ({total_alloc/self.total_aum*100:.1f}%)")
        print(f"Unique Pools: {allocated_pools_count} (Min Required: {self.min_pools})")
        print(f"Net Annual Yield (after costs): ${problem.value:,.2f}")
        print("="*80)
        
        return problem.status

if __name__ == "__main__":
    optimizer = AllocationOptimizerFixed(POOLS_DF, TOKEN_PRICES, WARM_WALLET, EXISTING_ALLOCATIONS, GAS_FEES, ALLOC_PARAMS)
    optimizer.solve()