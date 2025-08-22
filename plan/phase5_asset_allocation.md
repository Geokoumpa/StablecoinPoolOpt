# Phase 5: Asset Allocation Layer

This phase focuses on implementing the core asset allocation optimization logic and the snapshotting functionality for reproducibility.

## Detailed Tasks:

### 5.1 Implement `optimize_allocations.py`
- [x] Create a Python script to:
    - [x] Read forecasted APY and TVL for selected pools from `pool_daily_metrics`.
    - [x] Read current account balances and liquidity portfolio from relevant tables (e.g., `daily_ledger` or direct from Etherscan data).
    - [x] Read configurable parameters from `allocation_parameters` (or a configuration service).
    - [x] Implement the optimization problem using `cvxpy`, defining:
        - [x] **Objective Function:** Maximize (daily_yield * total_usd - total_gas_fees * total_usd - total_conversion_penalty * total_usd).
        - [x] **Constraints:**
            - [x] Sum of Weights Equals 1 (Full Allocation).
            - [x] Allocation Amounts Less Than or Equal to Per-Pool TVL Limit (`tvl_limit_percentage`).
            - [x] Non-Negative Weights.
            - [x] Binary Variable Linkage (Pool Selection). *(Note: Fully implemented in MILP script, conceptual in CVXPY)*
            - [x] Minimum Number of Pools (`min_pools`). *(Note: Fully implemented in MILP script, conceptual in CVXPY)*
            - [x] Maximum Allocation Percentage (`max_alloc_percentage`, conditional on `profit_optimization`).
            - [x] Position Limits (`position_max_pct_total_assets`, `position_max_pct_pool_tvl`).
            - [x] Group Allocation Limits (`group1_max_pct`, `group2_max_pct`, `group3_max_pct`).
    - [x] Implement the `profit_optimization` flag logic to dynamically adjust constraints.
    - [x] Compare net forecasted yield (including fees) vs. net yield from existing allocation to decide on full reallocation or reallocation of only yesterday's yield.
    - [x] Force reallocation if any token, pool, or protocol in the current allocation is no longer approved or available (e.g., moved to Icebox, blacklisted, or protocol removed).
    - [x] Store optimization parameters (including snapshots of dynamic lists) in the `allocation_parameters` table.
    - [x] Store allocation results in `asset_allocations` and `asset_allocation_details` tables. *(Note: `asset_allocation_details` deferred)*

### 5.1.1 Implement `optimize_allocations_milp.py` (Alternative MILP Approach)
- [x] Create a Python script for MILP optimization using CP-SAT.
- [x] Implement binary variables and linking constraints for pool selection.
- [x] Adapt objective and other constraints for CP-SAT.

### 5.2 Implement Snapshotting Functionality
- [x] Ensure that for each optimization run, the system creates a snapshot of all dynamic configuration lists (approved tokens, blacklisted tokens, approved protocols, icebox tokens).
- [x] Store these snapshots as JSONB fields within the `allocation_parameters` table to ensure full reproducibility and auditability of each run.
- [x] Develop helper functions or classes to manage the creation and retrieval of these snapshots.

### 5.3 Future Refinement: Implement `asset_allocation_details` Table
- Define schema for `asset_allocation_details` to store granular information about each allocation (e.g., token breakdown, specific transaction costs per pool).
- Modify `store_asset_allocations` to populate this new table.

### 5.4 Future Refinement: Enhance `should_force_reallocation`
- Improve the robustness of the `should_force_reallocation` function by fetching detailed pool information (e.g., token composition, protocol status) and comparing against dynamic lists more accurately.