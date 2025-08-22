# Phase 3: Data Processing & Transformation Layer

This phase focuses on processing the raw ingested data and transforming it into a format suitable for analysis, forecasting, and optimization.

## Detailed Tasks:

### 3.1 Implement `process_icebox_logic.py` - **COMPLETED**
- [x] Create a Python script to:
    - Read OHLCV data for approved stablecoins from the relevant raw data table (e.g., `raw_coinmarketcap_ohlcv`).
    - Apply Icebox rules based on configurable thresholds (`icebox_ohlc_l_threshold_pct`, `icebox_ohlc_l_days_threshold`, `icebox_ohlc_c_threshold_pct`, `icebox_ohlc_c_days_threshold`, `icebox_recovery_l_days_threshold`, `icebox_recovery_c_days_threshold`).
    - Identify tokens that breach thresholds and add them to the `icebox_tokens` table with a timestamp and reason.
    - Identify tokens that meet recovery criteria and remove them from the `icebox_tokens` table (by updating `removed_timestamp`).
    - Ensure only tokens from the `approved_tokens` set are eligible for Icebox logic.

### 3.2 Implement `calculate_pool_metrics.py` - **COMPLETED**
- [x] Create a Python script to:
    - Read historical pool data from `raw_defillama_pool_history` and `pools` tables.
    - Calculate 7-day and 30-day rolling APY for each pool.
    - Calculate Today - Yesterday APY delta.
    - Calculate 7-day APY standard deviation delta.
    - Calculate 30-day APY standard deviation delta.
    - Store these calculated metrics in the `pool_daily_metrics` table.

### 3.3 Implement `apply_pool_grouping.py` - **COMPLETED**
- [x] Create a Python script to:
    - Read calculated metrics from `pool_daily_metrics`.
    - Apply the pool grouping logic based on universal thresholds:
        - **Group 1:** ΔAPY ≤ 1% AND Δ7DSTDDEV ≤ 1.5% AND Δ30DSTDDEV ≤ 2%
        - **Group 2:** ΔAPY ≤ 3% AND Δ7DSTDDEV ≤ 4% AND Δ30DSTDDEV ≤ 5%
        - **Group 3:** ΔAPY > 3% AND Δ7DSTDDEV > 4% AND Δ30DSTDDEV > 2%
    - Assign pools to Group 1, Group 2, or Group 3.
    - Store pool grouping assignments in the `pool_groups` table, including the criteria snapshot for reproducibility.
    - Store mean volatility values per pool for reference.

### 3.4 Implement `filter_pools.py` - **COMPLETED**
- [x] Create a Python script to:
    - Read pool data and metrics from `pools` and `pool_daily_metrics`.
    - Read `approved_protocols`, `approved_tokens`, `blacklisted_tokens`, and `icebox_tokens` lists.
    - Apply all pool filtering criteria:
        - Exclude pools not belonging to approved protocols.
        - Exclude pools not containing approved tokens.
        - Exclude pools containing blacklisted tokens.
        - Exclude pools containing any Icebox tokens.
        - Apply additional trade logic filters:
            - Exclude tokens with MarketCap < `token_marketcap_limit`.
            - Exclude pools with TVL < `pool_tvl_limit`.
            - Exclude pools with APY < `pool_apy_limit`.
            - Exclude pools where side of pair is not within `pool_pair_tvl_ratio_min` - `pool_pair_tvl_ratio_max` of other side's TVL.
    - Mark filtered-out pools as `is_filtered_out = TRUE` in `pool_daily_metrics` and record the `filter_reason`.
    - Ensure that the filtering logic uses the snapshot of dynamic lists from `allocation_parameters` for reproducibility during backtesting.