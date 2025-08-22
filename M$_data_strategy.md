thinking
The user wants a new data storage strategy, focusing on database tables rather than cloud services. The current implementation uses a "bronze-silver-gold" strategy, which the user is not confident in. I need to propose a data model from scratch, considering the data sources and pipeline steps outlined in the product brief.

Here's a breakdown of the data types and their potential tables:

1.  **Raw Data (from APIs):**
    *   DeFiLlama Pool Names: Raw response from `https://yields.llama.fi/pools`
    *   DeFiLlama Pool History: Raw response from `https://yields.llama.fi/chart/{pool_id}`
    *   EthGasTracker.com Gas Fees (hourly averages): Raw response from `https://www.ethgastracker.com/api/gas/average`
    *   Etherscan Account Balances & Transactions: Raw response from `https://docs.etherscan.io/etherscan-v2/api-endpoints/accounts`
    *   CoinMarketCap OHLCV: Raw response for major cryptos and stablecoins.

2.  **Processed/Curated Data:**
    *   Filtered Pools: Pools that pass the dynamic filters, including exclusion of pools containing Icebox tokens.
    *   Icebox Tokens: Tokens temporarily excluded from allocation due to market signals, tracked with add/remove timestamps and reason.
    *   Enriched Pool History: Historical data for selected pools, potentially with calculated metrics.
    *   Forecasted Pool Data: APY and TVL forecasts for the current day.
    *   Gas Fee Actuals: Daily average gas fees.
    *   Forecasted Gas Fees: Forecasted gas fees for the current day.

I will propose a relational database schema with distinct tables for each logical entity, ensuring data integrity and efficient querying. I'll consider how data flows through the pipeline steps and how it's transformed.

**Proposed Tables:**

*   **`raw_defillama_pools_snapshot`**: Stores raw JSON responses from the DeFiLlama pools endpoint.
*   **`raw_defillama_pool_history_entries`**: Stores raw JSON responses for individual pool histories.
*   **`raw_ethgastracker_hourly_gas_data`**: Stores raw hourly average gas data from EthGasTracker.com.
*   **`raw_etherscan_account_transactions`**: Stores raw transaction history from Etherscan Accounts API.
*   **`raw_etherscan_account_balances`**: Stores raw historical balance data from Etherscan Accounts API.
*   **`raw_coinmarketcap_ohlcv`**: Stores raw OHLCV data for all cryptocurrencies (major cryptos and stablecoins).

*   **`defi_pools`**: Master table for all identified DeFi pools, with core metadata.
*   **`pool_daily_summary`**: Stores daily historical and forecasted metrics for pools (APY, TVL).
*   **`gas_fees_daily_summary`**: Stores daily actual and forecasted gas prices.
*   **`icebox_tokens`**: Stores the current Icebox token list, with timestamps for when tokens are added/removed and the reason for exclusion.
*   **`approved_tokens`**: Tracks the list of approved tokens for each run, with add/remove timestamps.
*   **`blacklisted_tokens`**: Tracks the list of blacklisted tokens for each run, with add/remove timestamps.
*   **`approved_protocols`**: Tracks the list of approved protocols for each run, with add/remove timestamps.
*   **`pool_groups`**: Stores pool grouping (Group 1, 2, 3) and their assignment per day, including criteria snapshot.
*   **`daily_ledger`**: Tracks daily balances and NAV for each token.
*   **`optimization_runs`**: Records each optimization run with its parameters and overall results.
*   **`daily_allocations`**: Stores the detailed breakdown of each daily asset allocation.
    *   `nav_usd` (NUMERIC, NOT NULL)
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())

Let's define the schema for each table.

**Table Schemas:**

1.  **`raw_defillama_pools`**
    *   `id` (PK, UUID/Serial)
    *   `fetch_timestamp` (TIMESTAMP, NOT NULL)
    *   `raw_data` (JSONB, NOT NULL) - Stores the entire JSON response for a given fetch.

2.  **`raw_defillama_pool_history`**
    *   `id` (PK, UUID/Serial)
    *   `pool_id` (TEXT, NOT NULL) - Reference to the DeFiLlama pool ID.
    *   `fetch_timestamp` (TIMESTAMP, NOT NULL)
    *   `raw_data` (JSONB, NOT NULL) - Stores the entire JSON response for a given pool's history.

3.  **`raw_etherscan_gas_data`**
    *   `id` (PK, UUID/Serial)
    *   `fetch_timestamp` (TIMESTAMP, NOT NULL)
    *   `raw_data` (JSONB, NOT NULL) - Stores the entire JSON response from the gas oracle.

4.  **`pools`**
    *   `pool_id` (PK, TEXT, NOT NULL) - Unique identifier from DeFiLlama.
    *   `name` (TEXT, NOT NULL)
    *   `chain` (TEXT, NOT NULL)
    *   `project` (TEXT, NOT NULL)
    *   `symbol` (TEXT)
    *   `tvl_usd` (NUMERIC) - Current TVL from the latest raw data.
    *   `apy_base` (NUMERIC) - Current APY from the latest raw data.
    *   `apy_reward` (NUMERIC)
    *   `apy` (NUMERIC)
    *   `tokens` (TEXT[]) - Array of associated token symbols.
    *   `is_filtered_out` (BOOLEAN, DEFAULT FALSE) - Flag to indicate if the pool was filtered out in the last run.
    *   `last_updated` (TIMESTAMP, NOT NULL) - Timestamp of the last update to this record.

5.  **`pool_daily_metrics`**
    *   `id` (PK, UUID/Serial)
    *   `pool_id` (FK, TEXT, NOT NULL) - References `pools.pool_id`.
    *   `date` (DATE, NOT NULL) - Date for which the metrics apply.
    *   `tvl_usd_actual` (NUMERIC) - Actual TVL for the day.
    *   `apy_actual` (NUMERIC) - Actual APY for the day.
    *   `tvl_usd_forecast` (NUMERIC) - Forecasted TVL for the current day.
    *   `apy_forecast` (NUMERIC) - Forecasted APY for the current day.
    *   `max_gas_fee_usd` (NUMERIC) - Max gas fee associated with this pool (from input).
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `updated_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `UNIQUE(pool_id, date)`

6.  **`gas_fees_5min`** (Renamed from `gas_fees_hourly` to reflect 5' interval)
    *   `id` (PK, UUID/Serial)
    *   `timestamp` (TIMESTAMP, NOT NULL)
    *   `fast_gas_price_gwei` (NUMERIC)
    *   `propose_gas_price_gwei` (NUMERIC)
    *   `safe_gas_price_gwei` (NUMERIC)
    *   `suggest_base_fee_gwei` (NUMERIC)
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `UNIQUE(timestamp)`

7.  **`gas_fees_daily`**
    *   `date` (PK, DATE, NOT NULL)
    *   `avg_gas_price_gwei_actual` (NUMERIC) - Actual average gas price for the day.
    *   `avg_gas_price_gwei_forecast` (NUMERIC) - Forecasted average gas price for the current day.
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `updated_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())

8.  **`asset_allocations`**
    *   `id` (PK, UUID/Serial)
    *   `allocation_date` (DATE, NOT NULL) - Date of this allocation.
    *   `total_aum_usd` (NUMERIC, NOT NULL) - Total AUM for this allocation run.
    *   `total_expected_profit_usd` (NUMERIC) - Maximize objective function value.
    *   `total_gas_cost_usd` (NUMERIC)
    *   `total_conversion_cost_usd` (NUMERIC)
    *   `optimization_run_id` (FK, UUID/Serial, NOT NULL) - References `optimization_runs.id`
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())

9.  **`asset_allocation_details`**
    *   `id` (PK, UUID/Serial)
    *   `allocation_id` (FK, UUID/Serial, NOT NULL) - References `asset_allocations.id`.
    *   `pool_id` (FK, TEXT, NOT NULL) - References `pools.pool_id`.
    *   `allocated_amount_usd` (NUMERIC, NOT NULL)
    *   `allocated_weight` (NUMERIC, NOT NULL)
    *   `expected_daily_apy` (NUMERIC) - APY used for this specific allocation.
    *   `expected_tvl` (NUMERIC) - TVL used for this specific allocation.
    *   `gas_fee_usd_at_allocation` (NUMERIC) - Gas fee used for this specific allocation.
    *   `conversion_penalty_at_allocation` (NUMERIC) - Conversion penalty used for this specific allocation.
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `UNIQUE(allocation_id, pool_id)`

10. **`optimization_runs`**
    *   `id` (PK, UUID/Serial)
    *   `run_timestamp` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `tvl_limit_percentage` (NUMERIC)
    *   `max_alloc_percentage` (NUMERIC)
    *   `conversion_rate` (NUMERIC)
    *   `min_pools` (INTEGER)
    *   `profit_optimization` (BOOLEAN)
    *   `initial_aum_usd` (NUMERIC)
    *   `current_allocations_json` (JSONB) - Stores the `current_allocations` input dictionary.
    *   `liquidity_portfolio_json` (JSONB) - Stores the `liquidity_portfolio` input dictionary.
    *   `status` (TEXT, NOT NULL) - e.g., 'SUCCESS', 'FAILED', 'IN_PROGRESS'
    *   `error_message` (TEXT) - If the run failed.

**Data Flow through Pipeline Steps and Table Interactions:**

*   **1) Fetch Pools:**
    *   Reads from: DeFiLlama API.
    *   Writes to: `raw_defillama_pools_snapshot`.
    *   Updates: `defi_pools` (upserting new pools, updating `tvl_usd`, `apy`, `last_updated`).

*   **2) Fetch Crypto OHLCV:**
    *   Reads from: CoinMarketCap API.
    *   Writes to: `raw_coinmarketcap_ohlcv`.

*   **3) Fetch Gas Fees:**
    *   Reads from: EthGasTracker.com API.
    *   Writes to: `raw_ethgastracker_hourly_gas_data`.
    *   Updates: `gas_fees_daily_summary` (for daily actuals).

*   **4) Fetch Account Balances & Transactions:**
    *   Reads from: Etherscan Accounts API.
    *   Writes to: `raw_etherscan_account_transactions`, `raw_etherscan_account_balances`.

*   **5) Apply Icebox Logic:**
    *   Reads from: `raw_coinmarketcap_ohlcv`.
    *   Updates: `icebox_tokens` (add/remove tokens based on thresholds).

*   **6) Calculate Pool Metrics:**
    *   Reads from: `pool_daily_summary` (historical data).
    *   Updates: `pool_daily_summary` (calculates and stores rolling APY and STDDEV).

*   **7) Apply Pool Grouping Logic:**
    *   Reads from: `pool_daily_summary` (calculated metrics).
    *   Writes to: `pool_groups`.

*   **8) Filter Pools:**
    *   Reads from: `defi_pools`, `approved_tokens`, `blacklisted_tokens`, `approved_protocols`, `icebox_tokens`, `pool_daily_summary`.
    *   Updates: `pool_daily_summary.is_filtered_out` flag.

*   **9) Fetch Full History for Selected Pools:**
    *   Reads from: `defi_pools` (filtered pools), DeFiLlama API.
    *   Writes to: `raw_defillama_pool_history_entries`.
    *   Updates: `pool_daily_summary` (upserting historical `actual_tvl_usd`, `actual_apy`).

*   **10) Retrain Forecasting Models:**
    *   Reads from: `pool_daily_summary` (historical actuals for APY and TVL), `gas_fees_daily_summary` (historical actuals for gas fees).
    *   Writes to: `model_performance_metrics` (stores retraining results).
    *   *Purpose:* Ensures models are continuously updated with the latest ground truth data.

*   **11) Forecast Pools APY and TVL:**
    *   Reads from: `pool_daily_summary` (historical data for forecasting).
    *   Updates: `pool_daily_summary` (upserting `forecasted_tvl_usd`, `forecasted_apy` for the *current* date).

*   **12) Forecast Gas Fees:**
    *   Reads from: `gas_fees_daily_summary` (for forecasting).
    *   Updates: `gas_fees_daily_summary` (upserting `forecasted_avg_gas_price_gwei` for current day).

*   **13) Optimize Asset Allocation:**
    *   Reads from: `pool_daily_summary` (forecasts), `gas_fees_daily_summary` (forecasts), `defi_pools` (for `tokens`), `current_allocations` and `liquidity_portfolio` (from input/config).
    *   Writes to: `optimization_runs` (record parameters and status), `daily_allocations`.

*   **14) Retain Daily Ledger:**
    *   Reads from: `raw_etherscan_account_balances`, `raw_coinmarketcap_ohlcv` (for token prices), `optimization_runs` (for total assets).
    *   Writes to: `daily_ledger`.

*   **15) Post to Slack Channel:**
    *   Reads from: `daily_allocations`, `optimization_runs`, `daily_ledger`. (This is an output step, not directly writing to the DB for the core pipeline).

This strategy separates raw data from processed data, allows for historical tracking of both raw and processed information, and provides clear tables for the final allocation results and the parameters that generated them. It's a more structured approach than a generic "bronze-silver-gold" and directly maps to the pipeline's data needs.Here's a proposed data storage strategy focusing on database tables, designed from scratch to support the DeFi Yield Optimizer pipeline:

The core idea is to maintain a clear separation between raw, unprocessed data, and curated, enriched, or derived data. This approach ensures data lineage, allows for reprocessing of raw data if needed, and provides optimized tables for analytical and operational queries.

**1. Raw Data Tables (Ingestion Layer):**
These tables store the direct, untransformed data fetched from external APIs. They serve as an immutable record of the source data.

*   **`raw_defillama_pools_snapshot`**
    *   `snapshot_id` (PK, UUID)
    *   `fetch_timestamp` (TIMESTAMP, NOT NULL)
    *   `raw_json_data` (JSONB, NOT NULL): Stores the complete JSON response from the `yields.llama.fi/pools` endpoint.
    *   *Purpose:* Captures the full state of all pools at a given fetch time.

*   **`raw_defillama_pool_history_entries`**
    *   `entry_id` (PK, UUID)
    *   `pool_id` (TEXT, NOT NULL)
    *   `fetch_timestamp` (TIMESTAMP, NOT NULL)
    *   `raw_json_data` (JSONB, NOT NULL): Stores the complete JSON response for a specific pool's history from `yields.llama.fi/chart/{pool_id}`.
    *   *Purpose:* Stores the granular historical data for each pool as fetched.

*   **`raw_etherscan_gas_data_5min`**
    *   `record_id` (PK, UUID)
    *   `fetch_timestamp` (TIMESTAMP, NOT NULL, UNIQUE): Timestamp of the 5-minute fetch.
    *   `raw_json_data` (JSONB, NOT NULL): Stores the complete JSON response from the Etherscan gas oracle.
    *   *Purpose:* Records the raw 5-minute gas price data.

*   **`raw_etherscan_account_transactions`**
    *   `record_id` (PK, UUID)
    *   `fetch_timestamp` (TIMESTAMP, NOT NULL)
    *   `wallet_address` (TEXT, NOT NULL)
    *   `raw_json_data` (JSONB, NOT NULL): Stores the complete JSON response for account transactions.
    *   *Purpose:* Records raw transaction history for auditing.

*   **`raw_etherscan_account_balances`**
    *   `record_id` (PK, UUID)
    *   `fetch_timestamp` (TIMESTAMP, NOT NULL)
    *   `wallet_address` (TEXT, NOT NULL)
    *   `token_symbol` (TEXT, NOT NULL)
    *   `balance` (NUMERIC, NOT NULL)
    *   `raw_json_data` (JSONB, NOT NULL): Stores the complete JSON response for account balances.
    *   *Purpose:* Records raw historical balance data for performance tracking.

*   **`raw_coinmarketcap_ohlcv`**
    *   `record_id` (PK, UUID)
    *   `fetch_timestamp` (TIMESTAMP, NOT NULL)
    *   `token_symbol` (TEXT, NOT NULL)
    *   `raw_json_data` (JSONB, NOT NULL): Stores the complete JSON response for OHLCV data.
    *   *Purpose:* Records raw OHLCV data for all cryptocurrencies (major cryptos and stablecoins).

**2. Curated Data Tables (Processing Layer):**
These tables contain structured, cleaned, and enriched data derived from the raw tables. This is where filtering, feature engineering, and forecasting results are stored.

*   **`defi_pools`**
    *   `pool_id` (PK, TEXT, NOT NULL): Unique identifier for the DeFi pool (from DefiLlama).
    *   `name` (TEXT, NOT NULL)
    *   `chain` (TEXT, NOT NULL)
    *   `project` (TEXT, NOT NULL)
    *   `symbol` (TEXT)
    *   `tokens` (TEXT[]): Array of associated token symbols (e.g., `['USDC', 'DAI']`).
    *   `is_active` (BOOLEAN, NOT NULL, DEFAULT TRUE): Flag to indicate if the pool is currently active/relevant.
    *   `last_updated_metadata` (TIMESTAMP, NOT NULL): Timestamp of the last metadata update.
    *   *Purpose:* Master data for all known DeFi pools, updated from the latest `raw_defillama_pools_snapshot`.

*   **`pool_daily_summary`**
    *   `pool_id` (FK, TEXT, NOT NULL)
    *   `date` (PK, DATE, NOT NULL)
    *   `actual_tvl_usd` (NUMERIC)
    *   `actual_apy` (NUMERIC)
    *   `forecasted_tvl_usd` (NUMERIC)
    *   `forecasted_apy` (NUMERIC)
    *   `max_gas_fee_usd` (NUMERIC): Max gas fee associated with interacting with this pool.
    *   `is_filtered_out` (BOOLEAN, NOT NULL, DEFAULT FALSE): Indicates if the pool was filtered out for this specific day's allocation run.
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `updated_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   *Purpose:* Stores daily actuals and forecasts for each pool, used as input for optimization. `UNIQUE(pool_id, date)`.

*   **`gas_fees_daily_summary`**
    *   `date` (PK, DATE, NOT NULL)
    *   `actual_avg_gas_price_gwei` (NUMERIC)
    *   `forecasted_avg_gas_price_gwei` (NUMERIC)
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `updated_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   *Purpose:* Stores daily actual and forecasted gas prices.

*   **`icebox_tokens`**
    *   `token_symbol` (PK, TEXT, NOT NULL)
    *   `added_at` (TIMESTAMP, NOT NULL)
    *   `removed_at` (TIMESTAMP, NULLABLE)
    *   `reason` (TEXT, NOT NULL)
    *   *Purpose:* Stores the current Icebox token list, with timestamps for when tokens are added/removed and the reason for exclusion.

*   **`approved_tokens`**
    *   `token_symbol` (PK, TEXT, NOT NULL)
    *   `added_at` (TIMESTAMP, NOT NULL)
    *   `removed_at` (TIMESTAMP, NULLABLE)
    *   *Purpose:* Tracks the list of approved tokens for each run, with add/remove timestamps.

*   **`blacklisted_tokens`**
    *   `token_symbol` (PK, TEXT, NOT NULL)
    *   `added_at` (TIMESTAMP, NOT NULL)
    *   `removed_at` (TIMESTAMP, NULLABLE)
    *   *Purpose:* Tracks the list of blacklisted tokens for each run, with add/remove timestamps.

*   **`approved_protocols`**
    *   `protocol_name` (PK, TEXT, NOT NULL)
    *   `added_at` (TIMESTAMP, NOT NULL)
    *   `removed_at` (TIMESTAMP, NULLABLE)
    *   *Purpose:* Tracks the list of approved protocols for each run, with add/remove timestamps.

*   **`pool_groups`**
    *   `pool_id` (FK, TEXT, NOT NULL)
    *   `group` (INTEGER, NOT NULL)  // 1, 2, or 3
    *   `date` (DATE, NOT NULL)
    *   `criteria_snapshot` (JSONB, NOT NULL) // stores the APY/stddev values used for assignment
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `UNIQUE(pool_id, date)`
    *   *Purpose:* Stores pool grouping (Group 1, 2, 3) and their assignment per day.

*   **`daily_ledger`**
    *   `date` (PK, DATE, NOT NULL)
    *   `token_symbol` (PK, TEXT, NOT NULL)
    *   `start_balance` (NUMERIC, NOT NULL)
    *   `end_balance` (NUMERIC, NOT NULL)
    *   `nav_usd` (NUMERIC, NOT NULL)
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   *Purpose:* Tracks daily balances and NAV for each token.

**3. Application Data Tables (Output/Decision Layer):**
These tables store the results of the optimization process and the parameters used to generate them, enabling auditing and performance tracking.

*   **`optimization_runs`**
    *   `run_id` (PK, UUID)
    *   `run_timestamp` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   `total_aum_usd` (NUMERIC, NOT NULL)
    *   `total_expected_profit_usd` (NUMERIC): The maximized objective function value.
    *   `total_gas_cost_usd` (NUMERIC)
    *   `total_conversion_cost_usd` (NUMERIC)
    *   `tvl_limit_percentage_param` (NUMERIC)
    *   `max_alloc_percentage_param` (NUMERIC)
    *   `conversion_rate_param` (NUMERIC)
    *   `min_pools_param` (INTEGER)
    *   `profit_optimization_flag` (BOOLEAN)
    *   `token_marketcap_limit_param` (NUMERIC)
    *   `pool_tvl_limit_param` (NUMERIC)
    *   `pool_apy_limit_param` (NUMERIC)
    *   `pool_pair_tvl_ratio_min_param` (NUMERIC)
    *   `pool_pair_tvl_ratio_max_param` (NUMERIC)
    *   `group1_max_pct_param` (NUMERIC)
    *   `group2_max_pct_param` (NUMERIC)
    *   `group3_max_pct_param` (NUMERIC)
    *   `position_max_pct_total_assets_param` (NUMERIC)
    *   `position_max_pct_pool_tvl_param` (NUMERIC)
    *   `group1_apy_delta_max_param` (NUMERIC)
    *   `group1_7d_stddev_max_param` (NUMERIC)
    *   `group1_30d_stddev_max_param` (NUMERIC)
    *   `group2_apy_delta_max_param` (NUMERIC)
    *   `group2_7d_stddev_max_param` (NUMERIC)
    *   `group2_30d_stddev_max_param` (NUMERIC)
    *   `group3_apy_delta_min_param` (NUMERIC)
    *   `group3_7d_stddev_min_param` (NUMERIC)
    *   `group3_30d_stddev_min_param` (NUMERIC)
    *   `approved_tokens_snapshot_json` (JSONB): Snapshot of approved tokens list.
    *   `blacklisted_tokens_snapshot_json` (JSONB): Snapshot of blacklisted tokens list.
    *   `approved_protocols_snapshot_json` (JSONB): Snapshot of approved protocols list.
    *   `icebox_tokens_snapshot_json` (JSONB): Snapshot of icebox tokens list.
    *   `input_current_allocations_json` (JSONB): Snapshot of `current_allocations` input.
    *   `input_liquidity_portfolio_json` (JSONB): Snapshot of `liquidity_portfolio` input.
    *   `other_dynamic_limits_json` (JSONB): Any other dynamic limits or configuration parameters set for the run.
    *   `status` (TEXT, NOT NULL): e.g., 'SUCCESS', 'FAILED', 'PARTIAL_SUCCESS'.
    *   `error_message` (TEXT): If the run failed.
    *   *Purpose:* Records each optimization run with its parameters and overall results.

*   **`daily_allocations`**
    *   `allocation_id` (PK, UUID)
    *   `run_id` (FK, UUID, NOT NULL): References `optimization_runs.run_id`.
    *   `pool_id` (FK, TEXT, NOT NULL): References `defi_pools.pool_id`.
    *   `allocated_amount_usd` (NUMERIC, NOT NULL)
    *   `allocated_weight` (NUMERIC, NOT NULL)
    *   `expected_daily_apy_at_allocation` (NUMERIC): The forecasted APY used for this specific allocation.
    *   `expected_tvl_at_allocation` (NUMERIC): The forecasted TVL used for this specific allocation.
    *   `gas_fee_usd_at_allocation` (NUMERIC): The gas fee used for this specific allocation.
    *   `conversion_penalty_at_allocation` (NUMERIC): The conversion penalty used for this specific allocation.
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   *Purpose:* Stores the detailed breakdown of each daily asset allocation, linking back to the specific optimization run. `UNIQUE(run_id, pool_id)`.

*   **`model_performance_metrics`**
    *   `metric_id` (PK, UUID)
    *   `model_name` (TEXT, NOT NULL): e.g., 'pool_apy_forecast', 'gas_fee_forecast'
    *   `metric_date` (DATE, NOT NULL)
    *   `metric_type` (TEXT, NOT NULL): e.g., 'MAE', 'RMSE', 'R2'
    *   `metric_value` (NUMERIC, NOT NULL)
    *   `training_data_start_date` (DATE)
    *   `training_data_end_date` (DATE)
    *   `created_at` (TIMESTAMP, NOT NULL, DEFAULT NOW())
    *   *Purpose:* Stores performance metrics for retrained models, enabling tracking of forecasting accuracy over time.

This structure provides a robust foundation for the pipeline, allowing for clear data stages, historical tracking, and detailed auditing of the daily optimization decisions.