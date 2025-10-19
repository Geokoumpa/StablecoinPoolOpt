# Product Brief

The DefiYieldOpt application is designed to help users maximize their returns in Decentralized Finance (DeFi) by intelligently optimizing yield strategies. It achieves this by automatically collecting and analyzing vast amounts of data on various DeFi pools and associated transaction costs. Based on this continuous analysis, the system generates daily asset allocation recommendations, enabling a daily reallocation of Assets Under Management (AUM) to capitalize on the most profitable opportunities.

# Data Sources

## Stablecoin Pools
Defillama API

### Pool Names
Endpoint: https://yields.llama.fi/pools
Docs: https://api-docs.defillama.com/#tag/yields/get/pools

### Pool History
Endpoint: https://yields.llama.fi/chart/{pool_id}
Docs: https://api-docs.defillama.com/#tag/yields/get/chart/{pool}

## Crypto OHLCV
CoinMarketCap API is used to fetch OHLCV data for major cryptocurrencies (BTC, ETH) and stablecoins.
Docs: https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyOhlcvHistorical

## Gas Fees
EthGasTracker.com
Docs: https://www.ethgastracker.com/docs

## Account Balances & Transactions
Etherscan Accounts API is used to fetch real-time and historical balance and transaction data for the main asset holding address. This serves as the ground truth for actual investment performance and provides a full historical record for auditing.
Docs: https://docs.etherscan.io/etherscan-v2/api-endpoints/accounts

### Icebox Tokens
CoinMarketCap API is used to fetch OHLCV data for all stablecoins. This data serves as the source for Icebox logic, which determines tokens to be temporarily excluded from allocation based on configurable market signal thresholds.


# Data Pipeline (Optimized Execution Sequence)

- No intra-day handling for extreme depegs; manual override only for extreme events.
- If data is missing, retry fetch up to N times before defaulting to last known value; delay allocation run rather than fail.
- Non-USD stablecoins are ignored for now.
- CoinMarketCap is the single source for price feeds; fallback deferred.

The pipeline follows an optimized execution sequence to maximize efficiency and data freshness:

## Phase 1: Initial Data Ingestion
1) **Apply database migrations** - Ensure schema is up-to-date
2) **Fetch Pools** from Defillama API and update `pools` table
3) **Fetch Crypto OHLCV** data for major cryptocurrencies (BTC, ETH) and *approved* stablecoins from CoinMarketCap (based on user-defined stablecoin whitelist/blacklist)
4) **Fetch Gas Fees** from EthGasTracker.com (using `/gas/average` for historical/daily averages and `/gas/latest` for real-time data). Hourly data is stored and daily averages for forecasting are computed from the hourly records
5) **Fetch Account Balances** and Transaction History from Etherscan Accounts API for the main asset holding address

## Phase 2: Pre-Filtering & Pool History Ingestion
6) **Pre-filter pools** based on basic criteria (protocols, tokens, blacklists) - *excludes icebox logic at this stage*:
   - Exclude pools not belonging to approved protocols (from `approved_protocols` table)
   - Exclude pools not containing approved tokens (from `approved_tokens` table)
   - Exclude pools containing blacklisted tokens (from `blacklisted_tokens` table)
   - Apply basic trade logic filters: exclude tokens with MarketCap < `token_marketcap_limit` (default: $35M), pools with TVL < `pool_tvl_limit` (default: $500K), pools with APY < `pool_apy_limit` (default: 6%)

7) **Fetch full history** for pools passing the pre-filtering step

## Phase 3: Pool Analysis & Final Filtering
8) **Calculate pool metrics** - 7-day and 30-day Rolling APY, standard deviations, and volatility metrics for pools

9) **Apply Pool Grouping Logic**:
   - Calculate ΔAPY (1 day) = |Today_Yield - Yesterday_Yield|, Δ7DSTDDEV = absolute stddev of daily APYs over last 7 days, Δ30DSTDDEV = absolute stddev of daily APYs over last 30 days
   - Assign pools to Group 1, Group 2, or Group 3 based on universal thresholds, with boundary values assigned to the lower group:
     - **Group 1:** ΔAPY ≤ 1% AND Δ7DSTDDEV ≤ 1.5% AND Δ30DSTDDEV ≤ 2%
     - **Group 2:** ΔAPY ≤ 3% AND Δ7DSTDDEV ≤ 4% AND Δ30DSTDDEV ≤ 5%
     - **Group 3:** ΔAPY > 3% AND Δ7DSTDDEV > 4% AND Δ30DSTDDEV > 2%
   - Store pool grouping assignments in `pool_daily_metrics` table with group assignments

10) **Apply Icebox logic**:
    - Only tokens from the approved token set are eligible for Icebox
    - Analyze daily OHLCV for each approved stablecoin based on configurable thresholds (e.g., `icebox_ohlc_l_threshold_pct` (default: 2%), `icebox_ohlc_l_days_threshold` (default: 2), `icebox_ohlc_c_threshold_pct` (default: 1%), `icebox_ohlc_c_days_threshold` (default: 1), `icebox_recovery_l_days_threshold` (default: 2), `icebox_recovery_c_days_threshold` (default: 3))
    - Add tokens to Icebox if thresholds are breached; Icebox list is a subset of approved tokens temporarily excluded from allocation
    - Remove tokens from Icebox if recovery criteria met

11) **Final pool filtering** - Apply icebox exclusions and remaining trade logic filters:
    - Exclude pools containing any Icebox tokens
    - Apply remaining pair TVL ratio constraints: pools where side of pair is not within `pool_pair_tvl_ratio_min` (default: 30%) - `pool_pair_tvl_ratio_max` (default: 50%) of other side's TVL

## Phase 4: Fresh Data & Snapshots
12) **Fetch fresh OHLCV** data from CoinMarketCap to ensure latest price data for allocation decisions

13) **Create allocation snapshots** - Create snapshots of all dynamic configuration lists (approved tokens, blacklisted tokens, approved protocols, icebox tokens) for the optimization run

## Phase 5: Forecasting
14) **Forecast selected pools** APY and TVL for current day using filtered pool set

15) **Forecast gas fees** - Calculate actuals for previous day's gas fees and forecast gas fees for current day

## Phase 6: Asset Allocation
16) **Optimize asset allocation**:
    - Run optimization for total assets, considering all constraints and objective function
    - If forecasted APY in optimization output + fees > forecasted APY in existing position, reallocate
    - If not, run optimization only for the yield generated yesterday
    - Force reallocation if any token, pool, or protocol in current allocation is no longer approved or available
    - Store optimization parameters in `allocation_parameters` table (including snapshots of dynamic lists)
    - Store allocation results in `asset_allocations` table

## Phase 7: Reporting & Notification
17) **Retain daily ledger**:
    - Retain a daily ledger with all the number of tokens per stablecoin at the beginning of day (before optimization run for today) and at the end of the day (before optimization run for tomorrow)
    - Calculate daily NAV = Balance of token N * C price from OHLC every day before running optimization for today
    - Calculate realized Yield for yesterday ((Total Assets before optimization run for tomorrow) - (Total Assets before optimization run for today))/(Total Assets before optimization run for today)
    - Calculate realized Yield YTD ((((Total Assets before optimization run for tomorrow) - (Total Assets before optimization DAY 0))/(Total Assets before optimization DAY 0))
    - Record daily token balances and NAV in `daily_ledger` table

18) **Post to slack channel** - Send daily allocation recommendations and performance summary


# Data Storage Strategy

## Database Tables

* **`raw_defillama_pools`**: Stores raw JSON responses from the DeFiLlama pools endpoint.
* **`raw_defillama_pool_history`**: Stores raw JSON responses for individual pool histories.
* **`raw_ethgastracker_hourly_gas_data`**: Stores raw hourly average gas data from EthGasTracker.com.
* **`raw_etherscan_account_transactions`**: Stores raw transaction history from Etherscan Accounts API.
* **`raw_etherscan_account_balances`**: Stores raw historical balance data from Etherscan Accounts API.

* **`pools`**: Master table for all identified DeFi pools, with core metadata.
* **`pool_daily_metrics`**: Stores daily historical and forecasted metrics for pools (APY, TVL). This will be the "enriched" data.
* **`gas_fees_hourly`**: Stores processed gas fee data at a higher granularity (e.g., hourly averages from 5-min data). Daily averages for forecasting are computed from this table.
* **`gas_fees_daily`**: Stores daily actual and forecasted gas fees, derived from hourly data.
* **`icebox_tokens`**: Stores the current Icebox token list, with timestamps for when tokens are added/removed and the reason for exclusion.
* **`approved_tokens`**: Tracks the list of approved tokens for each run, with add/remove timestamps.
* **`blacklisted_tokens`**: Tracks the list of blacklisted tokens for each run, with add/remove timestamps.
* **`approved_protocols`**: Tracks the list of approved protocols for each run, with add/remove timestamps.
* **`pool_groups`**: Stores pool grouping (Group 1, 2, 3) and their assignment per day, including criteria snapshot.
* **`daily_ledger`**: Tracks daily balances and NAV for each token.
* **`asset_allocations`**: Stores the results of the daily optimization, including allocated amounts per pool.
* **`allocation_parameters`**: Stores the configurable parameters used for each optimization run, for reproducibility and auditing.

# Allocation Parameters

The following parameters are stored in the `allocation_parameters` table for each optimization run:

- `tvl_limit_percentage`: Maximum percentage of pool TVL that can be allocated (default 0.05)
- `max_alloc_percentage`: Maximum allocation to any single pool (default 0.25, only when profit_optimization is False)
- `conversion_rate`: Token conversion fee rate (default 0.0004)
- `min_pools`: Minimum number of pools for diversification (default 4, or 1 if profit_optimization is True)
- `profit_optimization`: Boolean flag to enable profit-focused allocation
- `approved_tokens`: List of approved tokens for the run (snapshot)
- `blacklisted_tokens`: List of blacklisted tokens for the run (snapshot)
- `approved_protocols`: List of approved protocols for the run (snapshot)
- `icebox_tokens`: List of tokens excluded due to Icebox logic (snapshot)
- `token_marketcap_limit`: Minimum market cap for token inclusion (default $35M)
- `pool_tvl_limit`: Minimum TVL for pool inclusion (default $500K)
- `pool_apy_limit`: Minimum APY for pool inclusion (default 6%)
- `pool_pair_tvl_ratio_min`: Minimum ratio for pool pair TVL (default 30%)
- `pool_pair_tvl_ratio_max`: Maximum ratio for pool pair TVL (default 50%)
- `group1_max_pct`: Maximum allocation to Group 1 pools (default 35%)
- `group2_max_pct`: Maximum allocation to Group 2 pools (default 35%)
- `group3_max_pct`: Maximum allocation to Group 3 pools (default 30%)
- `position_max_pct_total_assets`: Maximum position size as % of total assets (default 25%)
- `position_max_pct_pool_tvl`: Maximum position size as % of pool TVL (default 5%)
- `group1_apy_delta_max` (default: 1%): Maximum allowed APY delta for Group 1
- `group1_7d_stddev_max` (default: 1.5%): Maximum allowed 7-day APY stddev delta for Group 1
- `group1_30d_stddev_max` (default: 2%): Maximum allowed 30-day APY stddev delta for Group 1
- `group2_apy_delta_max` (default: 3%): Maximum allowed APY delta for Group 2
- `group2_7d_stddev_max` (default: 4%): Maximum allowed 7-day APY stddev delta for Group 2
- `group2_30d_stddev_max` (default: 5%): Maximum allowed 30-day APY stddev delta for Group 2
- `group3_apy_delta_min` (default: >3%): Minimum APY delta for Group 3
- `group3_7d_stddev_min` (default: >4%): Minimum 7-day APY stddev delta for Group 3
- `group3_30d_stddev_min` (default: >2%): Minimum 30-day APY stddev delta for Group 3
- `other_dynamic_limits`: Any other dynamic limits or configuration parameters set for the run

# Snapshotting Functionality

For each optimization run, the system creates a snapshot of all dynamic configuration lists (approved tokens, blacklisted tokens, approved protocols, icebox tokens) and stores references or serialized copies in the `allocation_parameters` table. This ensures that every allocation run is fully reproducible and auditable, even if the underlying lists change in future runs. The snapshot may be implemented as foreign keys to versioned tables or as JSON fields containing the exact state of each list at the time of the run.

# Pool Filtering

Pools are selected based on approved protocols and tokens.
- Only pools belonging to approved protocols (AAVE v2/3, Morpho, Sky Lending, SpartkLend, Curve, Uniswap V2/3, Compound V2/3, Goldfinch, Yearn, Balancer V2/3, Liquity, Pendle, Convex Finance, Maple Finance, Fluid Lending) on Ethereum are considered.
- Only pools containing approved tokens (USDT, USDC, USDS, DAI, LUSD, BOLD, PYUSD, USDP, GUSD, GHO, RLUSD, USDtb, USD0, USDG, USDL, USDM, USDe, USYC, BUIDL, FOBXX, USDY, USTB, USCC, OUSG, TBILL, USTBL, crvUSD, USDF) are considered.
- Pools containing any blacklisted tokens (FRAX, USDD, MIM, FRXUSD, SUSD, USD0++) are excluded.
- Pools containing any Icebox tokens are excluded for the current run.
- Approved/blacklisted tokens and protocols can be updated before each run.

# Pool Grouping Logic

For each pool, the following metrics are calculated daily:
- 7-day rolling APY
- 30-day rolling APY
- Today - Yesterday APY delta
- 7-day APY standard deviation delta
- 30-day APY standard deviation delta

Pools are assigned to groups based on configurable thresholds:
- **Group 1:** ΔAPY ≤ 1% AND Δ7DSTDDEV ≤ 1.5% AND Δ30DSTDDEV ≤ 2%
- **Group 2:** ΔAPY ≤ 3% AND Δ7DSTDDEV ≤ 4% AND Δ30DSTDDEV ≤ 5%
- **Group 3:** ΔAPY > 3% AND Δ7DSTDDEV > 4% AND Δ30DSTDDEV > 2%

Thresholds for group assignment can be changed before every run and are stored in the `allocation_parameters` table for reproducibility.


# Pool APY, TVL and Gas Fee Forecasting

## Data Preprocessing & Feature Engineering

* Date Conversion: Convert raw date strings into proper datetime objects to enable time-based operations and indexing
* Unit Conversion: Transform gas price values from Wei to Gwei for easier interpretation and calculation
* Derived Financial Metrics: Calculate estimated gas fees in USD and ETH by multiplying the gas price (in Gwei) by a fixed gas limit and the corresponding cryptocurrency's price (using ETH High price for the day for conservative USD cost estimate). Future enhancement: switch to real-time node-based fee capture.
* Lagged Features: Create new features by shifting existing time series data (e.g., Ethereum closing price) by a specific number of periods (e.g., 2 days). These "lagged" values capture the influence of past observations on current or future values
* Time-Based Features: Extract cyclical components from the date, such as the day of the week. These features can help the model identify recurring patterns or seasonality
* Exogenous Variable Identification: Define which columns, besides the target variable, will be used as external predictors (exogenous variables) in the forecasting model
* Exogenous Variable Shifting (to prevent leakage): Shift the exogenous variables by one period (e.g., one day) relative to the target variable. This crucial step ensures that when the model makes a prediction for a given day, it only uses exogenous information that would have been available before that prediction point, preventing data leakage from the future
* Missing Value Handling: Remove any rows that contain missing values (NaN) after all preprocessing and feature engineering steps, ensuring a clean and complete dataset for model training

## Model Training & Hyperparameter Tuning

Rolling forecast with daily refitting using Sliding Window strategy:

1) For each day in the forecasting period, the forecaster (which wraps the XGBRegressor) is fit again using all available data up to that current_date.
2) It then predicts for the next_date.

Tools used:
* skforecast.recursive.ForecasterRecursive: To define the recursive forecasting model, which uses past predictions as inputs for future predictions.
* xgboost.XGBRegressor: The chosen regressor for its efficiency and performance in tabular data. It will be the core predictive model within ForecasterRecursive.
* skforecast.model_selection.bayesian_search_forecaster: For performing Bayesian optimization to find the optimal hyperparameters for the XGBRegressor and the ForecasterRecursive (e.g., n_estimators, max_depth, lags).
* skforecast.model_selection.TimeSeriesFold: To ensure proper time series cross-validation during hyperparameter tuning, preventing data leakage and providing realistic performance estimates.

## Retraining Strategy

The forecasting models for pool APYs, TVLs, and gas fees will be retrained daily. This ensures the models continuously learn from the latest market conditions and actual performance data.

- **Frequency:** Daily.
- **Data Source:** Retraining will utilize the actuals recorded in `pool_daily_metrics` (for APY and TVL) and `gas_fees_daily` (for gas fees). This ensures the models are updated with the most recent ground truth data.
- **Methodology:** The rolling forecast with daily refitting (Sliding Window strategy) described above will be applied, where the forecaster is refit using all available historical actuals up to the current day.

## Forecasting

Tools used:

* skforecast.ForecasterRecursive.predict(): To generate single-step or multi-step forecasts.
* datetime and timedelta: For managing the time window in rolling forecasts.

## Model Persistence & Data Storage

Tools used:
* pickle: For serializing and de-serializing the trained skforecast forecaster object, allowing it to be saved to disk and reloaded without retraining.


# Asset Allocation

## Optimization & Reallocation Logic

- Run optimization for total assets.
- Forecasted daily yield is calculated as (AUM * (Forecasted_APY% / 365)). Compare net forecasted yield (including fees) vs. net yield from existing allocation; if greater, reallocate assets according to optimization output and send step-by-step actions to Slack.
- If not, run optimization only for the yield generated yesterday and send step-by-step actions to Slack.
- Regardless of APY comparison, force reallocation if any token, pool, or protocol in the current allocation is no longer approved or has become unavailable (e.g., moved to Icebox, blacklisted, or protocol removed).

## Constraints

* Sum of Weights Equals 1 (Full Allocation): Ensures the entire capital is distributed.
* Allocation Amounts Less Than or Equal to Per-Pool TVL Limit: Prevents over-allocation to individual pools based on their Total Value Locked (TVL) forecast, controlled by `tvl_limit_percentage` (default 0.05).
* Non-Negative Weights: Ensures only positive allocations are made.
* Binary Variable Linkage (Pool Selection): Links allocation weights to binary variables to determine which pools are selected.
* Minimum Number of Pools: Guarantees a minimum number of pools are selected for diversification, controlled by `min_pools` (default 4, or 1 if `profit_optimization` is True).
* Maximum Allocation Percentage (General Optimization Only): Limits the maximum percentage of capital allocated to any single pool, applied only in general optimization scenarios (not profit optimization), controlled by `max_alloc_percentage` (default 0.25).
* Position Limits:
    * No single position will be more than `position_max_pct_total_assets` (default 25%) of Total Assets.
    * No single position will be more than `position_max_pct_pool_tvl` (default 5%) of the targeted Pool’s TVL including our contribution.
* Group Allocation Limits:
    * No more than `group1_max_pct` (default 35%) in Group 1 pools.
    * No more than `group2_max_pct` (default 35%) in Group 2 pools.
    * No more than `group3_max_pct` (default 30%) in Group 3 pools.

## Objective Function

Maximize (daily_yield * total_usd - total_gas_fees * total_usd - total_conversion_penalty * total_usd)
241.1 | - Rolling APY = daily annualized APY from historical daily yield % values.
241.2 | - Pool Pair TVL Ratio constraint applies to each token share of TVL for >2-token pools.
241.3 | - Forked protocol risk is ignored as unapproved.
241.4 | - APY source manipulation is mitigated by trusting DeFiLlama data curation.
241.5 | - Gas fee units: Gas in Gwei → USD via ETH/USD; all APY calculations normalized to USD; objective function operates on weighted APY% in USD terms.

### Objective Breakdown

1) daily_yield * total_usd
    * daily_yield: This is calculated as cp.sum(cp.multiply(daily_apys, weights)). It represents the sum of the expected daily APY (Annual Percentage Yield) for each pool, weighted by the proportion of the total capital allocated to that pool.
    * total_usd: This is the total amount of capital being managed (sum of current_allocations and liquidity_portfolio).
    * Purpose: This term represents the total expected gross profit generated from the allocated assets, based on the forecasted daily APYs. The optimizer aims to maximize this component.
    
2) total_gas_fees * total_usd:

    * total_gas_fees: This is calculated as cp.sum(cp.multiply(gas_fees, z)). It represents the sum of estimated gas fees associated with depositing into the selected pools. The gas_fees are normalized by total_usd, and z is a binary variable indicating if a pool is selected.
    * Purpose: This term represents the total cost incurred due to gas fees for making new deposits. The optimizer aims to minimize this cost, as it is subtracted from the gross profit.

3) total_conversion_penalty * total_usd:

    * total_conversion_penalty: This is calculated as cp.sum(cp.multiply(weights, conversion_fee_vector)). The conversion_fee_vector accounts for potential fees if the required tokens for a pool are not readily available in the liquidity_portfolio and need to be converted (e.g., from USDC).
    * Purpose: This term represents the total cost incurred due to token conversion fees. The optimizer aims to minimize this cost, as it is also subtracted from the gross profit.


## Input

* A dataset containing pool_id, next_day_tvl_forecast, rolling_apy_forecast_for_next_day, and max_gas_fee_usd
* A dictionary mapping pool_id to associated tokens (e.g., {'pool_123': 'USDC-DAI'})
* current_allocations: A dictionary where keys are pool_ids and values are the amounts of capital (in USD) that are currently deployed in specific DeFi pools. It represents the funds that are already actively earning yield
* liquidity_portfolio: A dictionary where keys are token symbols (e.g., 'USDC', 'DAI') and values are the amounts of those tokens (in USD equivalent) that are liquid and available for new allocations or to cover transaction costs. It represents the funds that are not currently deployed in any yield-generating pool but are ready to be used

## Configurable Parameters

* tvl_limit_percentage (default 0.05)
* max_alloc_percentage (default 0.25, used only when profit_optimization is False)
* conversion_rate (default 0.0004)
* min_pools (default 4, or 1 if profit_optimization is True)
* profit_optimization (boolean flag)

## How the profit_optimization Flag Works

The profit_optimization is a boolean flag that significantly alters the constraints of the optimization problem, shifting its focus:

* When profit_optimization is True:

    * Objective: The primary goal becomes to maximize profit, even if it means concentrating capital in a very small number of highly profitable pools.
    * Constraint Changes:
        * Minimum Pools (min_pools): The min_pools constraint is relaxed from its default value (4) to 1. This allows the optimizer to select as few as one pool if that single pool offers the highest expected net profit.
        * Maximum Allocation Percentage (max_alloc_percentage): The constraint weights <= max_alloc_percentage is not applied. This means there is no upper limit on the proportion of the total capital that can be allocated to a single pool, enabling full concentration if it's deemed most profitable.
    * Use Case: This mode is typically used for scenarios where the system is trying to decide if any profit can be made by reinvesting realized profits, or when a very aggressive, high-risk/high-reward strategy is desired.

* When profit_optimization is False (Default Behavior):

    * Objective: The goal is to maximize profit while also ensuring a degree of diversification and risk management.
    * Constraint Behavior:
        * Minimum Pools (min_pools): The min_pools constraint defaults to 4, requiring the allocation to be spread across at least four pools.
        * Maximum Allocation Percentage (max_alloc_percentage): The weights <= max_alloc_percentage constraint (default 0.25 or 25%) is applied, limiting the maximum proportion of capital that can be allocated to any single pool.
    * Use Case: This is the standard operational mode for daily asset allocation, aiming for a balanced approach between maximizing returns and managing portfolio risk through diversification.


# Data Pipeline Architecture

## Cloud Infrastructure

* GCP Cloud Scheduler to have the pipeline run at 00:01 EST
* GCP Cloud run jobs
* GCP Workflows
* GCP Cloud SQL with Postgresql

## Modular Pipeline Scripts

To ensure a well-architected and maintainable workflow, the data pipeline functionality will be split into several distinct Python scripts, each responsible for a specific set of tasks. This modular approach facilitates independent development, testing, and deployment of pipeline components.

### 1. Data Ingestion Scripts
- `fetch_defillama_pools.py`: Fetches raw pool data from Defillama API and stores it in `raw_defillama_pools` and updates `pools` metadata.
- `fetch_ohlcv_coinmarketcap.py`: Fetches Crypto OHLCV data for major cryptocurrencies (BTC, ETH) and stablecoins from CoinMarketCap.
- `fetch_gas_ethgastracker.py`: Fetches Gas Fees from EthGasTracker.com (hourly averages and latest).
- `fetch_account_data_etherscan.py`: Fetches Account Balances and Transaction History from Etherscan Accounts API.

### 2. Data Processing & Transformation Scripts
- `process_icebox_logic.py`: Analyzes OHLCV data, applies Icebox rules, and updates the `icebox_tokens` table.
- `calculate_pool_metrics.py`: Computes 7-day and 30-day rolling APY and their standard deviations for pools.
- `apply_pool_grouping.py`: Assigns pools to Group 1, 2, or 3 based on calculated metrics and configurable thresholds, storing results in `pool_groups`.
- `filter_pools.py`: Applies all pool filtering criteria (approved/blacklisted tokens/protocols, Icebox status, market cap, TVL, APY, pair TVL ratio), marking pools as `is_filtered_out` in `pool_daily_metrics`.

### 3. Forecasting Scripts
- `forecast_pools.py`: Generates APY and TVL forecasts for selected pools, updating `pool_daily_metrics`.
- `forecast_gas_fees.py`: Calculates actuals and forecasts gas fees, updating `gas_fees_daily_summary`.

### 4. Optimization & Allocation Scripts
- `optimize_allocations.py`: Executes the core asset allocation optimization logic, considering all constraints and objectives. Stores run parameters in `optimization_runs` and detailed allocations in `daily_allocations`.
- `manage_ledger.py`: Updates the daily ledger with token balances and NAV.

### 5. Reporting & Notification Scripts
- `post_slack_notification.py`: Formats and sends daily allocation recommendations and status updates to Slack.

### Orchestration
A central orchestration mechanism (e.g., GCP Workflows, or a `main_pipeline.py` script) will coordinate the execution of these individual scripts in the defined sequence, ensuring data dependencies are met.

## Tech Stack

* Scripting language: Python
* Optimization library: cvxpy
* Forecasting: xgboost, skforecast, pickle
* Infrastructure as code: Terraform
* Containerization: Docker

# Backtesting Functionality

A dedicated backtesting framework will be developed to evaluate the performance of the asset allocation strategy against historical data. This functionality is crucial for validating model effectiveness, optimizing parameters, and understanding potential returns and risks under various market conditions.

## Key Aspects

*   **Historical Data Utilization:** The backtesting engine will leverage the comprehensive historical data stored in the database, including:
    *   `pool_daily_metrics`: Historical actuals for APY and TVL.
    *   `gas_fees_daily_summary`: Historical actuals for gas fees.
    *   `raw_etherscan_account_transactions` and `raw_etherscan_account_balances`: For simulating historical portfolio state and transaction costs.
    *   Historical snapshots of `approved_tokens`, `blacklisted_tokens`, `approved_protocols`, and `icebox_tokens` (from `optimization_runs` table or dedicated historical tables if implemented).
*   **Simulation Engine:** A simulation engine will re-run the entire data pipeline and optimization logic for historical periods, applying the configured parameters and rules as they would have existed on each historical day.
*   **Performance Metrics:** The backtesting results will include key performance indicators such as:
    *   Simulated daily and cumulative returns.
    *   Drawdowns and volatility.
    *   Sharpe ratio and other risk-adjusted returns.
    *   Transaction costs incurred during rebalancing.
*   **Parameter Optimization:** The backtesting framework can be used to systematically test different sets of allocation parameters to identify optimal configurations that maximize returns or minimize risk over historical periods.
*   **Reporting:** Detailed reports and visualizations will be generated to present backtesting results, allowing for in-depth analysis of strategy performance.

# Admin Web Application

A web-based administrative application will be developed to provide a user interface for managing configuration parameters and visualizing ledger information.

## Key Features

*   **Configuration Management:**
    *   Ability to view and modify all configurable parameters stored in the `allocation_parameters` table.
    *   Interface for adding/removing approved tokens, blacklisted tokens, and approved protocols.
    *   Management of Icebox token thresholds and manual override capabilities.
*   **Ledger Visualization:**
    *   Interactive dashboards and charts to display daily ledger information from the `daily_ledger` table.
    *   Tracking of token balances, daily NAV, realized yield (daily and YTD).
    *   Historical views and trend analysis of financial metrics.
*   **Account Performance & Audit:**
    *   Display actual account balances and transaction history fetched from Etherscan.
    *   Provide tools for auditing past transactions and verifying investment performance against ledger records.
*   **Audit Trail:**
    *   View historical optimization runs and the specific parameters (snapshots) used for each run.

## Technology Stack

*   **Frontend Framework:** Next.js
*   **Authentication:** Clerk (for secure user authentication and authorization)
*   **Database Interaction:** Direct connection to the same PostgreSQL database used by the backend pipeline.
*   **Deployment:** GCP Cloud Run for web apps.
