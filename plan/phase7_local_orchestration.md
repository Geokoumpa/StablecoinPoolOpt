# Phase 7: Local Orchestration for Testing - Detailed Plan

This phase focuses on implementing a local orchestration mechanism, `main_pipeline.py`, to sequentially execute the various data pipeline scripts. This will allow for comprehensive local testing and debugging of the entire DefiYieldOpt application's data flow before deployment to GCP Workflows.

## 7.1. `main_pipeline.py` Implementation

The `main_pipeline.py` script will serve as the central entry point for local execution. It will import and call functions from the individual scripts in the correct order, mimicking the dependencies and flow defined in the overall development plan.

### 7.1.1. Script Structure and Execution Flow

The `main_pipeline.py` will follow this general structure:

1.  **Configuration Loading**: Load necessary configurations (e.g., database connection strings, API keys, logging settings) from `config.py` or environment variables.
2.  **Logging Setup**: Initialize a robust logging system to capture execution progress, warnings, and errors.
3.  **Sequential Phase Execution**:
    *   **Phase 1: Data Ingestion**:
        *   Call `fetch_defillama_pools.py` to fetch pool data.
        *   Call `fetch_defillama_pool_history.py` to fetch historical pool data for each pool.
        *   Call `fetch_ohlcv_coinmarketcap.py` for Crypto OHLCV data.
        *   Call `fetch_gas_ethgastracker.py` for gas fees.
        *   Call `fetch_account_data_etherscan.py` for account balances and transaction history.
    *   **Phase 3: Data Processing & Transformation**:
        *   Call `process_icebox_logic.py` to apply Icebox rules.
        *   Call `calculate_pool_metrics.py` for 7-day and 30-day rolling APY and standard deviations.
        *   Call `apply_pool_grouping.py` to assign pools to groups.
        *   Call `filter_pools.py` to apply all pool filtering criteria.
    *   **Phase 4: Forecasting**:
        *   Call `forecast_pools.py` for APY and TVL forecasts.
        *   Call `forecast_gas_fees.py` for gas fee actuals and forecasts.
    *   **Phase 5: Asset Allocation**:
        *   Call `optimize_allocations.py` for core asset allocation optimization.
        *   Call `optimize_allocations_milp.py` for MILP approach.
    *   **Phase 6: Reporting & Notification**:
        *   Call `manage_ledger.py` to update daily token balances and NAV.
        *   Call `post_slack_notification.py` for daily allocation recommendations.
4.  **Error Handling**: Implement `try-except` blocks around each script call to catch and log exceptions, preventing the entire pipeline from crashing due to a single script failure.
5.  **Reporting/Summary**: At the end of the execution, provide a summary of the pipeline run, including any errors or warnings.

### 7.1.2. Key Considerations

*   **Modularity**: Ensure that individual scripts are designed to be callable as functions, allowing `main_pipeline.py` to import and execute them directly.
*   **Dependency Management**: Explicitly manage dependencies between scripts within `main_pipeline.py` to ensure correct execution order.
*   **Idempotency**: Where applicable, design scripts to be idempotent, meaning they can be run multiple times without causing unintended side effects.
*   **Parameterization**: Allow `main_pipeline.py` to accept command-line arguments or environment variables to control which phases to run or to pass specific parameters to individual scripts (e.g., date ranges for historical data).

## 7.2. Local Orchestration Pipeline Diagram

Refer to the [Local Orchestration Pipeline Diagram](local_orchestration_diagram.md) for a visual representation of the execution flow.