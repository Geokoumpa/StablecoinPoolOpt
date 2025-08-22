# Local Orchestration Pipeline Diagram

```mermaid
graph TD
    A[Start Local Pipeline] --> B{Phase 1: Data Ingestion};
    B --> B1[fetch_defillama_pools.py];
    B1 --> B2[fetch_ohlcv_coinmarketcap.py];
    B2 --> B3[fetch_gas_ethgastracker.py];
    B3 --> B4[fetch_account_data_etherscan.py];
    B4 --> C{Phase 3: Data Processing & Transformation};
    C --> C1[process_icebox_logic.py];
    C1 --> C2[calculate_pool_metrics.py];
    C2 --> C3[apply_pool_grouping.py];
    C3 --> C4[filter_pools.py];
    C4 --> D{Phase 4: Forecasting};
    D --> D1[forecast_pools.py];
    D1 --> D2[forecast_gas_fees.py];
    D2 --> E{Phase 5: Asset Allocation};
    E --> E1[optimize_allocations.py];
    E1 --> E2[optimize_allocations_milp.py];
    E2 --> F{Phase 6: Reporting & Notification};
    F --> F1[manage_ledger.py];
    F1 --> F2[post_slack_notification.py];
    F2 --> G[End Local Pipeline];