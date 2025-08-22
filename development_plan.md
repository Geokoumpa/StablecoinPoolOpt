# DefiYieldOpt Development Plan

This document outlines the detailed development plan for the DefiYieldOpt application, prioritizing core data pipeline and backtesting functionality, with the Admin Web Application as the lowest priority.

## Phase 1: Data Ingestion Layer ([Detailed Plan](plan/phase1_data_ingestion.md)) - **COMPLETED**
- [x] Design and implement database schemas for raw data tables (`raw_defillama_pools`, `raw_defillama_pool_history`, `raw_ethgastracker_hourly_gas_data`, `raw_etherscan_account_transactions`, `raw_etherscan_account_balances`).
- [x] Implement `fetch_defillama_pools.py` to fetch pool data and update `pools` metadata.
- [x] Implement `fetch_defillama_pool_history.py` to fetch historical pool data.
- [x] Implement `fetch_ohlcv_coinmarketcap.py` for Crypto OHLCV data.
- [x] Implement `fetch_gas_ethgastracker.py` for gas fees.
- [x] Implement `fetch_account_data_etherscan.py` for account balances and transaction history.

## Phase 2: Data Storage Layer ([Detailed Plan](plan/phase2_data_storage.md)) - **COMPLETED**
- Design and implement database schemas for processed/enriched data tables (`pools`, `pool_daily_metrics`, `gas_fees_hourly`, `gas_fees_daily`, `icebox_tokens`, `approved_tokens`, `blacklisted_tokens`, `approved_protocols`, `pool_groups`, `daily_ledger`, `asset_allocations`, `allocation_parameters`).
- Set up GCP Cloud SQL with PostgreSQL using Terraform, including provisioning the instance, configuring users, permissions, network access, and implementing initial database creation and schema application scripts.

## Phase 3: Data Processing & Transformation Layer ([Detailed Plan](plan/phase3_data_processing_transformation.md)) - **COMPLETED**
- [x] Implement `process_icebox_logic.py` to apply Icebox rules.
- [x] Implement `calculate_pool_metrics.py` for 7-day and 30-day rolling APY and standard deviations.
- [x] Implement `apply_pool_grouping.py` to assign pools to groups.
- [x] Implement `filter_pools.py` to apply all pool filtering criteria.

## Phase 4: Forecasting Layer ([Detailed Plan](plan/phase4_forecasting.md)) - **COMPLETED**
- [x] Implement data preprocessing and feature engineering for forecasting models.
- [x] Implement `forecast_pools.py` for APY and TVL forecasts.
- [x] Implement `forecast_gas_fees.py` for gas fee actuals and forecasts.
- [x] Integrate `skforecast`, `xgboost`, and `pickle` for model training, persistence, and forecasting.

## Phase 5: Asset Allocation Layer ([Detailed Plan](plan/phase5_asset_allocation.md)) - **COMPLETED**
- [x] Implement `optimize_allocations.py` for the core asset allocation optimization logic using `cvxpy`.
- [x] Implement snapshotting functionality for allocation parameters.
- [x] Implement `optimize_allocations_milp.py` for MILP approach using CP-SAT.
- [x] Implement profit optimization flag logic and conditional second optimization run.
- [x] Implement group allocation limits.
- [ ] **Future Refinement:** Implement `asset_allocation_details` table for granular allocation data.
- [ ] **Future Refinement:** Enhance `should_force_reallocation` for more robust checks.

## Phase 6: Reporting & Notification Layer ([Detailed Plan](plan/phase6_reporting_notification.md)) - **COMPLETED**
- [x] Implement `manage_ledger.py` to update daily token balances and NAV.
- [x] Implement `post_slack_notification.py` for daily allocation recommendations.

## Phase 7: Local Orchestration for Testing ([Detailed Plan](plan/phase7_local_orchestration.md))
- Implement `main_pipeline.py` to sequentially execute all data ingestion, processing, forecasting, and asset allocation scripts for local testing.
- Incorporate robust error handling and logging within `main_pipeline.py`.
- Ensure `main_pipeline.py` can be configured to run specific phases or the entire pipeline.
- Reference the [Local Orchestration Pipeline Diagram](local_orchestration_diagram.md) for the execution flow.

## Phase 8: Orchestration & Deployment ([Detailed Plan](plan/phase8_orchestration_deployment.md))
- Implement a central orchestration mechanism (e.g., GCP Workflows).
- Set up GCP Cloud Scheduler for daily pipeline runs.
- Containerize pipeline components using Docker.
- Define and deploy GCP Cloud Run jobs using Terraform.

## Phase 9: Backtesting Framework ([Detailed Plan](plan/phase9_backtesting_framework.md))
- Develop the backtesting simulation engine to re-run the pipeline with historical data.
- Implement performance metric calculation (returns, drawdowns, Sharpe ratio, transaction costs).
- Develop reporting and visualization for backtesting results.

## Phase 10: Admin Web Application (Lowest Priority) ([Detailed Plan](plan/phase10_admin_web_application.md))
- Set up Next.js frontend project.
- Integrate Clerk for authentication.
- Develop configuration management UI for `allocation_parameters`, approved/blacklisted tokens/protocols, and Icebox thresholds.
- Develop ledger visualization dashboards for `daily_ledger` data.
- Implement account performance and audit trail features.
- Deploy Admin Web Application to GCP Cloud Run.