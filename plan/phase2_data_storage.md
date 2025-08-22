# Phase 2: Data Storage Layer

This phase focuses on designing and implementing the database schemas for processed and enriched data, and setting up the PostgreSQL database on GCP Cloud SQL.

## Detailed Tasks:

### 2.1 Database Schema Design for Processed/Enriched Data
- [x] Define SQL schemas for the following tables:
    - `pools`: Master table for all identified DeFi pools, with core metadata.
        - Fields: `pool_id` (PK), `name`, `chain`, `protocol`, `symbol`, `tvl`, `apy`, `last_updated`
    - `pool_daily_metrics`: Stores daily historical and forecasted metrics for pools (APY, TVL). This will be the "enriched" data.
        - Fields: `id` (PK), `pool_id` (FK), `date`, `actual_apy`, `forecasted_apy`, `actual_tvl`, `forecasted_tvl`, `is_filtered_out`, `filter_reason`
    - `gas_fees_hourly`: Stores processed gas fee data at a higher granularity (e.g., hourly averages from 5-min data). Daily averages for forecasting are computed from this table.
        - Fields: `id` (PK), `timestamp`, `gas_price_gwei`, `estimated_gas_usd`, `estimated_gas_eth`
    - `gas_fees_daily`: Stores daily actual and forecasted gas fees, derived from hourly data.
        - Fields: `id` (PK), `date`, `actual_avg_gas_gwei`, `forecasted_avg_gas_gwei`, `actual_max_gas_gwei`, `forecasted_max_gas_gwei`
    - `icebox_tokens`: Stores the current Icebox token list, with timestamps for when tokens are added/removed and the reason for exclusion.
        - Fields: `id` (PK), `token_symbol`, `added_timestamp`, `removed_timestamp`, `reason`
    - `approved_tokens`: Tracks the list of approved tokens for each run, with add/remove timestamps.
        - Fields: `id` (PK), `token_symbol`, `added_timestamp`, `removed_timestamp`
    - `blacklisted_tokens`: Tracks the list of blacklisted tokens for each run, with add/remove timestamps.
        - Fields: `id` (PK), `token_symbol`, `added_timestamp`, `removed_timestamp`
    - `approved_protocols`: Tracks the list of approved protocols for each run, with add/remove timestamps.
        - Fields: `id` (PK), `protocol_name`, `added_timestamp`, `removed_timestamp`
    - `pool_groups`: Stores pool grouping (Group 1, 2, 3) and their assignment per day, including criteria snapshot.
        - Fields: `id` (PK), `pool_id` (FK), `date`, `group_assignment`, `apy_delta`, `7d_stddev`, `30d_stddev`
    - `daily_ledger`: Tracks daily balances and NAV for each token.
        - Fields: `id` (PK), `date`, `token_symbol`, `start_of_day_balance`, `end_of_day_balance`, `daily_nav`, `realized_yield_yesterday`, `realized_yield_ytd`
    - `asset_allocations`: Stores the results of the daily optimization, including allocated amounts per pool.
        - Fields: `id` (PK), `run_id` (FK to `allocation_parameters`), `pool_id` (FK), `allocated_amount_usd`, `allocation_percentage`
    - `allocation_parameters`: Stores the configurable parameters used for each optimization run, for reproducibility and auditing.
        - Fields: `run_id` (PK), `timestamp`, `tvl_limit_percentage`, `max_alloc_percentage`, `conversion_rate`, `min_pools`, `profit_optimization`, `approved_tokens_snapshot` (JSONB), `blacklisted_tokens_snapshot` (JSONB), `approved_protocols_snapshot` (JSONB), `icebox_tokens_snapshot` (JSONB), `token_marketcap_limit`, `pool_tvl_limit`, `pool_apy_limit`, `pool_pair_tvl_ratio_min`, `pool_pair_tvl_ratio_max`, `group1_max_pct`, `group2_max_pct`, `group3_max_pct`, `position_max_pct_total_assets`, `position_max_pct_pool_tvl`, `group1_apy_delta_max`, `group1_7d_stddev_max`, `group1_30d_stddev_max`, `group2_apy_delta_max`, `group2_7d_stddev_max`, `group2_30d_stddev_max`, `group3_apy_delta_min`, `group3_7d_stddev_min`, `group3_30d_stddev_min`, `other_dynamic_limits` (JSONB)
- [x] Establish foreign key relationships and appropriate indexing.

### 2.2 Set up GCP Cloud SQL with PostgreSQL using Terraform
- [x] Create Terraform configuration files for GCP Cloud SQL PostgreSQL instance provisioning, including `variables.tf`, `main.tf`, and `outputs.tf`.
- [x] Configure database users, permissions, and network access through Terraform (e.g., private IP, authorized networks).
- [x] Implement initial database creation and schema application scripts (e.g., using Alembic or raw SQL files) as part of the Terraform setup or a subsequent step.
- [x] Set up backups and monitoring for the database instance through Terraform.
- [x] Define Terraform variables for database configuration (instance class, storage, region, etc.) in `variables.tf`.
- [x] Create Terraform outputs for database connection details in `outputs.tf`.
- [x] Ensure Terraform state is managed in a GCS bucket (e.g., `defiyieldopt-terraform-state`).
- [x] Test the Terraform setup (init, plan, apply) to ensure successful provisioning of the Cloud SQL instance, database, and user.