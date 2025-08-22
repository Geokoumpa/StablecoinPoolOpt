# Phase 1: Data Ingestion Layer

This phase focuses on setting up the initial data collection mechanisms and storing raw data from various external APIs.

## Detailed Tasks:

### 1.1 Database Schema Design for Raw Data - **COMPLETED**
- [x] Define SQL schemas for the following tables:
    - `raw_defillama_pools`: Stores raw JSON responses from the DeFiLlama pools endpoint.
        - Fields: `id` (PK), `timestamp`, `raw_json_data` (JSONB)
    - `raw_defillama_pool_history`: Stores raw JSON responses for individual pool histories.
        - Fields: `id` (PK), `pool_id` (FK), `timestamp`, `raw_json_data` (JSONB)
    - `raw_ethgastracker_hourly_gas_data`: Stores raw hourly average gas data from EthGasTracker.com.
        - Fields: `id` (PK), `timestamp`, `raw_json_data` (JSONB)
    - `raw_etherscan_account_transactions`: Stores raw transaction history from Etherscan Accounts API.
        - Fields: `id` (PK), `timestamp`, `raw_json_data` (JSONB)
    - `raw_etherscan_account_balances`: Stores raw historical balance data from Etherscan Accounts API.
        - Fields: `id` (PK), `timestamp`, `raw_json_data` (JSONB)
- [x] Consider indexing strategies for `timestamp` and `pool_id` fields.

### 1.2 Implement `fetch_defillama_pools.py` - **COMPLETED**
- [x] Create a Python script to:
    - Connect to the Defillama API (`https://yields.llama.fi/pools`).
    - Fetch all available pool data.
    - Store the raw JSON response in the `raw_defillama_pools` table.
    - Parse relevant metadata (e.g., pool name, chain, protocol, symbol) and update/insert into the `pools` master table.
    - Implement error handling and retry logic for API calls.

### 1.3 Implement `fetch_ohlcv_coinmarketcap.py` - **COMPLETED**
- [x] Create a Python script to:
    - Connect to the CoinMarketCap API.
    - Fetch OHLCV data for major cryptocurrencies (BTC, ETH) and *approved* stablecoins.
    - Store the raw JSON response in a suitable raw data table (e.g., `raw_coinmarketcap_ohlcv`).
    - Implement logic to handle API rate limits and pagination.

### 1.4 Implement `fetch_gas_ethgastracker.py` - **COMPLETED**
- [x] Create a Python script to:
    - Connect to EthGasTracker.com API (`/gas/average` for historical/daily averages and `/gas/latest` for real-time data).
    - Fetch hourly gas fee data.
    - Store the raw JSON response in the `raw_ethgastracker_hourly_gas_data` table.
    - Ensure proper timestamping for hourly records.

### 1.5 Implement `fetch_account_data_etherscan.py` - **COMPLETED**
- [x] Create a Python script to:
    - Connect to the Etherscan Accounts API.
    - Fetch real-time and historical balance data for the main asset holding address.
    - Store raw balance data in `raw_etherscan_account_balances`.
    - Fetch transaction history for the main asset holding address.
    - Store raw transaction data in `raw_etherscan_account_transactions`.
    - Implement secure handling of API keys.