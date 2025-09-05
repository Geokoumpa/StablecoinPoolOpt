-- Create pools table
CREATE TABLE pools (
    pool_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    chain VARCHAR(255) NOT NULL,
    protocol VARCHAR(255) NOT NULL,
    symbol VARCHAR(255) NOT NULL,
    tvl DECIMAL(20, 2),
    apy DECIMAL(20, 4),
    last_updated TIMESTAMP WITH TIME ZONE
);

-- Create pool_daily_metrics table
CREATE TABLE pool_daily_metrics (
    id SERIAL PRIMARY KEY,
    pool_id VARCHAR(255) NOT NULL,
    date DATE NOT NULL,
    actual_apy DECIMAL(20, 4),
    forecasted_apy DECIMAL(20, 4),
    actual_tvl DECIMAL(20, 2),
    forecasted_tvl DECIMAL(20, 2),
    eth_open DECIMAL(20, 8),
    btc_open DECIMAL(20, 8),
    gas_price_gwei DECIMAL(20, 4),
    is_filtered_out BOOLEAN DEFAULT FALSE,
    filter_reason TEXT,
    FOREIGN KEY (pool_id) REFERENCES pools(pool_id),
    UNIQUE (pool_id, date)
);

-- Create gas_fees_hourly table
CREATE TABLE gas_fees_hourly (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    gas_price_gwei DECIMAL(10, 2)
);

-- Create gas_fees_daily table
CREATE TABLE gas_fees_daily (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    actual_avg_gas_gwei DECIMAL(10, 2),
    forecasted_avg_gas_gwei DECIMAL(10, 2),
    actual_max_gas_gwei DECIMAL(10, 2),
    forecasted_max_gas_gwei DECIMAL(10, 2),
    eth_open DECIMAL(20, 8),
    btc_open DECIMAL(20, 8)
);

-- Create icebox_tokens table
CREATE TABLE icebox_tokens (
    id SERIAL PRIMARY KEY,
    token_symbol VARCHAR(255) NOT NULL UNIQUE,
    added_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    removed_timestamp TIMESTAMP WITH TIME ZONE,
    reason TEXT
);

-- Create approved_tokens table
CREATE TABLE approved_tokens (
    id SERIAL PRIMARY KEY,
    token_symbol VARCHAR(255) NOT NULL UNIQUE,
    added_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    removed_timestamp TIMESTAMP WITH TIME ZONE
);

-- Create blacklisted_tokens table
CREATE TABLE blacklisted_tokens (
    id SERIAL PRIMARY KEY,
    token_symbol VARCHAR(255) NOT NULL UNIQUE,
    added_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    removed_timestamp TIMESTAMP WITH TIME ZONE
);

-- Create approved_protocols table
CREATE TABLE approved_protocols (
    id SERIAL PRIMARY KEY,
    protocol_name VARCHAR(255) NOT NULL UNIQUE,
    added_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    removed_timestamp TIMESTAMP WITH TIME ZONE
);

-- Create pool_groups table
CREATE TABLE pool_groups (
    id SERIAL PRIMARY KEY,
    pool_id VARCHAR(255) NOT NULL,
    date DATE NOT NULL,
    group_assignment VARCHAR(50) NOT NULL,
    apy_delta DECIMAL(20, 4),
    "7d_stddev" DECIMAL(20, 4),
    "30d_stddev" DECIMAL(20, 4),
    FOREIGN KEY (pool_id) REFERENCES pools(pool_id)
);

-- Create daily_ledger table
CREATE TABLE daily_ledger (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    token_symbol VARCHAR(255) NOT NULL,
    start_of_day_balance DECIMAL(20, 10),
    end_of_day_balance DECIMAL(20, 10),
    daily_nav DECIMAL(20, 2),
    realized_yield_yesterday DECIMAL(20, 4),
    realized_yield_ytd DECIMAL(20, 4),
    UNIQUE (date, token_symbol)
);

-- Create asset_allocations table
CREATE TABLE asset_allocations (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    pool_id VARCHAR(255) NOT NULL,
    allocated_amount_usd DECIMAL(20, 2),
    allocation_percentage DECIMAL(5, 4),
    FOREIGN KEY (pool_id) REFERENCES pools(pool_id)
);

-- Create allocation_parameters table
CREATE TABLE allocation_parameters (
    run_id UUID PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tvl_limit_percentage DECIMAL(5, 4),
    max_alloc_percentage DECIMAL(5, 4),
    conversion_rate DECIMAL(20, 10),
    min_pools INTEGER,
    profit_optimization BOOLEAN,
    approved_tokens_snapshot JSONB,
    blacklisted_tokens_snapshot JSONB,
    approved_protocols_snapshot JSONB,
    icebox_tokens_snapshot JSONB,
    token_marketcap_limit DECIMAL(20, 2),
    pool_tvl_limit DECIMAL(20, 2),
    pool_apy_limit DECIMAL(20, 4),
    pool_pair_tvl_ratio_min DECIMAL(5, 4),
    pool_pair_tvl_ratio_max DECIMAL(5, 4),
    group1_max_pct DECIMAL(5, 4),
    group2_max_pct DECIMAL(5, 4),
    group3_max_pct DECIMAL(5, 4),
    position_max_pct_total_assets DECIMAL(5, 4),
    position_max_pct_pool_tvl DECIMAL(5, 4),
    group1_apy_delta_max DECIMAL(20, 4),
    group1_7d_stddev_max DECIMAL(20, 4),
    group1_30d_stddev_max DECIMAL(20, 4),
    group2_apy_delta_max DECIMAL(20, 4),
    group2_7d_stddev_max DECIMAL(20, 4),
    group2_30d_stddev_max DECIMAL(20, 4),
    group3_apy_delta_min DECIMAL(20, 4),
    group3_7d_stddev_min DECIMAL(20, 4),
    group3_30d_stddev_min DECIMAL(20, 4),
    other_dynamic_limits JSONB
);