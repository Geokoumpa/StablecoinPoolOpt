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