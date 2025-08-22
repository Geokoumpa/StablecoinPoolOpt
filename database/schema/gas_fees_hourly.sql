CREATE TABLE gas_fees_hourly (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    gas_price_gwei DECIMAL(10, 2),
    estimated_gas_usd DECIMAL(10, 4),
    estimated_gas_eth DECIMAL(20, 10)
);