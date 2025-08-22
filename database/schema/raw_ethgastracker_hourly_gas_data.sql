CREATE TABLE IF NOT EXISTS raw_ethgastracker_hourly_gas_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    raw_json_data JSONB
);