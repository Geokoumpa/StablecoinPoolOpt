CREATE TABLE IF NOT EXISTS raw_etherscan_account_transactions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    raw_json_data JSONB
);