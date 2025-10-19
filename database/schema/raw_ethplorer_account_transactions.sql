CREATE TABLE raw_ethplorer_account_transactions (
    id SERIAL PRIMARY KEY,
    raw_json_data JSONB NOT NULL,
    insertion_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);