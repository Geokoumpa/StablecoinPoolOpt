CREATE TABLE account_transactions (
    id SERIAL PRIMARY KEY,
    transaction_hash VARCHAR(255) UNIQUE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    from_address VARCHAR(255),
    to_address VARCHAR(255),
    value NUMERIC,
    token_symbol VARCHAR(50),
    token_decimals INTEGER,
    raw_transaction_id INTEGER REFERENCES raw_ethplorer_account_transactions(id),
    insertion_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_at_transaction_hash ON account_transactions(transaction_hash);
CREATE INDEX idx_at_timestamp ON account_transactions(timestamp);