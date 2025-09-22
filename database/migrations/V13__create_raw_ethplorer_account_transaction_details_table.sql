CREATE TABLE raw_ethplorer_account_transaction_details (
    transaction_hash VARCHAR(255) PRIMARY KEY,
    raw_json JSONB NOT NULL,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reatd_fetched_at ON raw_ethplorer_account_transaction_details(fetched_at);