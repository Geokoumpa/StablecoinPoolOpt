CREATE TABLE account_transactions (
    transaction_hash VARCHAR(255) NOT NULL,
    operation_index INT NOT NULL,
    id SERIAL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    from_address VARCHAR(255),
    to_address VARCHAR(255),
    value NUMERIC,
    token_symbol VARCHAR(50),
    token_decimals INTEGER,
    raw_transaction_id INTEGER,
    insertion_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    block_number INTEGER,
    confirmations INTEGER,
    success BOOLEAN,
    transaction_index INTEGER,
    nonce BIGINT,
    raw_value NUMERIC,
    input_data TEXT,
    gas_limit NUMERIC,
    gas_price NUMERIC,
    gas_used NUMERIC,
    method_id VARCHAR(20),
    function_name TEXT,
    creates VARCHAR(255),
    operation_type VARCHAR(255),
    operation_priority INT,
    token_address VARCHAR(255),
    token_name VARCHAR(255),
    token_price_rate NUMERIC,
    token_price_currency VARCHAR(10),
    PRIMARY KEY (transaction_hash, operation_index),
    FOREIGN KEY (raw_transaction_id) REFERENCES raw_ethplorer_account_transactions(id)
);

CREATE INDEX idx_at_transaction_hash ON account_transactions(transaction_hash);
CREATE INDEX idx_at_timestamp ON account_transactions(timestamp);
CREATE INDEX idx_at_block_number ON account_transactions(block_number);