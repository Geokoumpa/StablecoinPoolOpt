-- This table is referenced in the schema but may not be fully implemented yet
-- Placeholder for future transaction operation tracking and analysis
CREATE TABLE transaction_operations (
    id SERIAL PRIMARY KEY,
    transaction_hash VARCHAR(255) NOT NULL,
    operation_type VARCHAR(50) NOT NULL,
    operation_status VARCHAR(20) DEFAULT 'pending',
    pool_id VARCHAR(255),
    amount DECIMAL(20, 8),
    token_symbol VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP WITH TIME ZONE,
    FOREIGN KEY (pool_id) REFERENCES pools(pool_id)
);

CREATE INDEX idx_to_transaction_hash ON transaction_operations(transaction_hash);
CREATE INDEX idx_to_operation_type ON transaction_operations(operation_type);
CREATE INDEX idx_to_status ON transaction_operations(operation_status);