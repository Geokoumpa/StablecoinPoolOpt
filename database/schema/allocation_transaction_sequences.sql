-- This table is referenced in the schema but may not be fully implemented yet
-- Placeholder for future allocation transaction sequence tracking
CREATE TABLE allocation_transaction_sequences (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    sequence_number INT NOT NULL,
    transaction_hash VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES allocation_parameters(run_id)
);