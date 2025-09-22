ALTER TABLE account_transactions
    ADD COLUMN block_number INTEGER,
    ADD COLUMN confirmations INTEGER,
    ADD COLUMN success BOOLEAN,
    ADD COLUMN transaction_index INTEGER,
    ADD COLUMN nonce BIGINT,
    ADD COLUMN raw_value NUMERIC,
    ADD COLUMN input_data TEXT,
    ADD COLUMN gas_limit NUMERIC,
    ADD COLUMN gas_price NUMERIC,
    ADD COLUMN gas_used NUMERIC,
    ADD COLUMN method_id VARCHAR(20),
    ADD COLUMN function_name TEXT,
    ADD COLUMN creates VARCHAR(255);

-- Optional: index on block_number for queries by block
CREATE INDEX IF NOT EXISTS idx_at_block_number ON account_transactions(block_number);