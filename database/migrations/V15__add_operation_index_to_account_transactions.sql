-- Add columns to store detailed information for each transaction operation
ALTER TABLE account_transactions ADD COLUMN operation_index INT;
ALTER TABLE account_transactions ADD COLUMN operation_type VARCHAR(255);
ALTER TABLE account_transactions ADD COLUMN operation_priority INT;
ALTER TABLE account_transactions ADD COLUMN token_address VARCHAR(255);
ALTER TABLE account_transactions ADD COLUMN token_name VARCHAR(255);
ALTER TABLE account_transactions ADD COLUMN token_price_rate NUMERIC;
ALTER TABLE account_transactions ADD COLUMN token_price_currency VARCHAR(10);

-- To ensure that each operation within a transaction is unique, the primary key must be a composite of transaction_hash and operation_index.
-- First, we must drop the existing primary key and any unique constraints on transaction_hash.
ALTER TABLE account_transactions DROP CONSTRAINT IF EXISTS account_transactions_pkey;
ALTER TABLE account_transactions DROP CONSTRAINT IF EXISTS account_transactions_transaction_hash_key;

-- Now, create a new composite primary key.
ALTER TABLE account_transactions ADD PRIMARY KEY (transaction_hash, operation_index);