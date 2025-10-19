DROP TABLE IF EXISTS daily_ledger;

CREATE TABLE daily_balances (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    token_symbol VARCHAR(255) NOT NULL,
    wallet_address VARCHAR(255),
    unallocated_balance DECIMAL(20, 10),
    allocated_balance DECIMAL(20, 10),
    pool_id VARCHAR(255)
);