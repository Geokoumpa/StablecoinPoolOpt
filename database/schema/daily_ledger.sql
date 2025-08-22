CREATE TABLE daily_ledger (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    token_symbol VARCHAR(255) NOT NULL,
    start_of_day_balance DECIMAL(20, 10),
    end_of_day_balance DECIMAL(20, 10),
    daily_nav DECIMAL(20, 2),
    realized_yield_yesterday DECIMAL(20, 4),
    realized_yield_ytd DECIMAL(20, 4)
);