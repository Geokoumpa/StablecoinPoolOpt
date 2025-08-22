CREATE TABLE pools (
    pool_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    chain VARCHAR(255) NOT NULL,
    protocol VARCHAR(255) NOT NULL,
    symbol VARCHAR(255) NOT NULL,
    tvl DECIMAL(20, 2),
    apy DECIMAL(20, 4),
    last_updated TIMESTAMP WITH TIME ZONE
);