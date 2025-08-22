CREATE TABLE pool_groups (
    id SERIAL PRIMARY KEY,
    pool_id VARCHAR(255) NOT NULL,
    date DATE NOT NULL,
    group_assignment VARCHAR(50) NOT NULL,
    apy_delta DECIMAL(20, 4),
    "7d_stddev" DECIMAL(20, 4),
    "30d_stddev" DECIMAL(20, 4),
    FOREIGN KEY (pool_id) REFERENCES pools(pool_id)
);