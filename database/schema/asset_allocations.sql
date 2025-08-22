CREATE TABLE asset_allocations (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    pool_id VARCHAR(255) NOT NULL,
    allocated_amount_usd DECIMAL(20, 2),
    allocation_percentage DECIMAL(5, 4),
    FOREIGN KEY (pool_id) REFERENCES pools(pool_id)
);