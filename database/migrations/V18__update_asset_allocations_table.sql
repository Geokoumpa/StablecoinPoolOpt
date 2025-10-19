DROP TABLE IF EXISTS asset_allocations;

CREATE TABLE asset_allocations (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    step_number INT NOT NULL,
    operation VARCHAR(50) NOT NULL,
    from_asset VARCHAR(50),
    to_asset VARCHAR(50),
    amount DECIMAL(20, 8),
    pool_id VARCHAR(255),
    FOREIGN KEY (pool_id) REFERENCES pools(pool_id),
    UNIQUE (run_id, step_number)
);