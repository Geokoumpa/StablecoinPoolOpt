CREATE TABLE pool_daily_metrics (
    id SERIAL PRIMARY KEY,
    pool_id VARCHAR(255) NOT NULL,
    date DATE NOT NULL,
    actual_apy DECIMAL(20, 4),
    forecasted_apy DECIMAL(20, 4),
    actual_tvl DECIMAL(20, 2),
    forecasted_tvl DECIMAL(20, 2),
    is_filtered_out BOOLEAN DEFAULT FALSE,
    filter_reason TEXT,
    rolling_apy_7d DECIMAL(20, 4),
    rolling_apy_30d DECIMAL(20, 4),
    apy_delta_today_yesterday DECIMAL(20, 4),
    stddev_apy_7d DECIMAL(20, 4),
    stddev_apy_30d DECIMAL(20, 4),
    stddev_apy_7d_delta DECIMAL(20, 4),
    stddev_apy_30d_delta DECIMAL(20, 4),
    FOREIGN KEY (pool_id) REFERENCES pools(pool_id)
);