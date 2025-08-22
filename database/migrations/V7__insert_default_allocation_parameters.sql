-- Insert default allocation parameters
INSERT INTO allocation_parameters (
    run_id,
    tvl_limit_percentage,
    max_alloc_percentage,
    conversion_rate,
    min_pools,
    profit_optimization,
    token_marketcap_limit,
    pool_tvl_limit,
    pool_apy_limit,
    pool_pair_tvl_ratio_min,
    pool_pair_tvl_ratio_max,
    group1_max_pct,
    group2_max_pct,
    group3_max_pct,
    position_max_pct_total_assets,
    position_max_pct_pool_tvl,
    group1_apy_delta_max,
    group1_7d_stddev_max,
    group1_30d_stddev_max,
    group2_apy_delta_max,
    group2_7d_stddev_max,
    group2_30d_stddev_max,
    group3_apy_delta_min,
    group3_7d_stddev_min,
    group3_30d_stddev_min,
    icebox_ohlc_l_threshold_pct,
    icebox_ohlc_l_days_threshold,
    icebox_ohlc_c_threshold_pct,
    icebox_ohlc_c_days_threshold,
    icebox_recovery_l_days_threshold,
    icebox_recovery_c_days_threshold
) VALUES (
    gen_random_uuid(), -- Generate a new UUID for run_id
    0.05,   -- tvl_limit_percentage (default: 5%)
    0.25,   -- max_alloc_percentage (default: 25%)
    0.0004, -- conversion_rate (default: 0.04%)
    4,      -- min_pools (default: 4)
    false,  -- profit_optimization (default: false)
    35000000, -- token_marketcap_limit ($35M)
    500000,   -- pool_tvl_limit ($500K)
    0.06,     -- pool_apy_limit (6%)
    0.3,      -- pool_pair_tvl_ratio_min (30%)
    0.5,      -- pool_pair_tvl_ratio_max (50%)
    0.35,     -- group1_max_pct (35%)
    0.35,     -- group2_max_pct (35%)
    0.30,     -- group3_max_pct (30%)
    0.25,     -- position_max_pct_total_assets (25%)
    0.05,     -- position_max_pct_pool_tvl (5%)
    0.01,     -- group1_apy_delta_max (1%)
    0.015,    -- group1_7d_stddev_max (1.5%)
    0.02,     -- group1_30d_stddev_max (2%)
    0.03,     -- group2_apy_delta_max (3%)
    0.04,     -- group2_7d_stddev_max (4%)
    0.05,     -- group2_30d_stddev_max (5%)
    0.03,     -- group3_apy_delta_min (>3%)
    0.04,     -- group3_7d_stddev_min (>4%)
    0.02,     -- group3_30d_stddev_min (>2%)
    0.02,     -- icebox_ohlc_l_threshold_pct (2%)
    2,        -- icebox_ohlc_l_days_threshold (2 days)
    0.01,     -- icebox_ohlc_c_threshold_pct (1%)
    1,        -- icebox_ohlc_c_days_threshold (1 day)
    2,        -- icebox_recovery_l_days_threshold (2 days)
    3         -- icebox_recovery_c_days_threshold (3 days)
);