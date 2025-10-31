CREATE TABLE default_allocation_parameters (
    id SERIAL PRIMARY KEY,
    parameter_name VARCHAR(255) UNIQUE NOT NULL,
    parameter_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert default values
INSERT INTO default_allocation_parameters (parameter_name, parameter_value, description) VALUES
('tvl_limit_percentage', '0.05', 'Maximum percentage of pool TVL that can be allocated'),
('max_alloc_percentage', '0.25', 'Maximum allocation to any single pool'),
('conversion_rate', '0.0004', 'Token conversion fee rate'),
('min_pools', '5', 'Minimum number of pools in allocation'),
('profit_optimization', 'true', 'Enable profit optimization mode'),
('token_marketcap_limit', '1000000000', 'Minimum token market cap requirement'),
('pool_tvl_limit', '100000', 'Minimum pool TVL requirement'),
('pool_apy_limit', '0.01', 'Minimum pool APY requirement'),
('pool_pair_tvl_ratio_min', '0.3', 'Minimum TVL ratio for pool pairs'),
('pool_pair_tvl_ratio_max', '0.5', 'Maximum TVL ratio for pool pairs'),
('group1_max_pct', '0.35', 'Maximum allocation percentage for group 1'),
('group2_max_pct', '0.35', 'Maximum allocation percentage for group 2'),
('group3_max_pct', '0.3', 'Maximum allocation percentage for group 3'),
('position_max_pct_total_assets', '0.25', 'Maximum position percentage of total assets'),
('position_max_pct_pool_tvl', '0.05', 'Maximum position percentage of pool TVL'),
('group1_apy_delta_max', '0.01', 'Maximum APY delta for group 1'),
('group1_7d_stddev_max', '0.015', 'Maximum 7-day standard deviation for group 1'),
('group1_30d_stddev_max', '0.02', 'Maximum 30-day standard deviation for group 1'),
('group2_apy_delta_max', '0.03', 'Maximum APY delta for group 2'),
('group2_7d_stddev_max', '0.04', 'Maximum 7-day standard deviation for group 2'),
('group2_30d_stddev_max', '0.05', 'Maximum 30-day standard deviation for group 2'),
('group3_apy_delta_min', '0.03', 'Minimum APY delta for group 3'),
('group3_7d_stddev_min', '0.04', 'Minimum 7-day standard deviation for group 3'),
('group3_30d_stddev_min', '0.02', 'Minimum 30-day standard deviation for group 3'),
('icebox_ohlc_l_threshold_pct', '0.02', 'Icebox OHLC L threshold percentage'),
('icebox_ohlc_l_days_threshold', '2', 'Icebox OHLC L days threshold'),
('icebox_ohlc_c_threshold_pct', '0.01', 'Icebox OHLC C threshold percentage'),
('icebox_ohlc_c_days_threshold', '1', 'Icebox OHLC C days threshold'),
('icebox_recovery_l_days_threshold', '2', 'Icebox recovery L days threshold'),
('icebox_recovery_c_days_threshold', '3', 'Icebox recovery C days threshold');

-- Create index for faster lookups
CREATE INDEX idx_default_allocation_parameters_name ON default_allocation_parameters(parameter_name);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_default_allocation_parameters_updated_at 
    BEFORE UPDATE ON default_allocation_parameters 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();