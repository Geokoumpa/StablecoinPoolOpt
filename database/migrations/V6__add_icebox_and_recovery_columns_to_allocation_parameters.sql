ALTER TABLE allocation_parameters
ADD COLUMN icebox_ohlc_l_threshold_pct DECIMAL(5, 4),
ADD COLUMN icebox_ohlc_l_days_threshold INTEGER,
ADD COLUMN icebox_ohlc_c_threshold_pct DECIMAL(5, 4),
ADD COLUMN icebox_ohlc_c_days_threshold INTEGER,
ADD COLUMN icebox_recovery_l_days_threshold INTEGER,
ADD COLUMN icebox_recovery_c_days_threshold INTEGER;