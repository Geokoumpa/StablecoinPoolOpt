ALTER TABLE pool_daily_metrics
ADD COLUMN rolling_apy_7d DECIMAL(20, 4),
ADD COLUMN rolling_apy_30d DECIMAL(20, 4),
ADD COLUMN apy_delta_today_yesterday DECIMAL(20, 4),
ADD COLUMN stddev_apy_7d DECIMAL(20, 4),
ADD COLUMN stddev_apy_30d DECIMAL(20, 4),
ADD COLUMN stddev_apy_7d_delta DECIMAL(20, 4),
ADD COLUMN stddev_apy_30d_delta DECIMAL(20, 4);