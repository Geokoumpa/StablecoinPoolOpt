-- Create optimized database view for pool metrics with extended information
CREATE VIEW pool_metrics_extended AS
SELECT 
    p.pool_id,
    p.name,
    p.protocol,
    p.chain,
    p.pool_address as address,
    p.currently_filtered_out,
    p.last_updated,
    pdm.actual_apy,
    pdm.forecasted_apy,
    pdm.actual_tvl,
    pdm.forecasted_tvl,
    pdm.date,
    pdm.is_filtered_out,
    pdm.filter_reason
FROM pools p
JOIN pool_daily_metrics pdm ON p.pool_id = pdm.pool_id;

-- Create index for better performance on the view
CREATE INDEX idx_pool_metrics_extended_pool_id ON pool_daily_metrics(pool_id);
CREATE INDEX idx_pool_metrics_extended_date ON pool_daily_metrics(date);