-- Add column to store normalized token symbols for pools with partial matches
-- This will be used by the optimization module to fetch correct prices
ALTER TABLE pool_daily_metrics ADD COLUMN normalized_tokens TEXT;

-- Add comment to explain the purpose
COMMENT ON COLUMN pool_daily_metrics.normalized_tokens IS 'JSON string containing normalized token mappings for pools with partial token matches. Format: {"original_token": "approved_token"}';