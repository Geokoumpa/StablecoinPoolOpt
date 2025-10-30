-- Migration to remove normalized_tokens column from pool_daily_metrics table
-- This column is no longer needed as we now use underlying_tokens exclusively

-- Drop the column
ALTER TABLE pool_daily_metrics DROP COLUMN normalized_tokens;

-- Add comment to document the change
COMMENT ON TABLE pool_daily_metrics IS 'normalized_tokens column removed - now using underlying_tokens exclusively for address-based filtering';