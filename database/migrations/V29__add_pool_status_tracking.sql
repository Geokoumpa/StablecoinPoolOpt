-- Add is_active column to pools table
-- This migration enables tracking of active/inactive status for pools that no longer exist in DeFiLlama API

-- Add status tracking column
ALTER TABLE pools ADD COLUMN is_active BOOLEAN DEFAULT TRUE;

-- Add index for performance
CREATE INDEX idx_pools_is_active ON pools(is_active);

-- Add comment to explain purpose
COMMENT ON COLUMN pools.is_active IS 'Flag indicating if pool is currently active (exists in DeFiLlama API)';