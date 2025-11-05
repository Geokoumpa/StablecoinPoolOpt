-- Add poolMeta column to pools table
-- This will store the poolMeta field from the DeFiLlama API response

ALTER TABLE pools ADD COLUMN poolMeta TEXT;

-- Add comment to explain the purpose
COMMENT ON COLUMN pools.poolMeta IS 'Pool metadata from DeFiLlama API response (stored as TEXT)';