-- Add pool_address column to pools table to store blockchain addresses
ALTER TABLE pools ADD COLUMN pool_address VARCHAR(255);

-- Create index for efficient lookups
CREATE INDEX idx_pools_pool_address ON pools(pool_address);

-- Add comment explaining the column
COMMENT ON COLUMN pools.pool_address IS 'Blockchain address of the pool contract for transaction tracking';