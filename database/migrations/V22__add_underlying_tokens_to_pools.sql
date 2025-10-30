-- Add underlying_tokens and underlying_token_addresses columns to pools table
-- This will store the token symbols and addresses for each pool's underlying tokens

ALTER TABLE pools ADD COLUMN underlying_tokens TEXT[];
ALTER TABLE pools ADD COLUMN underlying_token_addresses TEXT[];

-- Add GIN index for efficient array operations on underlying_token_addresses
CREATE INDEX idx_pools_underlying_token_addresses ON pools USING GIN (underlying_token_addresses);

-- Add GIN index for efficient array operations on underlying_tokens
CREATE INDEX idx_pools_underlying_tokens ON pools USING GIN (underlying_tokens);

-- Add comments to explain the purpose
COMMENT ON COLUMN pools.underlying_tokens IS 'Array of token symbols for the pool (populated during filtering for approved pools only)';
COMMENT ON COLUMN pools.underlying_token_addresses IS 'Array of token addresses for the pool (populated during data ingestion from DeFiLlama API)';