-- Add token_address column to approved_tokens table for address-based filtering
-- This will store the on-chain Ethereum address for each approved token

ALTER TABLE approved_tokens ADD COLUMN token_address VARCHAR(42);

-- Add unique constraint to ensure no duplicate addresses
ALTER TABLE approved_tokens ADD CONSTRAINT uk_approved_tokens_token_address UNIQUE (token_address);

-- Add index for efficient lookups during filtering
CREATE INDEX idx_approved_tokens_token_address ON approved_tokens(token_address);

-- Add comment to explain the purpose
COMMENT ON COLUMN approved_tokens.token_address IS 'Ethereum mainnet address for the token (42-character hex string with 0x prefix)';