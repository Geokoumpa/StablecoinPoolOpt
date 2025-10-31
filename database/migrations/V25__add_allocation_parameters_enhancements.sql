-- Add new fields to allocation_parameters table for enhanced tracking
ALTER TABLE allocation_parameters
ADD COLUMN projected_apy DECIMAL(20, 4),
ADD COLUMN transaction_costs DECIMAL(20, 2),
ADD COLUMN transaction_sequence JSONB;

-- Add comments to describe the new fields
COMMENT ON COLUMN allocation_parameters.projected_apy IS 'The projected APY for the allocation run';
COMMENT ON COLUMN allocation_parameters.transaction_costs IS 'Total transaction costs in USD for the allocation';
COMMENT ON COLUMN allocation_parameters.transaction_sequence IS 'JSON array describing the sequence of transactions to execute';