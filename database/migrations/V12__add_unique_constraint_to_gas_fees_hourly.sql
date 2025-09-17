ALTER TABLE gas_fees_hourly
ADD CONSTRAINT unique_timestamp UNIQUE (timestamp);