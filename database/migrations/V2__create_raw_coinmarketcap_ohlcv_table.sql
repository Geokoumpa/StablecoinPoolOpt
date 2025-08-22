CREATE TABLE IF NOT EXISTS raw_coinmarketcap_ohlcv (
    id SERIAL PRIMARY KEY,
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    symbol VARCHAR(255) NOT NULL,
    raw_json_data JSONB,
    insertion_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (data_timestamp, symbol)
);