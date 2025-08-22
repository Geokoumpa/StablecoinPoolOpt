CREATE TABLE IF NOT EXISTS raw_defillama_pool_history (
    id SERIAL PRIMARY KEY,
    pool_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    raw_json_data JSONB
);