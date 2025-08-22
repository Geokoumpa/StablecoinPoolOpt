CREATE TABLE approved_protocols (
    id SERIAL PRIMARY KEY,
    protocol_name VARCHAR(255) NOT NULL UNIQUE,
    added_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    removed_timestamp TIMESTAMP WITH TIME ZONE
);