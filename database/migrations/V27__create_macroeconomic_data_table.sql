-- Create macroeconomic_data table
CREATE TABLE macroeconomic_data (
    id SERIAL PRIMARY KEY,
    series_id VARCHAR(50) NOT NULL,  -- FRED series code (e.g., 'DGS1', 'SP500')
    series_name VARCHAR(255) NOT NULL,  -- Human-readable name
    frequency VARCHAR(20) NOT NULL,  -- 'daily' or 'monthly'
    date DATE NOT NULL,
    value DECIMAL(20, 8),  -- The actual data value
    unit VARCHAR(50),  -- Unit of measurement (e.g., 'Percent', 'Billions of Dollars')
    description TEXT,  -- Series description from FRED
    insertion_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(series_id, date)
);

-- Performance indexes
CREATE INDEX idx_macroeconomic_series_date ON macroeconomic_data(series_id, date);
CREATE INDEX idx_macroeconomic_frequency ON macroeconomic_data(frequency);