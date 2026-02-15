-- Add sector_attribution table for portfolio sector analysis

CREATE TABLE sector_attribution (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    total_portfolio_value DECIMAL(16, 4) NOT NULL,
    benchmark_name VARCHAR(20) NOT NULL DEFAULT 'SPY',
    contributions JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for efficient queries
CREATE INDEX idx_sector_attribution_timestamp ON sector_attribution (timestamp DESC);
CREATE INDEX idx_sector_attribution_created_at ON sector_attribution (created_at);

-- Rollback instructions:
-- DROP INDEX IF EXISTS idx_sector_attribution_created_at;
-- DROP INDEX IF EXISTS idx_sector_attribution_timestamp;
-- DROP TABLE IF EXISTS sector_attribution;
