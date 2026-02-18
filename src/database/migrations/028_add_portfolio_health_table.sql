CREATE TABLE IF NOT EXISTS portfolio_health_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    total_positions INTEGER NOT NULL,
    portfolio_value DECIMAL(16, 4) NOT NULL,
    cash_percent DECIMAL(8, 4) NOT NULL,
    max_concentration_percent DECIMAL(8, 4) NOT NULL,
    max_concentration_symbol VARCHAR(20) NOT NULL,
    total_pnl_percent DECIMAL(8, 4) NOT NULL,
    biggest_drawdown_symbol VARCHAR(20),
    biggest_drawdown_percent DECIMAL(8, 4) NOT NULL,
    health_status VARCHAR(20) NOT NULL,
    recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
    constraints JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_health_timestamp ON portfolio_health_reports (timestamp);
CREATE INDEX IF NOT EXISTS idx_portfolio_health_status ON portfolio_health_reports (health_status);
