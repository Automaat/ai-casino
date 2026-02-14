-- Add risk_audit table for compliance tracking

CREATE TABLE IF NOT EXISTS risk_audit (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL', 'HOLD')),
    current_price DECIMAL(12, 4) NOT NULL CHECK (current_price > 0),

    -- Validation
    approved BOOLEAN NOT NULL,
    risk_level VARCHAR(10) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    risk_score DECIMAL(5, 4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),

    -- Position sizing
    recommended_shares INTEGER NOT NULL CHECK (recommended_shares >= 0),
    position_value DECIMAL(12, 2) NOT NULL CHECK (position_value >= 0),
    risk_amount DECIMAL(12, 2) NOT NULL CHECK (risk_amount >= 0),
    risk_percent DECIMAL(5, 4) NOT NULL CHECK (risk_percent >= 0),

    -- Stop loss
    stop_loss_price DECIMAL(12, 4) NOT NULL CHECK (stop_loss_price >= 0),

    -- Validation warnings (array)
    warnings TEXT[] NOT NULL DEFAULT '{}',

    -- Portfolio VaR (optional)
    portfolio_var_95 DECIMAL(5, 4) CHECK (portfolio_var_95 >= 0),
    portfolio_cvar_99 DECIMAL(5, 4) CHECK (portfolio_cvar_99 >= 0),
    portfolio_cdar_95 DECIMAL(5, 4) CHECK (portfolio_cdar_95 >= 0),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX idx_risk_audit_timestamp ON risk_audit (timestamp DESC);
CREATE INDEX idx_risk_audit_symbol ON risk_audit (symbol);
CREATE INDEX idx_risk_audit_symbol_timestamp ON risk_audit (symbol, timestamp DESC);

-- Analytics indexes
CREATE INDEX idx_risk_audit_approved ON risk_audit (approved);
CREATE INDEX idx_risk_audit_risk_level ON risk_audit (risk_level);

-- Partial index for violations
CREATE INDEX idx_risk_audit_violations ON risk_audit (symbol, timestamp DESC)
    WHERE approved = false;

-- Rollback instructions:
-- DROP INDEX IF EXISTS idx_risk_audit_violations;
-- DROP INDEX IF EXISTS idx_risk_audit_risk_level;
-- DROP INDEX IF EXISTS idx_risk_audit_approved;
-- DROP INDEX IF EXISTS idx_risk_audit_symbol_timestamp;
-- DROP INDEX IF EXISTS idx_risk_audit_symbol;
-- DROP INDEX IF EXISTS idx_risk_audit_timestamp;
-- DROP TABLE IF EXISTS risk_audit;
