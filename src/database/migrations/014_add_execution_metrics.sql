-- Add execution_metrics table for order execution tracking

CREATE TABLE execution_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Core execution data
    order_id VARCHAR(100) NOT NULL UNIQUE,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(12, 4) NOT NULL CHECK (quantity > 0),
    requested_price DECIMAL(12, 4) NOT NULL CHECK (requested_price > 0),
    filled_price DECIMAL(12, 4) CHECK (filled_price > 0),

    -- Timing
    submitted_at TIMESTAMP WITH TIME ZONE NOT NULL,
    filled_at TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER,

    -- Slippage analysis
    slippage_bps DECIMAL(8, 2),

    -- Metadata
    broker VARCHAR(50) NOT NULL DEFAULT 'alpaca',
    venue VARCHAR(50),
    status VARCHAR(20) NOT NULL,

    -- Audit
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Performance indices for common queries
CREATE INDEX idx_execution_metrics_symbol ON execution_metrics(symbol);
CREATE INDEX idx_execution_metrics_submitted_at ON execution_metrics(submitted_at DESC);
CREATE INDEX idx_execution_metrics_broker ON execution_metrics(broker);
CREATE INDEX idx_execution_metrics_status ON execution_metrics(status);
CREATE INDEX idx_execution_metrics_symbol_broker ON execution_metrics(symbol, broker);

-- Rollback instructions:
-- DROP INDEX IF EXISTS idx_execution_metrics_symbol_broker;
-- DROP INDEX IF EXISTS idx_execution_metrics_status;
-- DROP INDEX IF EXISTS idx_execution_metrics_broker;
-- DROP INDEX IF EXISTS idx_execution_metrics_submitted_at;
-- DROP INDEX IF EXISTS idx_execution_metrics_symbol;
-- DROP TABLE IF EXISTS execution_metrics;
