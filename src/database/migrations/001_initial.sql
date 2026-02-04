-- Initial schema for AI Casino trade history

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    exit_price DECIMAL(12,4),
    shares INTEGER NOT NULL,
    stop_loss_price DECIMAL(12,4) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    risk_level VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    pnl DECIMAL(12,4),
    pnl_percent DECIMAL(8,4),
    strategy_name VARCHAR(50),
    broker_order_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_created_at ON trades(created_at);
CREATE INDEX idx_trades_status ON trades(status);

CREATE TABLE portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    balance DECIMAL(16,4) NOT NULL,
    available_cash DECIMAL(16,4) NOT NULL,
    total_exposure DECIMAL(16,4) NOT NULL,
    portfolio_value DECIMAL(16,4) NOT NULL,
    positions JSONB NOT NULL DEFAULT '{}',
    trigger VARCHAR(50) NOT NULL
);

CREATE INDEX idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp);
