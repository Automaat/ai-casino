-- Add position_records table for tracking active positions

CREATE TABLE position_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    entry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL CHECK (entry_price > 0),
    entry_signal VARCHAR(10) NOT NULL,
    entry_confidence DECIMAL(5,4) NOT NULL CHECK (entry_confidence >= 0 AND entry_confidence <= 1),
    current_qty DECIMAL(12,4) NOT NULL CHECK (current_qty >= 0),
    current_stop_loss DECIMAL(12,4) NOT NULL CHECK (current_stop_loss > 0),
    initial_stop_loss DECIMAL(12,4) NOT NULL CHECK (initial_stop_loss > 0),
    stop_loss_order_id VARCHAR(100),
    profit_targets JSONB NOT NULL DEFAULT '[]'::jsonb,
    days_held INTEGER NOT NULL DEFAULT 0 CHECK (days_held >= 0),
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL,
    trailing_stop_activated BOOLEAN NOT NULL DEFAULT false,
    breakeven_activated BOOLEAN NOT NULL DEFAULT false,
    high_water_mark DECIMAL(12,4),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_position_records_symbol ON position_records(symbol);
CREATE INDEX idx_position_records_last_updated ON position_records(last_updated);
