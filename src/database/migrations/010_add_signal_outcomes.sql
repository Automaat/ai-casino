-- Add signal_outcomes table for persistent learning from trading decisions

CREATE TABLE signal_outcomes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Core signal data
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    signal VARCHAR(10) NOT NULL CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    price_at_signal DECIMAL(12,4) NOT NULL CHECK (price_at_signal > 0),

    -- Context
    strategy_used VARCHAR(50),
    regime VARCHAR(30),
    trading_session VARCHAR(20) NOT NULL DEFAULT 'REGULAR',

    -- Component signals
    technical_signal VARCHAR(10),
    sentiment_signal VARCHAR(10),
    news_signal VARCHAR(10),

    -- Outcome prices (populated by SignalOutcomeTracker)
    price_at_1d DECIMAL(12,4) CHECK (price_at_1d > 0),
    price_at_5d DECIMAL(12,4) CHECK (price_at_5d > 0),
    price_at_20d DECIMAL(12,4) CHECK (price_at_20d > 0),
    actual_exit_price DECIMAL(12,4),
    actual_exit_date TIMESTAMP WITH TIME ZONE,

    -- Metadata
    outcome_updated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_signal_per_symbol_timestamp UNIQUE(symbol, timestamp)
);

-- Performance indexes
CREATE INDEX idx_signal_outcomes_symbol ON signal_outcomes(symbol);
CREATE INDEX idx_signal_outcomes_timestamp ON signal_outcomes(timestamp DESC);
CREATE INDEX idx_signal_outcomes_regime ON signal_outcomes(regime) WHERE regime IS NOT NULL;
CREATE INDEX idx_signal_outcomes_regime_signal ON signal_outcomes(regime, signal) WHERE regime IS NOT NULL;
CREATE INDEX idx_signal_outcomes_needs_update_1d ON signal_outcomes(timestamp) WHERE price_at_1d IS NULL;
CREATE INDEX idx_signal_outcomes_needs_update_5d ON signal_outcomes(timestamp) WHERE price_at_5d IS NULL;
CREATE INDEX idx_signal_outcomes_needs_update_20d ON signal_outcomes(timestamp) WHERE price_at_20d IS NULL;
