CREATE TABLE IF NOT EXISTS economic_calendar_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    risk_level VARCHAR(10) NOT NULL,
    recommendation VARCHAR(30) NOT NULL,
    reason TEXT NOT NULL,
    upcoming_events JSONB NOT NULL DEFAULT '[]',
    avoid_until TIMESTAMP WITH TIME ZONE,
    computed_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX idx_economic_calendar_signals_computed_at
    ON economic_calendar_signals (computed_at DESC);
