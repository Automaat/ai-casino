-- Migration: Add market_event_queue table
-- Description: PostgreSQL-backed FIFO queue for real-time market event signals

CREATE TABLE IF NOT EXISTS market_event_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(255) NOT NULL UNIQUE,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    enqueued_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    consumed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_market_event_queue_pending
    ON market_event_queue (enqueued_at ASC)
    WHERE consumed_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_market_event_queue_expires ON market_event_queue (expires_at);
