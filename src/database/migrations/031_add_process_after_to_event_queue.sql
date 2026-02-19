ALTER TABLE market_event_queue
    ADD COLUMN IF NOT EXISTS process_after TIMESTAMP WITH TIME ZONE;

CREATE INDEX IF NOT EXISTS idx_market_event_queue_ready
    ON market_event_queue (enqueued_at, process_after)
    WHERE consumed_at IS NULL;
