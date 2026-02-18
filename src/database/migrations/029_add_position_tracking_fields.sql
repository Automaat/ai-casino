-- Add position tracking fields for conviction decay, R-multiple targets, and sell confirmation

ALTER TABLE position_records
    ADD COLUMN IF NOT EXISTS current_conviction DECIMAL(5,4),
    ADD COLUMN IF NOT EXISTS last_analysis_timestamp TIMESTAMP WITH TIME ZONE,
    ADD COLUMN IF NOT EXISTS conviction_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS initial_risk_per_share DECIMAL(12,4),
    ADD COLUMN IF NOT EXISTS r_multiple_targets_hit JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS pending_sell_signal_count INTEGER NOT NULL DEFAULT 0;
