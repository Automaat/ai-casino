-- Migration: Add coordinator_metrics table
-- Description: Store coordinator decision cycle metrics for analytics

CREATE TABLE coordinator_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    cycle_num INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    symbols_analyzed TEXT[] DEFAULT '{}'::TEXT[] NOT NULL,
    tool_calls_made INTEGER NOT NULL,
    trades_proposed INTEGER NOT NULL,
    trades_executed INTEGER NOT NULL,
    trades_pending INTEGER NOT NULL,
    game_plan_generated BOOLEAN NOT NULL,
    cycle_duration_seconds DECIMAL(12, 4) NOT NULL,
    patterns_detected INTEGER NOT NULL,

    CONSTRAINT coordinator_metrics_counts_non_negative CHECK (
        tool_calls_made >= 0 AND
        trades_proposed >= 0 AND
        trades_executed >= 0 AND
        trades_pending >= 0 AND
        patterns_detected >= 0
    ),
    CONSTRAINT coordinator_metrics_duration_non_negative CHECK (
        cycle_duration_seconds >= 0
    )
);

-- Indexes for common query patterns
CREATE INDEX idx_coordinator_metrics_timestamp ON coordinator_metrics (timestamp DESC);
CREATE INDEX idx_coordinator_metrics_cycle_num ON coordinator_metrics (cycle_num);
CREATE INDEX idx_coordinator_metrics_cycle_timestamp ON coordinator_metrics (cycle_num, timestamp DESC);
CREATE INDEX idx_coordinator_metrics_game_plan ON coordinator_metrics (game_plan_generated);
CREATE INDEX idx_coordinator_metrics_trades_executed ON coordinator_metrics (trades_executed) WHERE trades_executed > 0;
