-- Add workflow_execution_metrics table for LLM execution tracking

CREATE TABLE workflow_execution_metrics (
    workflow_id UUID PRIMARY KEY,

    -- Core workflow data
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    total_latency_ms DECIMAL(12, 2) NOT NULL CHECK (total_latency_ms >= 0),

    -- LLM provider info
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,

    -- Token usage
    total_input_tokens INTEGER NOT NULL CHECK (total_input_tokens >= 0),
    total_output_tokens INTEGER NOT NULL CHECK (total_output_tokens >= 0),
    total_estimated_cost_usd DECIMAL(12, 6) NOT NULL CHECK (total_estimated_cost_usd >= 0),

    -- Nested execution details (JSONB)
    llm_calls JSONB NOT NULL DEFAULT '[]'::jsonb,
    sub_operations JSONB NOT NULL DEFAULT '[]'::jsonb,
    agent_timings JSONB NOT NULL DEFAULT '[]'::jsonb,
    pipeline_stages JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Audit
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Performance indices for common queries
CREATE INDEX idx_workflow_execution_metrics_symbol ON workflow_execution_metrics(symbol);
CREATE INDEX idx_workflow_execution_metrics_timestamp ON workflow_execution_metrics(timestamp DESC);
CREATE INDEX idx_workflow_execution_metrics_symbol_timestamp ON workflow_execution_metrics(symbol, timestamp DESC);
CREATE INDEX idx_workflow_execution_metrics_created_at ON workflow_execution_metrics(created_at DESC);

-- Rollback instructions:
-- DROP INDEX IF EXISTS idx_workflow_execution_metrics_created_at;
-- DROP INDEX IF EXISTS idx_workflow_execution_metrics_symbol_timestamp;
-- DROP INDEX IF EXISTS idx_workflow_execution_metrics_timestamp;
-- DROP INDEX IF EXISTS idx_workflow_execution_metrics_symbol;
-- DROP TABLE IF EXISTS workflow_execution_metrics;
