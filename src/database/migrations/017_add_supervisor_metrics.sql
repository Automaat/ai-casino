-- Migration: Add supervisor_metrics table
-- Description: Store supervisor routing and worker execution metrics for observability

CREATE TABLE supervisor_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

    -- Identifiers
    workflow_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Routing decision
    required_analyses TEXT[] NOT NULL,
    optional_analyses TEXT[] NOT NULL,
    skip_analyses JSONB DEFAULT '{}'::JSONB NOT NULL,
    routing_reasoning TEXT NOT NULL,

    -- Execution metrics
    total_workers INTEGER NOT NULL,
    required_workers INTEGER NOT NULL,
    optional_workers INTEGER NOT NULL,
    successful_workers INTEGER NOT NULL,
    failed_workers INTEGER NOT NULL,

    -- Timing (milliseconds)
    routing_decision_ms DECIMAL(10, 2) NOT NULL,
    group1_execution_ms DECIMAL(10, 2) NOT NULL,
    research_execution_ms DECIMAL(10, 2) NOT NULL,
    total_supervisor_overhead_ms DECIMAL(10, 2) NOT NULL,

    -- Worker details
    worker_timings JSONB DEFAULT '{}'::JSONB NOT NULL,
    worker_errors JSONB DEFAULT '{}'::JSONB NOT NULL,

    -- LLM usage
    total_llm_calls INTEGER NOT NULL,
    total_cost_usd DECIMAL(10, 4) NOT NULL,
    planning_fallback_used BOOLEAN NOT NULL,
    synthesis_fallback_used BOOLEAN NOT NULL,

    -- Synthesis results
    confidence_adjustment DECIMAL(5, 4) NOT NULL,
    synthesis_reasoning TEXT NOT NULL,

    -- Efficiency
    parallel_efficiency_percent DECIMAL(5, 2) NOT NULL,
    timeout_triggered BOOLEAN NOT NULL,

    CONSTRAINT supervisor_metrics_workers_non_negative CHECK (
        total_workers >= 0 AND
        required_workers >= 0 AND
        optional_workers >= 0 AND
        successful_workers >= 0 AND
        failed_workers >= 0 AND
        total_llm_calls >= 0
    ),
    CONSTRAINT supervisor_metrics_timings_non_negative CHECK (
        routing_decision_ms >= 0 AND
        group1_execution_ms >= 0 AND
        research_execution_ms >= 0 AND
        total_supervisor_overhead_ms >= 0
    ),
    CONSTRAINT supervisor_metrics_efficiency_valid CHECK (
        parallel_efficiency_percent >= 0 AND
        parallel_efficiency_percent <= 100
    ),
    CONSTRAINT supervisor_metrics_cost_non_negative CHECK (
        total_cost_usd >= 0
    )
);

-- Indexes for common query patterns
CREATE INDEX idx_supervisor_metrics_symbol ON supervisor_metrics (symbol);
CREATE INDEX idx_supervisor_metrics_timestamp ON supervisor_metrics (timestamp);
CREATE INDEX idx_supervisor_metrics_workflow_id ON supervisor_metrics (workflow_id);
CREATE INDEX idx_supervisor_metrics_created_at ON supervisor_metrics (created_at);
