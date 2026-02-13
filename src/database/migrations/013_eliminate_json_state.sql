-- Eliminate daemon-state.json - PostgreSQL migration

-- Metadata table for scalar state
CREATE TABLE daemon_metadata (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_daemon_metadata_updated_at ON daemon_metadata(updated_at);

-- Portfolio optimization history
CREATE TABLE optimization_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    symbols_optimized JSONB NOT NULL DEFAULT '[]'::jsonb,
    symbols_skipped JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_time_seconds DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_optimization_records_timestamp ON optimization_records(timestamp);
CREATE INDEX idx_optimization_records_created_at ON optimization_records(created_at);

-- Portfolio rebalancing history
CREATE TABLE rebalancing_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    method VARCHAR(50) NOT NULL,
    allocations JSONB NOT NULL,
    expected_return DECIMAL(8,4) NOT NULL,
    expected_volatility DECIMAL(8,4) NOT NULL,
    sharpe_ratio DECIMAL(8,4) NOT NULL,
    rebalances_executed INTEGER NOT NULL,
    rebalances_pending INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_rebalancing_records_timestamp ON rebalancing_records(timestamp);
CREATE INDEX idx_rebalancing_records_created_at ON rebalancing_records(created_at);

-- Sector rotation analysis history
CREATE TABLE sector_rotation_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    leading_sectors JSONB NOT NULL DEFAULT '[]'::jsonb,
    lagging_sectors JSONB NOT NULL DEFAULT '[]'::jsonb,
    sector_strengths JSONB NOT NULL DEFAULT '{}'::jsonb,
    sector_momenta JSONB NOT NULL DEFAULT '{}'::jsonb,
    flagged_positions JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sector_rotation_records_timestamp ON sector_rotation_records(timestamp);
CREATE INDEX idx_sector_rotation_records_created_at ON sector_rotation_records(created_at);

-- Peer analysis history
CREATE TABLE peer_analysis_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    symbols_analyzed JSONB NOT NULL DEFAULT '[]'::jsonb,
    rankings JSONB NOT NULL DEFAULT '{}'::jsonb,
    swap_recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_peers INTEGER NOT NULL,
    total_duration_seconds DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_peer_analysis_records_timestamp ON peer_analysis_records(timestamp);
CREATE INDEX idx_peer_analysis_records_created_at ON peer_analysis_records(created_at);

-- Correlation audit history
CREATE TABLE correlation_audit_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    num_positions INTEGER NOT NULL,
    num_correlated_pairs INTEGER NOT NULL,
    max_correlation DECIMAL(5,4) NOT NULL,
    avg_correlation DECIMAL(5,4) NOT NULL,
    diversification_ratio DECIMAL(8,4) NOT NULL,
    num_substitutions INTEGER NOT NULL,
    total_duration_seconds DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_correlation_audit_records_timestamp ON correlation_audit_records(timestamp);
CREATE INDEX idx_correlation_audit_records_created_at ON correlation_audit_records(created_at);

-- Risk report history
CREATE TABLE risk_report_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    var_95 DECIMAL(12,4) NOT NULL,
    var_99 DECIMAL(12,4) NOT NULL,
    cvar_95 DECIMAL(12,4) NOT NULL,
    cvar_99 DECIMAL(12,4) NOT NULL,
    cdar_95 DECIMAL(12,4) NOT NULL,
    max_drawdown DECIMAL(8,4) NOT NULL,
    risk_status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_risk_report_records_timestamp ON risk_report_records(timestamp);
CREATE INDEX idx_risk_report_records_created_at ON risk_report_records(created_at);

-- Monte Carlo stress test history
CREATE TABLE monte_carlo_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    simulation_method VARCHAR(50) NOT NULL,
    num_simulations INTEGER NOT NULL,
    horizon_days INTEGER NOT NULL,
    prob_loss_gt_threshold DECIMAL(5,4) NOT NULL,
    expected_worst_drawdown DECIMAL(8,4) NOT NULL,
    var_95 DECIMAL(12,4) NOT NULL,
    cvar_95 DECIMAL(12,4) NOT NULL,
    median_recovery_days DECIMAL(10,2),
    exceeds_risk_tolerance BOOLEAN NOT NULL,
    alert_message TEXT,
    portfolio_symbols JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_market_value DECIMAL(16,4) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_monte_carlo_records_timestamp ON monte_carlo_records(timestamp);
CREATE INDEX idx_monte_carlo_records_created_at ON monte_carlo_records(created_at);

-- Data prefetch history
CREATE TABLE prefetch_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    symbols_prefetched INTEGER NOT NULL,
    symbols_failed INTEGER NOT NULL,
    finbert_ready BOOLEAN NOT NULL,
    total_duration_seconds DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_prefetch_records_timestamp ON prefetch_records(timestamp);
CREATE INDEX idx_prefetch_records_created_at ON prefetch_records(created_at);

-- Screening history
CREATE TABLE screening_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    criteria VARCHAR(100) NOT NULL,
    universe VARCHAR(50) NOT NULL,
    top_symbols JSONB NOT NULL DEFAULT '[]'::jsonb,
    candidates JSONB NOT NULL DEFAULT '[]'::jsonb,
    screened_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_screening_records_timestamp ON screening_records(timestamp);
CREATE INDEX idx_screening_records_created_at ON screening_records(created_at);

-- Earnings calendar history
CREATE TABLE earnings_calendar_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    events JSONB NOT NULL DEFAULT '[]'::jsonb,
    symbols_fetched INTEGER NOT NULL,
    symbols_failed INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_earnings_calendar_records_timestamp ON earnings_calendar_records(timestamp);
CREATE INDEX idx_earnings_calendar_records_created_at ON earnings_calendar_records(created_at);

-- Profiling history
CREATE TABLE profiling_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cycle_number INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    duration_seconds DECIMAL(10,4) NOT NULL,
    profiling_overhead_percent DECIMAL(5,2) NOT NULL,
    top_function VARCHAR(200),
    top_function_cumtime DECIMAL(10,4),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_profiling_records_timestamp ON profiling_records(timestamp);
CREATE INDEX idx_profiling_records_created_at ON profiling_records(created_at);
CREATE INDEX idx_profiling_records_cycle_number ON profiling_records(cycle_number);

-- Game plan history
CREATE TABLE game_plan_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    priority_symbols JSONB NOT NULL DEFAULT '[]'::jsonb,
    risk_stance VARCHAR(20) NOT NULL,
    sector_focus JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_game_plan_records_timestamp ON game_plan_records(timestamp);
CREATE INDEX idx_game_plan_records_created_at ON game_plan_records(created_at);

-- Degradation history
CREATE TABLE degradation_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    tier VARCHAR(20) NOT NULL,
    unavailable_services JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence_adjustment DECIMAL(5,4) NOT NULL,
    halt_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_degradation_records_timestamp ON degradation_records(timestamp);
CREATE INDEX idx_degradation_records_created_at ON degradation_records(created_at);

-- Active discovery candidates (volatile state with TTL)
CREATE TABLE active_discovery_candidates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    discovered_at TIMESTAMP WITH TIME ZONE NOT NULL,
    composite_score DECIMAL(5,4) NOT NULL CHECK (composite_score >= 0 AND composite_score <= 1),
    sources JSONB NOT NULL,
    ttl_expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_active_discovery_candidates_symbol ON active_discovery_candidates(symbol);
CREATE INDEX idx_active_discovery_candidates_ttl_expires_at ON active_discovery_candidates(ttl_expires_at);
CREATE INDEX idx_active_discovery_candidates_created_at ON active_discovery_candidates(created_at);
