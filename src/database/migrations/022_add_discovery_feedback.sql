-- Extend discovery_history for supervisor evaluation and outcome tracking

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='discovery_history' AND column_name='supervisor_evaluation_score') THEN
        ALTER TABLE discovery_history ADD COLUMN supervisor_evaluation_score DECIMAL(5,4);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='discovery_history' AND column_name='supervisor_recommendation') THEN
        ALTER TABLE discovery_history ADD COLUMN supervisor_recommendation VARCHAR(20);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='discovery_history' AND column_name='evaluation_reasoning') THEN
        ALTER TABLE discovery_history ADD COLUMN evaluation_reasoning TEXT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='discovery_history' AND column_name='price_at_discovery') THEN
        ALTER TABLE discovery_history ADD COLUMN price_at_discovery DECIMAL(12,4);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='discovery_history' AND column_name='outcome_updated_at') THEN
        ALTER TABLE discovery_history ADD COLUMN outcome_updated_at TIMESTAMP WITH TIME ZONE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_discovery_history_outcome_tracking
ON discovery_history(discovered_at)
WHERE outcome_7d IS NULL OR outcome_30d IS NULL;

-- Discovery source metrics for tracking performance
CREATE TABLE IF NOT EXISTS discovery_source_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type VARCHAR(50) NOT NULL,
    measurement_date DATE NOT NULL,

    -- Volume metrics
    total_discoveries INT NOT NULL DEFAULT 0,
    watchlist_additions INT NOT NULL DEFAULT 0,
    signal_conversions INT NOT NULL DEFAULT 0,

    -- 7d/30d outcome metrics
    discoveries_with_7d_outcome INT NOT NULL DEFAULT 0,
    positive_7d_outcomes INT NOT NULL DEFAULT 0,
    avg_7d_return DECIMAL(8,4),
    median_7d_return DECIMAL(8,4),

    discoveries_with_30d_outcome INT NOT NULL DEFAULT 0,
    positive_30d_outcomes INT NOT NULL DEFAULT 0,
    avg_30d_return DECIMAL(8,4),
    median_30d_return DECIMAL(8,4),

    -- Quality metrics
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),

    false_positives INT NOT NULL DEFAULT 0,
    false_negatives INT NOT NULL DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_source_date UNIQUE(source_type, measurement_date)
);

CREATE INDEX IF NOT EXISTS idx_discovery_source_metrics_date ON discovery_source_metrics(measurement_date);
CREATE INDEX IF NOT EXISTS idx_discovery_source_metrics_source ON discovery_source_metrics(source_type);

-- Scoring weights history for adaptive learning (Phase 2b - future)
CREATE TABLE IF NOT EXISTS scoring_weights_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    regime VARCHAR(30),
    technical_weight DECIMAL(5,4) NOT NULL,
    liquidity_weight DECIMAL(5,4) NOT NULL,
    timing_weight DECIMAL(5,4) NOT NULL,
    social_weight DECIMAL(5,4) NOT NULL,
    volatility_weight DECIMAL(5,4) NOT NULL,
    training_window_days INT NOT NULL,
    discoveries_analyzed INT NOT NULL,
    avg_performance_improvement DECIMAL(8,4),
    is_active BOOLEAN NOT NULL DEFAULT false,
    activated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scoring_weights_history_active ON scoring_weights_history(is_active)
WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_scoring_weights_history_regime ON scoring_weights_history(regime);
