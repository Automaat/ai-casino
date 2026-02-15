-- Add health_reports, trade_journals, and paper_trading_reports tables
-- Also enhance peer_analysis_records table with full analysis data

-- Health reports table
CREATE TABLE health_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    overall_status VARCHAR(20) NOT NULL,
    service_checks JSONB NOT NULL DEFAULT '[]'::jsonb,
    cleanup_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_duration_ms DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_health_reports_timestamp ON health_reports (timestamp);
CREATE INDEX idx_health_reports_status ON health_reports (overall_status);
CREATE INDEX idx_health_reports_created_at ON health_reports (created_at);

-- Trade journals table
CREATE TABLE trade_journals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL UNIQUE,
    outcomes JSONB NOT NULL DEFAULT '[]'::jsonb,
    winners JSONB NOT NULL DEFAULT '[]'::jsonb,
    losers JSONB NOT NULL DEFAULT '[]'::jsonb,
    lessons JSONB NOT NULL DEFAULT '[]'::jsonb,
    tomorrows_focus JSONB NOT NULL DEFAULT '[]'::jsonb,
    overall_assessment TEXT NOT NULL,
    markdown_content TEXT,
    total_signals INTEGER NOT NULL,
    correct_signals INTEGER NOT NULL,
    accuracy_pct DECIMAL(5, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_trade_journals_date ON trade_journals (date);
CREATE INDEX idx_trade_journals_created_at ON trade_journals (created_at);

-- Paper trading reports table
CREATE TABLE paper_trading_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assessment_date TIMESTAMP WITH TIME ZONE NOT NULL,
    ready_for_live BOOLEAN NOT NULL,
    paper_trading_duration_days INTEGER NOT NULL,
    total_paper_trades INTEGER NOT NULL,
    criteria JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_pnl DECIMAL(16, 4) NOT NULL,
    sharpe_ratio DECIMAL(8, 4) NOT NULL,
    sortino_ratio DECIMAL(8, 4) NOT NULL,
    max_drawdown DECIMAL(8, 4) NOT NULL,
    win_rate DECIMAL(5, 4) NOT NULL,
    simulated_live JSONB,
    recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_paper_trading_reports_date ON paper_trading_reports (assessment_date);
CREATE INDEX idx_paper_trading_reports_ready ON paper_trading_reports (ready_for_live);
CREATE INDEX idx_paper_trading_reports_created_at ON paper_trading_reports (created_at);

-- Enhance peer_analysis_records table (add columns if they don't exist)
-- The analyses and total_peers columns should already be in the ORM model
ALTER TABLE peer_analysis_records
    ADD COLUMN IF NOT EXISTS analyses JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS total_peers INTEGER NOT NULL DEFAULT 0;

-- Rollback instructions:
-- ALTER TABLE peer_analysis_records DROP COLUMN IF EXISTS total_peers;
-- ALTER TABLE peer_analysis_records DROP COLUMN IF EXISTS analyses;
-- DROP INDEX IF EXISTS idx_paper_trading_reports_created_at;
-- DROP INDEX IF EXISTS idx_paper_trading_reports_ready;
-- DROP INDEX IF EXISTS idx_paper_trading_reports_date;
-- DROP TABLE IF EXISTS paper_trading_reports;
-- DROP INDEX IF EXISTS idx_trade_journals_created_at;
-- DROP INDEX IF EXISTS idx_trade_journals_date;
-- DROP TABLE IF EXISTS trade_journals;
-- DROP INDEX IF EXISTS idx_health_reports_created_at;
-- DROP INDEX IF EXISTS idx_health_reports_status;
-- DROP INDEX IF EXISTS idx_health_reports_timestamp;
-- DROP TABLE IF EXISTS health_reports;
