-- Add tearsheets table for QuantStats performance reports

CREATE TABLE tearsheets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    start_date TIMESTAMP WITH TIME ZONE NOT NULL,
    end_date TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Performance
    cagr DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    calmar_ratio DECIMAL(8,4),

    -- Risk
    max_drawdown DECIMAL(8,4),
    max_drawdown_duration_days INTEGER,
    volatility_annual DECIMAL(8,4),

    -- Win/loss
    win_rate DECIMAL(8,4),
    profit_factor DECIMAL(8,4),
    avg_win DECIMAL(12,4),
    avg_loss DECIMAL(12,4),

    -- Distribution
    best_day DECIMAL(8,4),
    worst_day DECIMAL(8,4),
    monthly_returns JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Benchmark
    benchmark_symbol VARCHAR(10),
    benchmark_cagr DECIMAL(8,4),
    benchmark_sharpe DECIMAL(8,4),
    alpha DECIMAL(8,4),
    beta DECIMAL(8,4),

    -- Files
    html_report_path TEXT NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tearsheets_symbol ON tearsheets(symbol);
CREATE INDEX idx_tearsheets_generated_at ON tearsheets(generated_at);
