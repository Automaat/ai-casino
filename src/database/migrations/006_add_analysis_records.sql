-- Add analysis_records table for tracking all workflow analysis results

CREATE TABLE analysis_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    signal VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    executed_trade BOOLEAN NOT NULL DEFAULT false,
    trading_session VARCHAR(20) NOT NULL DEFAULT 'REGULAR',
    is_paper_trade BOOLEAN NOT NULL DEFAULT true,
    rsi DECIMAL(6,2),
    macd_hist DECIMAL(10,4),
    reasoning JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_analysis_records_symbol ON analysis_records(symbol);
CREATE INDEX idx_analysis_records_timestamp ON analysis_records(timestamp);
CREATE INDEX idx_analysis_records_symbol_timestamp ON analysis_records(symbol, timestamp DESC);
CREATE INDEX idx_analysis_records_signal ON analysis_records(signal);
CREATE INDEX idx_analysis_records_created_at ON analysis_records(created_at);
