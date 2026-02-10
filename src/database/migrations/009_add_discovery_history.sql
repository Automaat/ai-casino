-- Add discovery_history table for tracking stock discovery outcomes

CREATE TABLE discovery_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    discovered_at TIMESTAMP WITH TIME ZONE NOT NULL,
    composite_score DECIMAL(5,4) NOT NULL CHECK (composite_score >= 0 AND composite_score <= 1),
    sources JSONB NOT NULL,
    added_to_watchlist BOOLEAN NOT NULL,
    ttl_expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    first_signal VARCHAR(10),
    first_signal_date TIMESTAMP WITH TIME ZONE,
    outcome_7d DECIMAL(8,4),
    outcome_30d DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_discovery_history_symbol ON discovery_history(symbol);
CREATE INDEX idx_discovery_history_discovered_at ON discovery_history(discovered_at);
CREATE INDEX idx_discovery_history_ttl_expires_at ON discovery_history(ttl_expires_at);
CREATE INDEX idx_discovery_history_created_at ON discovery_history(created_at);
