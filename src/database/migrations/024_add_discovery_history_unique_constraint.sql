-- Add unique constraint to prevent duplicate discovery history records

CREATE UNIQUE INDEX idx_discovery_history_symbol_discovered_at
    ON discovery_history(symbol, discovered_at);
