-- Add is_paper_trade column for paper/live filtering
ALTER TABLE trades ADD COLUMN is_paper_trade BOOLEAN NOT NULL DEFAULT TRUE;
CREATE INDEX idx_trades_is_paper_trade ON trades(is_paper_trade);

-- Add closed_at timestamp for date-scoped validation
ALTER TABLE trades ADD COLUMN closed_at TIMESTAMP WITH TIME ZONE;
CREATE INDEX idx_trades_closed_at ON trades(closed_at);
