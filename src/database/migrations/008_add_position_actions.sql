-- Add position_management_actions table for tracking position adjustments

CREATE TABLE position_management_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    old_stop_loss DECIMAL(12,4),
    new_stop_loss DECIMAL(12,4),
    qty_sold DECIMAL(12,4),
    price DECIMAL(12,4) NOT NULL CHECK (price > 0),
    reason TEXT NOT NULL,
    executed BOOLEAN NOT NULL,
    order_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_position_actions_symbol ON position_management_actions(symbol);
CREATE INDEX idx_position_actions_timestamp ON position_management_actions(timestamp);
CREATE INDEX idx_position_actions_symbol_timestamp ON position_management_actions(symbol, timestamp DESC);
CREATE INDEX idx_position_actions_created_at ON position_management_actions(created_at);
