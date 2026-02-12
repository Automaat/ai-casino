-- Add execution_graphs table for workflow execution tracking

CREATE TABLE IF NOT EXISTS execution_graphs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(100) NOT NULL UNIQUE,
    symbol VARCHAR(10),
    graph_jsonb JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_execution_graphs_workflow_id ON execution_graphs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_execution_graphs_symbol ON execution_graphs(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_execution_graphs_created_at ON execution_graphs(created_at DESC);
