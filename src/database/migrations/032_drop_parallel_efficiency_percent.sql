-- Drop parallel_efficiency_percent column from supervisor_metrics
ALTER TABLE supervisor_metrics DROP COLUMN IF EXISTS parallel_efficiency_percent;
