-- Add missing fields to risk_report_records table

ALTER TABLE risk_report_records
ADD COLUMN IF NOT EXISTS portfolio_volatility DECIMAL(12, 4) NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS current_exposure_percent DECIMAL(5, 4) NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS num_positions INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS var_limit_breached BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS cvar_limit_breached BOOLEAN NOT NULL DEFAULT false;

-- Add index for breach queries
CREATE INDEX IF NOT EXISTS idx_risk_report_records_breaches
ON risk_report_records (timestamp DESC)
WHERE var_limit_breached = true OR cvar_limit_breached = true;

-- Rollback instructions:
-- DROP INDEX IF EXISTS idx_risk_report_records_breaches;
-- ALTER TABLE risk_report_records DROP COLUMN IF EXISTS cvar_limit_breached;
-- ALTER TABLE risk_report_records DROP COLUMN IF EXISTS var_limit_breached;
-- ALTER TABLE risk_report_records DROP COLUMN IF EXISTS num_positions;
-- ALTER TABLE risk_report_records DROP COLUMN IF EXISTS current_exposure_percent;
-- ALTER TABLE risk_report_records DROP COLUMN IF EXISTS portfolio_volatility;
