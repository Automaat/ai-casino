-- Add missing fields to game_plan_records for full game plan storage
-- Migration: 023_add_game_plan_fields

ALTER TABLE game_plan_records
ADD COLUMN IF NOT EXISTS reasoning TEXT,
ADD COLUMN IF NOT EXISTS confidence DECIMAL(5, 4),
ADD COLUMN IF NOT EXISTS overnight_summary TEXT,
ADD COLUMN IF NOT EXISTS key_levels JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS generated_at TIMESTAMP WITH TIME ZONE;

-- Backfill generated_at from timestamp for existing records
UPDATE game_plan_records
SET generated_at = timestamp
WHERE generated_at IS NULL;

-- Make generated_at non-null after backfill
ALTER TABLE game_plan_records
ALTER COLUMN generated_at SET NOT NULL;

-- Add index on generated_at
CREATE INDEX IF NOT EXISTS idx_game_plan_records_generated_at ON game_plan_records(generated_at DESC);
