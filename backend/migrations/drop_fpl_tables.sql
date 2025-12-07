-- Drop and recreate FPL data tables (preserving manager data)
-- This will apply the corrected BIGINT schema

-- Drop FPL data tables (in correct order due to foreign keys)
DROP TABLE IF EXISTS player_history CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS model_metrics CASCADE;
DROP TABLE IF EXISTS fixture_odds CASCADE;
DROP TABLE IF EXISTS fixtures CASCADE;
DROP TABLE IF EXISTS gameweeks CASCADE;
DROP TABLE IF EXISTS players CASCADE;
DROP TABLE IF EXISTS teams CASCADE;

-- Tables will be recreated by init_schema() with correct BIGINT types
SELECT 'FPL data tables dropped successfully' as status;
