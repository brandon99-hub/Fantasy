-- Migration: Fix ALL integer overflow errors
-- Date: 2025-12-06
-- Description: Change ALL INTEGER columns to BIGINT except IDs and small enums

-- Players table: Change ALL numeric columns to BIGINT (except element_type which is 1-4)
ALTER TABLE players 
    ALTER COLUMN now_cost TYPE BIGINT,
    ALTER COLUMN cost_change_start TYPE BIGINT,
    ALTER COLUMN cost_change_event TYPE BIGINT,
    ALTER COLUMN total_points TYPE BIGINT,
    ALTER COLUMN minutes TYPE BIGINT,
    ALTER COLUMN goals_scored TYPE BIGINT,
    ALTER COLUMN assists TYPE BIGINT,
    ALTER COLUMN clean_sheets TYPE BIGINT,
    ALTER COLUMN goals_conceded TYPE BIGINT,
    ALTER COLUMN own_goals TYPE BIGINT,
    ALTER COLUMN penalties_saved TYPE BIGINT,
    ALTER COLUMN penalties_missed TYPE BIGINT,
    ALTER COLUMN yellow_cards TYPE BIGINT,
    ALTER COLUMN red_cards TYPE BIGINT,
    ALTER COLUMN saves TYPE BIGINT,
    ALTER COLUMN bonus TYPE BIGINT,
    ALTER COLUMN bps TYPE BIGINT,
    ALTER COLUMN starts TYPE BIGINT;

-- Gameweeks table: Change statistics columns to BIGINT  
ALTER TABLE gameweeks
    ALTER COLUMN average_entry_score TYPE BIGINT,
    ALTER COLUMN highest_score TYPE BIGINT;

-- Verify changes
SELECT 'Players columns updated' as status;
SELECT 'Gameweeks columns updated' as status;
