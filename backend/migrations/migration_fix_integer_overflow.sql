-- Migration: Fix integer overflow errors
-- Date: 2025-12-06
-- Description: Change INTEGER columns to BIGINT for fields that receive large values from FPL API

-- Players table: Change transfer columns to BIGINT
ALTER TABLE players 
    ALTER COLUMN transfers_in TYPE BIGINT,
    ALTER COLUMN transfers_out TYPE BIGINT,
    ALTER COLUMN transfers_in_event TYPE BIGINT,
    ALTER COLUMN transfers_out_event TYPE BIGINT;

-- Gameweeks table: Change statistics columns to BIGINT
ALTER TABLE gameweeks
    ALTER COLUMN most_selected TYPE BIGINT,
    ALTER COLUMN most_transferred_in TYPE BIGINT,
    ALTER COLUMN most_captained TYPE BIGINT,
    ALTER COLUMN most_vice_captained TYPE BIGINT;

-- Verify changes
SELECT 
    column_name, 
    data_type 
FROM information_schema.columns 
WHERE table_name = 'players' 
    AND column_name IN ('transfers_in', 'transfers_out', 'transfers_in_event', 'transfers_out_event')
UNION ALL
SELECT 
    column_name, 
    data_type 
FROM information_schema.columns 
WHERE table_name = 'gameweeks' 
    AND column_name IN ('most_selected', 'most_transferred_in', 'most_captained', 'most_vice_captained');
