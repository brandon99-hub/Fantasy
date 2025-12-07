"""
Sync FPL data from SQLite to PostgreSQL

This script copies all FPL data (teams, players, gameweeks, fixtures, player_history)
from the existing SQLite database to the new PostgreSQL database.
Handles SQLite boolean (0/1) to PostgreSQL boolean conversion and integer overflow.
"""

import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.src.core.database import FPLDatabase
from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.core.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# PostgreSQL INTEGER max value (32-bit signed)
PG_INT_MAX = 2147483647
PG_INT_MIN = -2147483648


def convert_sqlite_booleans(df: pd.DataFrame, boolean_columns: list) -> pd.DataFrame:
    """
    Convert SQLite integer booleans (0/1) to proper Python booleans for PostgreSQL
    
    Args:
        df: DataFrame to convert
        boolean_columns: List of column names that should be boolean
    
    Returns:
        DataFrame with converted boolean columns
    """
    df = df.copy()
    for col in boolean_columns:
        if col in df.columns:
            # Convert to boolean, handling None/NaN
            df[col] = df[col].apply(lambda x: bool(x) if pd.notna(x) and x is not None else None)
    return df


def convert_large_integers(df: pd.DataFrame, int_columns: list) -> pd.DataFrame:
    """
    Convert integers that exceed PostgreSQL INTEGER range to NULL
    
    Args:
        df: DataFrame to convert
        int_columns: List of column names that are integers
    
    Returns:
        DataFrame with converted integer columns
    """
    df = df.copy()
    for col in int_columns:
        if col in df.columns:
            # Convert values outside PostgreSQL INTEGER range to NULL
            df[col] = df[col].apply(
                lambda x: None if pd.notna(x) and (x > PG_INT_MAX or x < PG_INT_MIN) else x
            )
    return df


def filter_columns(df: pd.DataFrame, allowed_columns: list) -> pd.DataFrame:
    """
    Filter dataframe to only include columns that exist in PostgreSQL schema
    
    Args:
        df: DataFrame to filter
        allowed_columns: List of column names allowed in PostgreSQL
    
    Returns:
        DataFrame with only allowed columns
    """
    existing_cols = [col for col in allowed_columns if col in df.columns]
    return df[existing_cols]


def sync_sqlite_to_postgres():
    """Sync all FPL data from SQLite to PostgreSQL"""
    
    logger.info("🚀 Starting SQLite to PostgreSQL sync...")
    
    # Initialize databases
    sqlite_db = FPLDatabase()
    postgres_db = PostgresManagerDB()
    
    # Test connections
    logger.info("Testing database connections...")
    if not sqlite_db.test_connection():
        logger.error("❌ SQLite connection failed")
        return False
    
    if not postgres_db.test_connection():
        logger.error("❌ PostgreSQL connection failed")
        return False
    
    logger.info("✅ Database connections successful")
    
    # Initialize PostgreSQL schema
    logger.info("Initializing PostgreSQL schema...")
    if not postgres_db.initialize_schema():
        logger.error("❌ Schema initialization failed")
        return False
    logger.info("✅ Schema initialized")
    
    try:
        # Sync teams - get ALL columns from SQLite
        logger.info("📊 Syncing teams...")
        with sqlite3.connect(settings.DATABASE_PATH) as conn:
            teams_df = pd.read_sql_query("SELECT * FROM teams", conn)
        
        # Convert boolean columns
        teams_bool_cols = ['unavailable']
        teams_df = convert_sqlite_booleans(teams_df, teams_bool_cols)
        
        # Filter to PostgreSQL schema columns
        teams_allowed_cols = ['id', 'name', 'short_name', 'strength', 'strength_overall_home',
                              'strength_overall_away', 'strength_attack_home', 'strength_attack_away',
                              'strength_defence_home', 'strength_defence_away', 'pulse_id']
        teams_df = filter_columns(teams_df, teams_allowed_cols)
        
        if teams_df is not None and len(teams_df) > 0:
            count = postgres_db.upsert_teams(teams_df)
            logger.info(f"✅ Synced {count} teams")
        else:
            logger.warning("⚠️  No teams found in SQLite")
        
        # Sync gameweeks
        logger.info("📅 Syncing gameweeks...")
        with sqlite3.connect(settings.DATABASE_PATH) as conn:
            gameweeks_df = pd.read_sql_query("SELECT * FROM gameweeks", conn)
        
        # Convert boolean columns
        gameweeks_bool_cols = [
            'finished', 'is_previous', 'is_current', 'is_next', 'data_checked',
            'cup_leagues_created', 'h2h_ko_matches_created', 'can_enter', 
            'can_manage', 'released'
        ]
        gameweeks_df = convert_sqlite_booleans(gameweeks_df, gameweeks_bool_cols)
        
        # Filter to ONLY columns that exist in PostgreSQL schema
        gameweeks_allowed_cols = ['id', 'name', 'deadline_time', 'finished', 'is_current',
                                  'is_next', 'is_previous', 'average_entry_score', 'highest_score',
                                  'most_selected', 'most_transferred_in', 'most_captained',
                                  'most_vice_captained', 'top_element']
        gameweeks_df = filter_columns(gameweeks_df, gameweeks_allowed_cols)
        
        # Convert pandas int64 to Python int for PostgreSQL compatibility
        int_cols = ['id', 'average_entry_score', 'highest_score', 'most_selected',
                   'most_transferred_in', 'most_captained', 'most_vice_captained', 'top_element']
        for col in int_cols:
            if col in gameweeks_df.columns:
                gameweeks_df[col] = gameweeks_df[col].apply(
                    lambda x: int(x) if pd.notna(x) and x is not None else None
                )
        
        # Convert deadline_time to proper timestamp string format
        if 'deadline_time' in gameweeks_df.columns:
            gameweeks_df['deadline_time'] = gameweeks_df['deadline_time'].apply(
                lambda x: str(x) if pd.notna(x) and x is not None else None
            )
        
        if gameweeks_df is not None and len(gameweeks_df) > 0:
            count = postgres_db.upsert_gameweeks(gameweeks_df)
            logger.info(f"✅ Synced {count} gameweeks")
        else:
            logger.warning("⚠️  No gameweeks found in SQLite")
        
        # Sync players
        logger.info("👥 Syncing players...")
        with sqlite3.connect(settings.DATABASE_PATH) as conn:
            all_players_df = pd.read_sql_query("SELECT * FROM players", conn)
        
        # Convert boolean columns
        players_bool_cols = [
            'can_transact', 'can_select', 'in_dreamteam', 'removed', 
            'special', 'has_temporary_code'
        ]
        all_players_df = convert_sqlite_booleans(all_players_df, players_bool_cols)
        
        # Filter to PostgreSQL schema columns (keep original SQLite column names)
        players_allowed_cols = ['id', 'web_name', 'first_name', 'second_name', 'team',
                               'element_type', 'now_cost', 'cost_change_start', 'cost_change_event',
                               'total_points', 'points_per_game', 'selected_by_percent', 'form',
                               'transfers_in', 'transfers_out', 'transfers_in_event', 'transfers_out_event',
                               'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
                               'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
                               'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat',
                               'ict_index', 'expected_goals', 'expected_assists', 'expected_goal_involvements',
                               'expected_goals_conceded', 'expected_goals_per_90', 'expected_assists_per_90',
                               'xg_source', 'starts', 'news', 'news_added', 'chance_of_playing_this_round',
                               'chance_of_playing_next_round', 'status']
        all_players_df = filter_columns(all_players_df, players_allowed_cols)
        
        # Convert pandas int64 to Python int for PostgreSQL compatibility
        int_cols = ['id', 'team', 'element_type', 'now_cost', 'cost_change_start', 'cost_change_event',
                   'total_points', 'transfers_in', 'transfers_out', 'transfers_in_event', 'transfers_out_event',
                   'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
                   'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
                   'red_cards', 'saves', 'bonus', 'bps', 'starts',
                   'chance_of_playing_this_round', 'chance_of_playing_next_round']
        for col in int_cols:
            if col in all_players_df.columns:
                all_players_df[col] = all_players_df[col].apply(
                    lambda x: int(x) if pd.notna(x) else None
                )
        
        if all_players_df is not None and len(all_players_df) > 0:
            count = postgres_db.upsert_players(all_players_df)
            logger.info(f"✅ Synced {count} players")
        else:
            logger.warning("⚠️  No players found in SQLite")
        
        # Sync fixtures
        logger.info("⚽ Syncing fixtures...")
        with sqlite3.connect(settings.DATABASE_PATH) as conn:
            fixtures_df = pd.read_sql_query("SELECT * FROM fixtures", conn)
        
        # Convert boolean columns
        fixtures_bool_cols = [
            'provisional_start_time', 'started', 'finished', 'finished_provisional'
        ]
        fixtures_df = convert_sqlite_booleans(fixtures_df, fixtures_bool_cols)
        
        # Filter to PostgreSQL schema columns
        fixtures_allowed_cols = ['id', 'event', 'team_h', 'team_a', 'team_h_score', 'team_a_score',
                                'kickoff_time', 'finished', 'finished_provisional', 'started',
                                'team_h_difficulty', 'team_a_difficulty', 'pulse_id']
        fixtures_df = filter_columns(fixtures_df, fixtures_allowed_cols)
        
        if fixtures_df is not None and len(fixtures_df) > 0:
            count = postgres_db.upsert_fixtures(fixtures_df)
            logger.info(f"✅ Synced {count} fixtures")
        else:
            logger.warning("⚠️  No fixtures found in SQLite")
        
        # Sync player history
        logger.info("📈 Syncing player history...")
        with sqlite3.connect(settings.DATABASE_PATH) as conn:
            history_df = pd.read_sql_query("SELECT * FROM player_history", conn)
        
        # Convert boolean columns
        history_bool_cols = ['was_home']
        history_df = convert_sqlite_booleans(history_df, history_bool_cols)
        
        # Filter to PostgreSQL schema columns (keep original SQLite column names)
        history_allowed_cols = ['element', 'round', 'fixture', 'opponent_team', 'total_points',
                               'was_home', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
                               'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed',
                               'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps', 'influence',
                               'creativity', 'threat', 'ict_index', 'value', 'transfers_balance',
                               'selected', 'transfers_in', 'transfers_out', 'expected_goals',
                               'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']
        history_df = filter_columns(history_df, history_allowed_cols)
        
        if history_df is not None and len(history_df) > 0:
            count = postgres_db.upsert_player_history(history_df)
            logger.info(f"✅ Synced {count} player history records")
        else:
            logger.warning("⚠️  No player history found in SQLite")
        
        logger.info("🎉 Sync completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sync failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = sync_sqlite_to_postgres()
    sys.exit(0 if success else 1)
