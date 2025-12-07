import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
import logging
from pathlib import Path

from backend.src.core.config import get_settings

settings = get_settings()

class FPLDatabase:
    """Handles SQLite database operations for FPL data"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _prepare_dataframe_for_insert(self, conn, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Align dataframe columns with target table schema (excluding primary key)."""
        table_info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        valid_cols = [row[1] for row in table_info if row[1] != 'id']
        aligned = df.copy()
        for col in valid_cols:
            if col not in aligned.columns:
                aligned[col] = None
        aligned = aligned[valid_cols]
        return aligned

    def _delete_conflicting_rows(self, conn, table_name: str, unique_cols: List[str], df: pd.DataFrame) -> None:
        """Remove existing rows that would conflict with new inserts."""
        if any(col not in df.columns for col in unique_cols):
            return
        keys_df = df[unique_cols].dropna()
        if keys_df.empty:
            return
        placeholders = " AND ".join(f"{col}=?" for col in unique_cols)
        query = f"DELETE FROM {table_name} WHERE {placeholders}"
        params = [tuple(row) for row in keys_df.drop_duplicates().itertuples(index=False, name=None)]
        if not params:
            return
        conn.executemany(query, params)
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Teams table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    short_name TEXT,
                    code INTEGER,
                    draw INTEGER,
                    loss INTEGER,
                    played INTEGER,
                    points INTEGER,
                    position INTEGER,
                    team_division INTEGER,
                    unavailable BOOLEAN,
                    win INTEGER,
                    pulse_id INTEGER,
                    form TEXT,
                    strength INTEGER,
                    strength_overall_home INTEGER,
                    strength_overall_away INTEGER,
                    strength_attack_home INTEGER,
                    strength_attack_away INTEGER,
                    strength_defence_home INTEGER,
                    strength_defence_away INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Players table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY,
                    code INTEGER,
                    web_name TEXT NOT NULL,
                    first_name TEXT,
                    second_name TEXT,
                    team INTEGER,
                    team_name TEXT,
                    team_code INTEGER,
                    position TEXT,
                    element_type INTEGER,
                    now_cost INTEGER,
                    total_points INTEGER,
                    points_per_game REAL,
                    form REAL,
                    selected_by_percent REAL,
                    transfers_in INTEGER,
                    transfers_in_event INTEGER,
                    transfers_out INTEGER,
                    transfers_out_event INTEGER,
                    cost_change_event INTEGER,
                    cost_change_event_fall INTEGER,
                    cost_change_start INTEGER,
                    cost_change_start_fall INTEGER,
                    value_form REAL,
                    value_season REAL,
                    minutes INTEGER,
                    goals_scored INTEGER,
                    assists INTEGER,
                    clean_sheets INTEGER,
                    clean_sheets_per_90 REAL,
                    goals_conceded INTEGER,
                    goals_conceded_per_90 REAL,
                    own_goals INTEGER,
                    penalties_saved INTEGER,
                    penalties_missed INTEGER,
                    yellow_cards INTEGER,
                    red_cards INTEGER,
                    saves INTEGER,
                    saves_per_90 REAL,
                    bonus INTEGER,
                    bps INTEGER,
                    influence REAL,
                    influence_rank INTEGER,
                    influence_rank_type INTEGER,
                    creativity REAL,
                    creativity_rank INTEGER,
                    creativity_rank_type INTEGER,
                    threat REAL,
                    threat_rank INTEGER,
                    threat_rank_type INTEGER,
                    ict_index REAL,
                    ict_index_rank INTEGER,
                    ict_index_rank_type INTEGER,
                    form_rank INTEGER,
                    form_rank_type INTEGER,
                    points_per_game_rank INTEGER,
                    points_per_game_rank_type INTEGER,
                    selected_rank INTEGER,
                    selected_rank_type INTEGER,
                    now_cost_rank INTEGER,
                    now_cost_rank_type INTEGER,
                    news TEXT,
                    news_added TIMESTAMP,
                    chance_of_playing_this_round INTEGER,
                    chance_of_playing_next_round INTEGER,
                    status TEXT,
                    can_transact BOOLEAN DEFAULT 1,
                    can_select BOOLEAN DEFAULT 1,
                    dreamteam_count INTEGER,
                    ep_next REAL,
                    ep_this REAL,
                    event_points INTEGER,
                    in_dreamteam BOOLEAN,
                    photo TEXT,
                    removed BOOLEAN,
                    special BOOLEAN,
                    squad_number INTEGER,
                    region TEXT,
                    birth_date TEXT,
                    opta_code TEXT,
                    has_temporary_code BOOLEAN,
                    team_join_date TEXT,
                    expected_goals REAL,
                    expected_goals_per_90 REAL,
                    expected_assists REAL,
                    expected_assists_per_90 REAL,
                    expected_goal_involvements REAL,
                    expected_goal_involvements_per_90 REAL,
                    expected_goals_conceded REAL,
                    expected_goals_conceded_per_90 REAL,
                    starts INTEGER,
                    starts_per_90 REAL,
                    clearances_blocks_interceptions INTEGER,
                    recoveries INTEGER,
                    tackles INTEGER,
                    defensive_contribution INTEGER,
                    defensive_contribution_per_90 REAL,
                    corners_and_indirect_freekicks_order INTEGER,
                    corners_and_indirect_freekicks_text TEXT,
                    direct_freekicks_order INTEGER,
                    direct_freekicks_text TEXT,
                    penalties_order INTEGER,
                    penalties_text TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team) REFERENCES teams (id)
                )
            """)
            
            # Gameweeks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gameweeks (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    deadline_time TEXT,
                    deadline_time_epoch INTEGER,
                    deadline_time_game_offset INTEGER,
                    release_time TEXT,
                    finished BOOLEAN,
                    is_previous BOOLEAN,
                    is_current BOOLEAN,
                    is_next BOOLEAN,
                    chip_plays TEXT,
                    most_selected INTEGER,
                    most_transferred_in INTEGER,
                    most_captained INTEGER,
                    most_vice_captained INTEGER,
                    top_element INTEGER,
                    top_element_info TEXT,
                    average_entry_score INTEGER,
                    data_checked BOOLEAN,
                    highest_scoring_entry INTEGER,
                    highest_score INTEGER,
                    cup_leagues_created BOOLEAN,
                    h2h_ko_matches_created BOOLEAN,
                    can_enter BOOLEAN,
                    can_manage BOOLEAN,
                    released BOOLEAN,
                    ranked_count INTEGER,
                    transfers_made INTEGER,
                    overrides TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Fixtures table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fixtures (
                    id INTEGER PRIMARY KEY,
                    code INTEGER,
                    event INTEGER,
                    team_h INTEGER,
                    team_a INTEGER,
                    team_h_difficulty INTEGER,
                    team_a_difficulty INTEGER,
                    kickoff_time TEXT,
                    provisional_start_time BOOLEAN,
                    started BOOLEAN,
                    finished BOOLEAN,
                    finished_provisional BOOLEAN,
                    minutes INTEGER,
                    team_h_score INTEGER,
                    team_a_score INTEGER,
                    stats TEXT,
                    pulse_id INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_h) REFERENCES teams (id),
                    FOREIGN KEY (team_a) REFERENCES teams (id),
                    FOREIGN KEY (event) REFERENCES gameweeks (id)
                )
            """)
            
            # Fixture odds table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fixture_odds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER UNIQUE,
                    event INTEGER,
                    home_team INTEGER,
                    away_team INTEGER,
                    bookmaker TEXT,
                    market TEXT,
                    home_implied_goals REAL,
                    away_implied_goals REAL,
                    raw_odds TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (fixture_id) REFERENCES fixtures (id)
                )
            """)
            
            # Player gameweek stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_gameweek_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    element INTEGER,
                    gameweek INTEGER,
                    minutes INTEGER,
                    goals_scored INTEGER,
                    assists INTEGER,
                    clean_sheets INTEGER,
                    goals_conceded INTEGER,
                    own_goals INTEGER,
                    penalties_saved INTEGER,
                    penalties_missed INTEGER,
                    yellow_cards INTEGER,
                    red_cards INTEGER,
                    saves INTEGER,
                    bonus INTEGER,
                    bps INTEGER,
                    influence REAL,
                    creativity REAL,
                    threat REAL,
                    ict_index REAL,
                    clearances_blocks_interceptions INTEGER,
                    defensive_contribution INTEGER,
                    recoveries INTEGER,
                    tackles INTEGER,
                    starts INTEGER,
                    expected_goals REAL,
                    expected_assists REAL,
                    expected_goal_involvements REAL,
                    expected_goals_conceded REAL,
                    in_dreamteam BOOLEAN,
                    total_points INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (element) REFERENCES players (id),
                    FOREIGN KEY (gameweek) REFERENCES gameweeks (id),
                    UNIQUE(element, gameweek)
                )
            """)
            
            # Player history table (season-long stats by GW)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    element INTEGER,
                    round INTEGER,
                    total_points INTEGER,
                    value INTEGER,
                    transfers_balance INTEGER,
                    selected INTEGER,
                    transfers_in INTEGER,
                    transfers_out INTEGER,
                    loaned_in INTEGER,
                    loaned_out INTEGER,
                    minutes INTEGER,
                    goals_scored INTEGER,
                    assists INTEGER,
                    clean_sheets INTEGER,
                    goals_conceded INTEGER,
                    own_goals INTEGER,
                    penalties_saved INTEGER,
                    penalties_missed INTEGER,
                    yellow_cards INTEGER,
                    red_cards INTEGER,
                    saves INTEGER,
                    bonus INTEGER,
                    bps INTEGER,
                    influence REAL,
                    creativity REAL,
                    threat REAL,
                    ict_index REAL,
                    clearances_blocks_interceptions INTEGER,
                    defensive_contribution INTEGER,
                    recoveries INTEGER,
                    tackles INTEGER,
                    expected_goals REAL,
                    expected_assists REAL,
                    expected_goal_involvements REAL,
                    expected_goals_conceded REAL,
                    starts INTEGER,
                    team_a_score INTEGER,
                    team_h_score INTEGER,
                    fixture INTEGER,
                    kickoff_time TEXT,
                    modified TEXT,
                    opponent_team INTEGER,
                    was_home BOOLEAN,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (element) REFERENCES players (id),
                    UNIQUE(element, round)
                )
            """)
            
            # Player fixtures table (upcoming games)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_fixtures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    element INTEGER,
                    event INTEGER,
                    code INTEGER,
                    fixture_id INTEGER,
                    team_h INTEGER,
                    team_a INTEGER,
                    is_home BOOLEAN,
                    opponent_team INTEGER,
                    kickoff_time TEXT,
                    difficulty INTEGER,
                    team_h_difficulty INTEGER,
                    team_a_difficulty INTEGER,
                    event_name TEXT,
                    finished BOOLEAN,
                    provisional_start_time BOOLEAN,
                    minutes INTEGER,
                    team_h_score INTEGER,
                    team_a_score INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (element) REFERENCES players (id),
                    FOREIGN KEY (event) REFERENCES gameweeks (id),
                    UNIQUE(element, event, code)
                )
            """)
            
            # System metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Model metrics registry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT,
                    train_start_gw INTEGER,
                    train_end_gw INTEGER,
                    validation_gw INTEGER,
                    test_gw INTEGER,
                    mae REAL,
                    rmse REAL,
                    r2 REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    artifact_path TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_metrics_model
                ON model_metrics(model_name, created_at DESC)
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_event ON fixtures(event)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_element ON player_gameweek_stats(element)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_history_element ON player_history(element)")
            
            # Create indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_position ON players(position)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_status ON players(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_total_points ON players(total_points)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_now_cost ON players(now_cost)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_web_name ON players(web_name)")
            
            conn.commit()
            self.logger.info("Database initialized successfully")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def update_teams(self, teams_df: pd.DataFrame) -> bool:
        """Update teams data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing data
                conn.execute("DELETE FROM teams")
                
                # Insert new data
                teams_df.to_sql('teams', conn, if_exists='append', index=False)
                
                self.logger.info(f"Updated {len(teams_df)} teams")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating teams: {str(e)}")
            return False
    
    def update_players(self, players_df: pd.DataFrame) -> bool:
        """Update players data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing data
                conn.execute("DELETE FROM players")
                
                # Insert new data
                players_df.to_sql('players', conn, if_exists='append', index=False)
                
                self.logger.info(f"Updated {len(players_df)} players")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating players: {str(e)}")
            return False
    
    def update_gameweeks(self, gameweeks_df: pd.DataFrame) -> bool:
        """Update gameweeks data"""
        try:
            import json
            
            # Convert any list/dict columns to JSON strings
            for col in gameweeks_df.columns:
                if gameweeks_df[col].dtype == 'object':
                    gameweeks_df[col] = gameweeks_df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                    )
            
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing data
                conn.execute("DELETE FROM gameweeks")
                
                # Insert new data
                gameweeks_df.to_sql('gameweeks', conn, if_exists='append', index=False)
                
                self.logger.info(f"Updated {len(gameweeks_df)} gameweeks")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating gameweeks: {str(e)}")
            return False
    
    def update_fixtures(self, fixtures_df: pd.DataFrame) -> bool:
        """Update fixtures data"""
        try:
            import json
            
            # Convert any list/dict columns to JSON strings
            for col in fixtures_df.columns:
                if fixtures_df[col].dtype == 'object':
                    fixtures_df[col] = fixtures_df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                    )
            
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing data
                conn.execute("DELETE FROM fixtures")
                
                # Insert new data
                fixtures_df.to_sql('fixtures', conn, if_exists='append', index=False)
                
                self.logger.info(f"Updated {len(fixtures_df)} fixtures")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating fixtures: {str(e)}")
            return False
    
    def update_player_gameweek_stats(self, stats_df: pd.DataFrame) -> bool:
        """Update player gameweek statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                prepared_df = self._prepare_dataframe_for_insert(conn, 'player_gameweek_stats', stats_df)
                self._delete_conflicting_rows(conn, 'player_gameweek_stats', ['element', 'gameweek'], prepared_df)
                prepared_df.to_sql('player_gameweek_stats', conn, if_exists='append', index=False)
                
                # Remove duplicates (keep latest)
                conn.execute("""
                    DELETE FROM player_gameweek_stats 
                    WHERE id NOT IN (
                        SELECT MAX(id) 
                        FROM player_gameweek_stats 
                        GROUP BY element, gameweek
                    )
                """)
                
                self.logger.info(f"Updated gameweek stats for {len(stats_df)} records")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating player gameweek stats: {str(e)}")
            return False
    
    def update_player_history(self, history_df: pd.DataFrame) -> bool:
        """Update player historical data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                prepared_df = self._prepare_dataframe_for_insert(conn, 'player_history', history_df)
                self._delete_conflicting_rows(conn, 'player_history', ['element', 'round'], prepared_df)
                prepared_df.to_sql('player_history', conn, if_exists='append', index=False)
                
                # Remove duplicates
                conn.execute("""
                    DELETE FROM player_history 
                    WHERE id NOT IN (
                        SELECT MAX(id) 
                        FROM player_history 
                        GROUP BY element, round
                    )
                """)
                
                self.logger.info(f"Updated history for {len(history_df)} records")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating player history: {str(e)}")
            return False
    
    def update_player_fixtures(self, fixtures_df: pd.DataFrame) -> bool:
        """Update player upcoming fixtures"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                fixtures_df = fixtures_df.rename(columns={'id': 'fixture_id'})
                prepared_df = self._prepare_dataframe_for_insert(conn, 'player_fixtures', fixtures_df)
                self._delete_conflicting_rows(conn, 'player_fixtures', ['element', 'event', 'code'], prepared_df)
                prepared_df.to_sql('player_fixtures', conn, if_exists='append', index=False)
                
                # Remove duplicates
                conn.execute("""
                    DELETE FROM player_fixtures 
                    WHERE id NOT IN (
                        SELECT MAX(id) 
                        FROM player_fixtures 
                        GROUP BY element, event, code
                    )
                """)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating player fixtures: {str(e)}")
            return False
    
    def get_player_count(self) -> int:
        """Get total number of players"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM players WHERE status != 'u'")
                return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Error getting player count: {str(e)}")
            return 0
    
    def get_current_gameweek(self) -> Optional[int]:
        """Get current gameweek number"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM gameweeks WHERE is_current = 1 LIMIT 1")
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Error getting current gameweek: {str(e)}")
            return None
    
    def get_upcoming_fixtures_count(self) -> int:
        """Get count of upcoming fixtures"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM fixtures WHERE finished = 0")
                return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Error getting upcoming fixtures count: {str(e)}")
            return 0
    
    def get_fixtures(self) -> pd.DataFrame:
        """Get all fixtures data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM fixtures 
                    ORDER BY event, kickoff_time
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            self.logger.error(f"Error getting fixtures: {str(e)}")
            return pd.DataFrame()
    
    def get_top_form_players(self, limit: int = 10) -> pd.DataFrame:
        """Get top players by recent form"""
        try:
            query = """
                SELECT 
                    p.web_name,
                    t.name as team_name,
                    p.position,
                    p.form as form_points,
                    p.total_points,
                    p.now_cost,
                    p.selected_by_percent
                FROM players p
                JOIN teams t ON p.team = t.id
                WHERE p.status = 'a' AND p.minutes > 0
                ORDER BY p.form DESC, p.total_points DESC
                LIMIT ?
            """
            
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn, params=[limit])
                
        except Exception as e:
            self.logger.error(f"Error getting top form players: {str(e)}")
            return pd.DataFrame()
    
    def get_recent_price_changes(self, days: int = 7) -> pd.DataFrame:
        """Get recent price changes (mock implementation)"""
        try:
            # This would need historical price tracking
            # For now, return empty DataFrame
            return pd.DataFrame(columns=['web_name', 'team_name', 'price_change', 'current_price'])
                
        except Exception as e:
            self.logger.error(f"Error getting price changes: {str(e)}")
            return pd.DataFrame()
    
    def upsert_fixture_odds(self, odds_records: List[Dict[str, Any]]) -> bool:
        """Insert or update fixture odds from bookmaker data."""
        if not odds_records:
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for record in odds_records:
                    cursor.execute(
                        """
                        INSERT INTO fixture_odds (
                            fixture_id, event, home_team, away_team,
                            bookmaker, market, home_implied_goals, away_implied_goals, raw_odds, fetched_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(fixture_id) DO UPDATE SET
                            event=excluded.event,
                            home_team=excluded.home_team,
                            away_team=excluded.away_team,
                            bookmaker=excluded.bookmaker,
                            market=excluded.market,
                            home_implied_goals=excluded.home_implied_goals,
                            away_implied_goals=excluded.away_implied_goals,
                            raw_odds=excluded.raw_odds,
                            fetched_at=CURRENT_TIMESTAMP
                        """,
                        (
                            record.get('fixture_id'),
                            record.get('event'),
                            record.get('home_team'),
                            record.get('away_team'),
                            record.get('bookmaker'),
                            record.get('market'),
                            record.get('home_implied_goals'),
                            record.get('away_implied_goals'),
                            record.get('raw_odds'),
                        )
                    )
            return True
        except Exception as e:
            self.logger.error(f"Error upserting fixture odds: {str(e)}")
            return False
    
    def get_fixture_odds_map(self) -> Dict[int, Dict[str, Any]]:
        """Return latest odds indexed by fixture_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                rows = cursor.execute("""
                    SELECT fixture_id, event, home_team, away_team,
                           bookmaker, market, home_implied_goals, away_implied_goals, raw_odds, fetched_at
                    FROM fixture_odds
                """).fetchall()
            odds_map = {}
            for row in rows:
                odds_map[row[0]] = {
                    'fixture_id': row[0],
                    'event': row[1],
                    'home_team': row[2],
                    'away_team': row[3],
                    'bookmaker': row[4],
                    'market': row[5],
                    'home_implied_goals': row[6],
                    'away_implied_goals': row[7],
                    'raw_odds': row[8],
                    'fetched_at': row[9]
                }
            return odds_map
        except Exception as e:
            self.logger.error(f"Error retrieving fixture odds: {str(e)}")
            return {}
    
    def get_active_player_ids(self) -> List[int]:
        """Get IDs of all active players"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM players WHERE status = 'a'")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting active player IDs: {str(e)}")
            return []
    
    def get_trackable_player_ids(self) -> List[int]:
        """Get IDs for all players that should have history tracked (not unavailable)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM players WHERE status != 'u' OR status IS NULL")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting trackable player IDs: {str(e)}")
            return []
    
    def get_players_missing_history(self) -> List[int]:
        """Get player IDs that lack any stored history rows."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.id
                    FROM players p
                    LEFT JOIN (
                        SELECT DISTINCT element
                        FROM player_history
                    ) ph ON p.id = ph.element
                    WHERE (p.status != 'u' OR p.status IS NULL)
                      AND ph.element IS NULL
                """)
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting players missing history: {str(e)}")
            return []
    
    def get_teams(self) -> pd.DataFrame:
        """Get all teams"""
        try:
            query = """
                SELECT id, name, short_name, code, strength, 
                       strength_overall_home, strength_overall_away,
                       strength_attack_home, strength_attack_away,
                       strength_defence_home, strength_defence_away
                FROM teams 
                ORDER BY name
            """
            return pd.read_sql_query(query, sqlite3.connect(self.db_path))
        except Exception as e:
            self.logger.error(f"Error getting teams: {str(e)}")
            return pd.DataFrame()
    
    def get_players_with_stats(self) -> pd.DataFrame:
        """Get players with comprehensive stats for modeling"""
        try:
            query = """
                SELECT 
                    p.id, p.code, p.web_name, p.first_name, p.second_name,
                    p.team, p.position, p.element_type, p.now_cost, p.total_points,
                    p.points_per_game, p.form, p.selected_by_percent,
                    p.transfers_in, p.transfers_in_event, p.transfers_out, p.transfers_out_event,
                    p.value_form, p.value_season, p.minutes, p.goals_scored, p.assists,
                    p.clean_sheets, p.goals_conceded, p.own_goals, p.penalties_saved,
                    p.penalties_missed, p.yellow_cards, p.red_cards, p.saves, p.bonus,
                    p.bps, p.influence, p.creativity, p.threat, p.ict_index,
                    p.news, p.chance_of_playing_this_round, p.chance_of_playing_next_round,
                    p.status, p.can_transact, p.can_select, p.dreamteam_count,
                    p.ep_next, p.ep_this, p.event_points, p.in_dreamteam,
                    t.name as team_name,
                    t.strength_overall_home,
                    t.strength_overall_away,
                    t.strength_attack_home,
                    t.strength_attack_away,
                    t.strength_defence_home,
                    t.strength_defence_away
                FROM players p
                LEFT JOIN teams t ON p.team = t.id
                WHERE p.status != 'u'
                ORDER BY p.total_points DESC
            """
            
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn)
                
        except Exception as e:
            self.logger.error(f"Error getting players with stats: {str(e)}")
            return pd.DataFrame()
    
    def get_player_history_data(self, player_id: int = None, limit_gws: int = 10) -> pd.DataFrame:
        """Get player historical performance data"""
        try:
            base_query = """
                SELECT 
                    ph.*,
                    p.web_name,
                    p.position,
                    t.name as team_name
                FROM player_history ph
                JOIN players p ON ph.element = p.id
                LEFT JOIN teams t ON p.team = t.id
            """
            
            params = []
            if player_id:
                base_query += " WHERE ph.element = ?"
                params.append(player_id)
            
            base_query += " ORDER BY ph.element, ph.round DESC"
            
            if limit_gws and not player_id:
                base_query += " LIMIT ?"
                params.append(limit_gws * 100)  # Rough limit
            
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(base_query, conn, params=params)
                
        except Exception as e:
            self.logger.error(f"Error getting player history: {str(e)}")
            return pd.DataFrame()

    def save_model_metrics(
        self,
        model_name: str,
        version: str,
        train_start_gw: int,
        train_end_gw: int,
        validation_gw: int,
        test_gw: int,
        metrics: Dict[str, float],
        artifact_path: Optional[str] = None
    ) -> bool:
        """Persist model evaluation metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO model_metrics (
                        model_name, version, train_start_gw, train_end_gw,
                        validation_gw, test_gw, mae, rmse, r2, artifact_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_name,
                        version,
                        train_start_gw,
                        train_end_gw,
                        validation_gw,
                        test_gw,
                        metrics.get('mae'),
                        metrics.get('rmse'),
                        metrics.get('r2'),
                        artifact_path
                    )
                )
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving model metrics: {str(e)}")
            return False

    def get_latest_model_metrics(self, model_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Fetch latest metrics for all models or a specific model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if model_name:
                    cursor.execute(
                        """
                        SELECT * FROM model_metrics
                        WHERE model_name = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (model_name,)
                    )
                    row = cursor.fetchone()
                    return {model_name: dict(row)} if row else {}

                cursor.execute(
                    """
                    SELECT *
                    FROM model_metrics
                    WHERE id IN (
                        SELECT MAX(id) FROM model_metrics GROUP BY model_name
                    )
                    """
                )
                rows = cursor.fetchall()
                return {row['model_name']: dict(row) for row in rows}
        except Exception as e:
            self.logger.error(f"Error fetching model metrics: {str(e)}")
            return {}
    
    def mark_last_update(self) -> bool:
        """Mark the time of last data update"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO system_metadata (key, value) VALUES (?, ?)",
                    ('last_update', datetime.now().isoformat())
                )
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error marking last update: {str(e)}")
            return False
    
    def get_last_update(self) -> Optional[str]:
        """Get timestamp of last data update"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM system_metadata WHERE key = 'last_update'")
                result = cursor.fetchone()
                if result:
                    # Parse and format the timestamp
                    dt = datetime.fromisoformat(result[0])
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                return None
        except Exception as e:
            self.logger.error(f"Error getting last update: {str(e)}")
            return None
