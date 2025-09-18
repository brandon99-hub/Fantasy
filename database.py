import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import logging

class FPLDatabase:
    """Handles SQLite database operations for FPL data"""
    
    def __init__(self, db_path: str = "fpl_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
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
                    web_name TEXT NOT NULL,
                    first_name TEXT,
                    second_name TEXT,
                    team INTEGER,
                    team_name TEXT,
                    position TEXT,
                    element_type INTEGER,
                    now_cost INTEGER,
                    total_points INTEGER,
                    points_per_game REAL,
                    form REAL,
                    selected_by_percent REAL,
                    transfers_in_event INTEGER,
                    transfers_out_event INTEGER,
                    value_form REAL,
                    value_season REAL,
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
                    news TEXT,
                    chance_of_playing_this_round INTEGER,
                    chance_of_playing_next_round INTEGER,
                    status TEXT,
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
                    finished BOOLEAN,
                    is_previous BOOLEAN,
                    is_current BOOLEAN,
                    is_next BOOLEAN,
                    chip_plays TEXT,
                    most_selected INTEGER,
                    most_transferred_in INTEGER,
                    top_element INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Fixtures table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fixtures (
                    id INTEGER PRIMARY KEY,
                    event INTEGER,
                    team_h INTEGER,
                    team_a INTEGER,
                    team_h_difficulty INTEGER,
                    team_a_difficulty INTEGER,
                    kickoff_time TEXT,
                    finished BOOLEAN,
                    team_h_score INTEGER,
                    team_a_score INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_h) REFERENCES teams (id),
                    FOREIGN KEY (team_a) REFERENCES teams (id),
                    FOREIGN KEY (event) REFERENCES gameweeks (id)
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
                    is_home BOOLEAN,
                    difficulty INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (element) REFERENCES players (id),
                    FOREIGN KEY (event) REFERENCES gameweeks (id),
                    UNIQUE(element, event)
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
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_event ON fixtures(event)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_element ON player_gameweek_stats(element)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_history_element ON player_history(element)")
            
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
                # Insert or replace data
                stats_df.to_sql('player_gameweek_stats', conn, if_exists='append', index=False)
                
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
                # Insert or replace data
                history_df.to_sql('player_history', conn, if_exists='append', index=False)
                
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
                fixtures_df.to_sql('player_fixtures', conn, if_exists='append', index=False)
                
                # Remove duplicates
                conn.execute("""
                    DELETE FROM player_fixtures 
                    WHERE id NOT IN (
                        SELECT MAX(id) 
                        FROM player_fixtures 
                        GROUP BY element, event
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
    
    def get_players_with_stats(self) -> pd.DataFrame:
        """Get players with comprehensive stats for modeling"""
        try:
            query = """
                SELECT 
                    p.*,
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
