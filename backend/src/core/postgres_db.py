"""PostgreSQL database layer for FPL manager index"""

import psycopg
from psycopg.rows import dict_row
from typing import List, Optional, Dict, Any
import logging
from contextlib import contextmanager
from datetime import datetime

from psycopg.types.json import Json

from backend.src.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class PostgresManagerDB:
    """PostgreSQL database operations for manager index"""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or settings.POSTGRES_CONNECTION_STRING
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = None
        try:
            conn = psycopg.connect(self.connection_string, row_factory=dict_row)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def initialize_schema(self) -> bool:
        """Create database tables and indexes if they don't exist"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Enable required extensions
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                    
                    # ==========================================
                    # MANAGER-RELATED TABLES (EXISTING)
                    # ==========================================
                    
                    # Create fpl_managers table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS fpl_managers (
                            id BIGSERIAL PRIMARY KEY,
                            manager_id INTEGER UNIQUE NOT NULL,
                            player_first_name VARCHAR(255),
                            player_last_name VARCHAR(255),
                            player_name VARCHAR(500),
                            team_name VARCHAR(500),
                            overall_rank INTEGER,
                            total_points INTEGER,
                            region VARCHAR(100),
                            started_event INTEGER,
                            favourite_team INTEGER,
                            player_region_id INTEGER,
                            summary_overall_points INTEGER,
                            summary_overall_rank INTEGER,
                            summary_event_points INTEGER,
                            current_event INTEGER,
                            leagues JSONB,
                            last_deadline_bank INTEGER,
                            last_deadline_value INTEGER,
                            last_deadline_total_transfers INTEGER,
                            is_active BOOLEAN DEFAULT TRUE,
                            first_crawled_at TIMESTAMP DEFAULT NOW(),
                            last_updated_at TIMESTAMP DEFAULT NOW(),
                            crawl_status VARCHAR(50) DEFAULT 'pending'
                        );
                    """)
                    
                    # Create indexes
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_manager_id ON fpl_managers(manager_id);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_overall_rank ON fpl_managers(overall_rank) WHERE is_active = TRUE;")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_last_updated ON fpl_managers(last_updated_at);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_crawl_status ON fpl_managers(crawl_status);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_name_search ON fpl_managers(player_name, team_name) WHERE is_active = TRUE;")
                    
                    # Full-text search indexes
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_player_name_fts 
                        ON fpl_managers USING gin(to_tsvector('english', COALESCE(player_name, '')));
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_team_name_fts 
                        ON fpl_managers USING gin(to_tsvector('english', COALESCE(team_name, '')));
                    """)
                    
                    # Trigram indexes for fuzzy search
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_player_name_trgm 
                        ON fpl_managers USING gin(COALESCE(player_name, '') gin_trgm_ops);
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_team_name_trgm 
                        ON fpl_managers USING gin(COALESCE(team_name, '') gin_trgm_ops);
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_leagues_gin
                        ON fpl_managers USING gin(COALESCE(leagues, '{}'::jsonb));
                    """)
                    
                    # Manager league memberships table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS manager_league_memberships (
                            id BIGSERIAL PRIMARY KEY,
                            manager_id INTEGER NOT NULL REFERENCES fpl_managers(manager_id) ON DELETE CASCADE,
                            league_id INTEGER NOT NULL,
                            league_name VARCHAR(500),
                            league_type VARCHAR(50),
                            league_rank INTEGER,
                            league_points INTEGER,
                            last_synced_at TIMESTAMP DEFAULT NOW(),
                            UNIQUE(manager_id, league_id)
                        );
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_manager_league_lookup
                        ON manager_league_memberships(league_id, league_rank);
                    """)
                    
                    # Manager gameweek history
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS manager_gameweeks (
                            manager_id INTEGER NOT NULL,
                            event INTEGER NOT NULL,
                            gw_points INTEGER,
                            total_points INTEGER,
                            overall_rank INTEGER,
                            event_rank INTEGER,
                            bank INTEGER,
                            team_value INTEGER,
                            event_transfers INTEGER,
                            event_transfers_cost INTEGER,
                            chip_played VARCHAR(50),
                            captain_id INTEGER,
                            vice_captain_id INTEGER,
                            points_on_bench INTEGER,
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW(),
                            PRIMARY KEY (manager_id, event),
                            FOREIGN KEY (manager_id) REFERENCES fpl_managers(manager_id) ON DELETE CASCADE
                        );
                    """)

                    # Manager picks per gameweek
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS manager_picks (
                            id BIGSERIAL PRIMARY KEY,
                            manager_id INTEGER NOT NULL,
                            event INTEGER NOT NULL,
                            element_id INTEGER NOT NULL,
                            position INTEGER,
                            is_captain BOOLEAN DEFAULT FALSE,
                            is_vice_captain BOOLEAN DEFAULT FALSE,
                            multiplier INTEGER,
                            added_at TIMESTAMP DEFAULT NOW(),
                            UNIQUE (manager_id, event, element_id),
                            FOREIGN KEY (manager_id) REFERENCES fpl_managers(manager_id) ON DELETE CASCADE
                        );
                    """)

                    # Manager transfers per gameweek
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS manager_transfers (
                            id BIGSERIAL PRIMARY KEY,
                            manager_id INTEGER NOT NULL,
                            event INTEGER NOT NULL,
                            element_in INTEGER NOT NULL,
                            element_out INTEGER NOT NULL,
                            purchase_price INTEGER,
                            selling_price INTEGER,
                            created_at TIMESTAMP DEFAULT NOW(),
                            FOREIGN KEY (manager_id) REFERENCES fpl_managers(manager_id) ON DELETE CASCADE
                        );
                    """)

                    # Manager cohorts (e.g. top10k_overall)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS manager_cohorts (
                            id BIGSERIAL PRIMARY KEY,
                            manager_id INTEGER NOT NULL,
                            cohort_name VARCHAR(100) NOT NULL,
                            event_from INTEGER NOT NULL,
                            event_to INTEGER,
                            rank_at_entry INTEGER,
                            meta JSONB,
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW(),
                            UNIQUE (manager_id, cohort_name, event_from),
                            FOREIGN KEY (manager_id) REFERENCES fpl_managers(manager_id) ON DELETE CASCADE
                        );
                    """)

                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_manager_gameweeks_mgr_event
                        ON manager_gameweeks(manager_id, event);
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_manager_picks_mgr_event
                        ON manager_picks(manager_id, event);
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_manager_transfers_mgr_event
                        ON manager_transfers(manager_id, event);
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_manager_cohorts_mgr
                        ON manager_cohorts(manager_id, cohort_name);
                    """)
                    
                    # Performance indexes for manager_picks aggregation
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_manager_picks_event_player 
                        ON manager_picks(event, element_id);
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_manager_picks_captaincy 
                        ON manager_picks(event, element_id, is_captain, is_vice_captain);
                    """)

                    # Create crawl_progress table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS crawl_progress (
                            id SERIAL PRIMARY KEY,
                            crawl_type VARCHAR(50) NOT NULL,
                            start_manager_id INTEGER,
                            end_manager_id INTEGER,
                            current_manager_id INTEGER,
                            total_discovered INTEGER DEFAULT 0,
                            total_indexed INTEGER DEFAULT 0,
                            total_failed INTEGER DEFAULT 0,
                            total_deleted INTEGER DEFAULT 0,
                            status VARCHAR(50) DEFAULT 'running',
                            started_at TIMESTAMP DEFAULT NOW(),
                            last_heartbeat TIMESTAMP DEFAULT NOW(),
                            completed_at TIMESTAMP,
                            error_message TEXT,
                            metadata JSONB
                        );
                    """)
                    
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_crawl_status ON crawl_progress(status, last_heartbeat);")
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_crawl_league_meta
                        ON crawl_progress ((metadata->>'league_id'))
                        WHERE crawl_type = 'league';
                    """)
                    
                    # Create crawl_errors table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS crawl_errors (
                            id SERIAL PRIMARY KEY,
                            manager_id INTEGER,
                            error_type VARCHAR(100),
                            error_message TEXT,
                            http_status_code INTEGER,
                            occurred_at TIMESTAMP DEFAULT NOW(),
                            retry_count INTEGER DEFAULT 0
                        );
                    """)
                    
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_crawl_errors_manager ON crawl_errors(manager_id);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_crawl_errors_type ON crawl_errors(error_type, occurred_at);")
                    
                    # ==========================================
                    # FPL DATA TABLES (NEW)
                    # ==========================================
                    
                    # Teams table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS teams (
                            id INTEGER PRIMARY KEY,
                            name VARCHAR(100) NOT NULL,
                            short_name VARCHAR(10),
                            strength INTEGER,
                            strength_overall_home INTEGER,
                            strength_overall_away INTEGER,
                            strength_attack_home INTEGER,
                            strength_attack_away INTEGER,
                            strength_defence_home INTEGER,
                            strength_defence_away INTEGER,
                            pulse_id INTEGER,
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(name);")
                    
                    # Players table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS players (
                            id INTEGER PRIMARY KEY,
                            web_name VARCHAR(100),
                            first_name VARCHAR(100),
                            second_name VARCHAR(100),
                            team_id INTEGER REFERENCES teams(id),
                            element_type INTEGER,
                            now_cost INTEGER,
                            cost_change_start INTEGER,
                            cost_change_event INTEGER,
                            total_points INTEGER,
                            points_per_game NUMERIC(5,2),
                            selected_by_percent NUMERIC(5,2),
                            form NUMERIC(5,2),
                            transfers_in BIGINT,
                            transfers_out BIGINT,
                            transfers_in_event BIGINT,
                            transfers_out_event BIGINT,
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
                            influence NUMERIC(10,2),
                            creativity NUMERIC(10,2),
                            threat NUMERIC(10,2),
                            ict_index NUMERIC(10,2),
                            expected_goals NUMERIC(10,2),
                            expected_assists NUMERIC(10,2),
                            expected_goal_involvements NUMERIC(10,2),
                            expected_goals_conceded NUMERIC(10,2),
                            expected_goals_per_90 NUMERIC(5,2),
                            expected_assists_per_90 NUMERIC(5,2),
                            xg_source VARCHAR(20) DEFAULT 'fpl',
                            starts INTEGER,
                            news TEXT,
                            news_added TIMESTAMP,
                            chance_of_playing_this_round INTEGER,
                            chance_of_playing_next_round INTEGER,
                            status VARCHAR(10),
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_players_position ON players(element_type);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_players_status ON players(status);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_players_cost ON players(now_cost);")
                    
                    # Gameweeks table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS gameweeks (
                            id INTEGER PRIMARY KEY,
                            name VARCHAR(50),
                            deadline_time TIMESTAMP,
                            finished BOOLEAN DEFAULT FALSE,
                            is_current BOOLEAN DEFAULT FALSE,
                            is_next BOOLEAN DEFAULT FALSE,
                            is_previous BOOLEAN DEFAULT FALSE,
                            average_entry_score INTEGER,
                            highest_score INTEGER,
                            most_selected BIGINT,
                            most_transferred_in BIGINT,
                            most_captained BIGINT,
                            most_vice_captained BIGINT,
                            top_element INTEGER,
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_gameweeks_current ON gameweeks(is_current) WHERE is_current = TRUE;")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_gameweeks_deadline ON gameweeks(deadline_time);")
                    
                    # Fixtures table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS fixtures (
                            id INTEGER PRIMARY KEY,
                            event INTEGER REFERENCES gameweeks(id),
                            team_h INTEGER REFERENCES teams(id),
                            team_a INTEGER REFERENCES teams(id),
                            team_h_score INTEGER,
                            team_a_score INTEGER,
                            kickoff_time TIMESTAMP,
                            finished BOOLEAN DEFAULT FALSE,
                            finished_provisional BOOLEAN DEFAULT FALSE,
                            started BOOLEAN DEFAULT FALSE,
                            team_h_difficulty INTEGER,
                            team_a_difficulty INTEGER,
                            pulse_id INTEGER,
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_event ON fixtures(event);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_teams ON fixtures(team_h, team_a);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff ON fixtures(kickoff_time);")
                    
                    # Player history table (per-gameweek stats)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS player_history (
                            player_id INTEGER REFERENCES players(id),
                            event INTEGER REFERENCES gameweeks(id),
                            fixture INTEGER REFERENCES fixtures(id),
                            opponent_team INTEGER REFERENCES teams(id),
                            total_points INTEGER,
                            was_home BOOLEAN,
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
                            influence NUMERIC(10,2),
                            creativity NUMERIC(10,2),
                            threat NUMERIC(10,2),
                            ict_index NUMERIC(10,2),
                            value INTEGER,
                            transfers_balance INTEGER,
                            selected INTEGER,
                            transfers_in INTEGER,
                            transfers_out INTEGER,
                            expected_goals NUMERIC(10,2),
                            expected_assists NUMERIC(10,2),
                            expected_goal_involvements NUMERIC(10,2),
                            expected_goals_conceded NUMERIC(10,2),
                            created_at TIMESTAMP DEFAULT NOW(),
                            PRIMARY KEY (player_id, event)
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_player_history_player ON player_history(player_id, event DESC);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_player_history_event ON player_history(event);")
                    
                    # Predictions table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS predictions (
                            player_id INTEGER REFERENCES players(id),
                            gameweek INTEGER REFERENCES gameweeks(id),
                            predicted_points NUMERIC(10,2),
                            predicted_minutes NUMERIC(10,2),
                            predicted_goals NUMERIC(10,2),
                            predicted_assists NUMERIC(10,2),
                            predicted_clean_sheet_prob NUMERIC(5,4),
                            confidence_score NUMERIC(5,4),
                            lower_bound NUMERIC(10,2),
                            upper_bound NUMERIC(10,2),
                            model_version VARCHAR(50),
                            created_at TIMESTAMP DEFAULT NOW(),
                            PRIMARY KEY (player_id, gameweek)
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_gameweek ON predictions(gameweek);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_points ON predictions(predicted_points DESC);")
                    
                    # Model metrics table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS model_metrics (
                            id SERIAL PRIMARY KEY,
                            model_name VARCHAR(100) NOT NULL,
                            version VARCHAR(50) NOT NULL,
                            train_start_gw INTEGER,
                            train_end_gw INTEGER,
                            validation_gw INTEGER,
                            test_gw INTEGER,
                            mae NUMERIC(10,4),
                            rmse NUMERIC(10,4),
                            r2_score NUMERIC(10,4),
                            artifact_path VARCHAR(500),
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON model_metrics(model_name, created_at DESC);")
                    
                    # Fixture odds table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS fixture_odds (
                            fixture_id INTEGER REFERENCES fixtures(id),
                            bookmaker VARCHAR(100),
                            home_implied_goals NUMERIC(5,2),
                            away_implied_goals NUMERIC(5,2),
                            home_win_odds NUMERIC(10,2),
                            draw_odds NUMERIC(10,2),
                            away_win_odds NUMERIC(10,2),
                            over_2_5_odds NUMERIC(10,2),
                            under_2_5_odds NUMERIC(10,2),
                            fetched_at TIMESTAMP DEFAULT NOW(),
                            PRIMARY KEY (fixture_id, bookmaker)
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_fixture_odds_fixture ON fixture_odds(fixture_id);")
                    
                    # Data updates tracking table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS data_updates (
                            update_type VARCHAR(50) PRIMARY KEY,
                            updated_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    
                    # Cache metadata tracking table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS cache_metadata (
                            cache_key VARCHAR(100) PRIMARY KEY,
                            last_updated TIMESTAMP DEFAULT NOW(),
                            last_event INTEGER,
                            record_count INTEGER,
                            metadata JSONB
                        );
                    """)
                    
                    # ==========================================
                    # ADAPTIVE LEARNING TABLES
                    # ==========================================
                    
                    # Prediction feedback - track prediction accuracy for continuous learning
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS prediction_feedback (
                            player_id INTEGER REFERENCES players(id),
                            gameweek INTEGER REFERENCES gameweeks(id),
                            predicted_points NUMERIC(10,2),
                            actual_points INTEGER,
                            prediction_error NUMERIC(10,2),
                            absolute_error NUMERIC(10,2),
                            squared_error NUMERIC(10,2),
                            model_version VARCHAR(50),
                            player_position INTEGER,
                            player_price INTEGER,
                            player_team INTEGER,
                            was_home BOOLEAN,
                            created_at TIMESTAMP DEFAULT NOW(),
                            PRIMARY KEY (player_id, gameweek, model_version)
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_feedback_gw ON prediction_feedback(gameweek);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_feedback_error ON prediction_feedback(absolute_error DESC);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_feedback_position ON prediction_feedback(player_position, gameweek);")
                    
                    # Transfer outcomes - measure ROI of transfer recommendations
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS transfer_outcomes (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER,
                            gameweek INTEGER REFERENCES gameweeks(id),
                            player_out_id INTEGER REFERENCES players(id),
                            player_in_id INTEGER REFERENCES players(id),
                            predicted_points_gain NUMERIC(10,2),
                            actual_points_out INTEGER,
                            actual_points_in INTEGER,
                            actual_points_gain INTEGER,
                            transfer_cost INTEGER DEFAULT 4,
                            net_gain INTEGER,
                            was_successful BOOLEAN,
                            was_recommended BOOLEAN DEFAULT TRUE,
                            recommendation_rank INTEGER,
                            created_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_transfer_outcomes_user ON transfer_outcomes(user_id, gameweek);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_transfer_outcomes_success ON transfer_outcomes(was_successful, gameweek);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_transfer_outcomes_gw ON transfer_outcomes(gameweek);")
                    
                    # User decisions - track user acceptance/rejection of recommendations
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_decisions (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER NOT NULL,
                            gameweek INTEGER REFERENCES gameweeks(id),
                            decision_type VARCHAR(50) NOT NULL,
                            recommended_option_id INTEGER,
                            recommended_option_name VARCHAR(200),
                            actual_choice_id INTEGER,
                            actual_choice_name VARCHAR(200),
                            user_accepted BOOLEAN,
                            recommended_points NUMERIC(10,2),
                            actual_points INTEGER,
                            confidence_score NUMERIC(5,4),
                            context JSONB,
                            created_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_decisions_user ON user_decisions(user_id, gameweek);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_decisions_type ON user_decisions(decision_type, user_accepted);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_decisions_gw ON user_decisions(gameweek);")
                    
                    # User preferences - store learned user preferences
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_preferences (
                            user_id INTEGER PRIMARY KEY,
                            risk_tolerance NUMERIC(3,2) DEFAULT 0.5,
                            prefers_differentials BOOLEAN DEFAULT FALSE,
                            differential_threshold NUMERIC(5,2) DEFAULT 10.0,
                            budget_strategy VARCHAR(50) DEFAULT 'balanced',
                            preferred_formation VARCHAR(10),
                            favorite_teams JSONB,
                            avoided_teams JSONB,
                            captain_strategy VARCHAR(50) DEFAULT 'safe',
                            transfer_aggressiveness NUMERIC(3,2) DEFAULT 0.5,
                            learned_from_decisions INTEGER DEFAULT 0,
                            confidence_level NUMERIC(3,2) DEFAULT 0.0,
                            last_updated TIMESTAMP DEFAULT NOW(),
                            created_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_preferences_updated ON user_preferences(last_updated);")
                    
                    # Feature importance history - track which features predict best over time
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS feature_importance_history (
                            id SERIAL PRIMARY KEY,
                            model_name VARCHAR(100) NOT NULL,
                            model_version VARCHAR(50),
                            gameweek INTEGER REFERENCES gameweeks(id),
                            feature_name VARCHAR(100) NOT NULL,
                            importance_score NUMERIC(10,6),
                            rank INTEGER,
                            feature_type VARCHAR(50),
                            created_at TIMESTAMP DEFAULT NOW()
                        );
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_feature_importance_model ON feature_importance_history(model_name, gameweek);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_feature_importance_feature ON feature_importance_history(feature_name, gameweek);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_feature_importance_gw ON feature_importance_history(gameweek DESC);")
                    
                    # Materialized view for player meta features (performance optimization)
                    cur.execute("""
                        CREATE MATERIALIZED VIEW IF NOT EXISTS player_meta_features AS
                        WITH manager_counts AS (
                            SELECT event, COUNT(DISTINCT manager_id) as manager_count
                            FROM manager_picks
                            GROUP BY event
                        )
                        SELECT 
                            mp.event,
                            mp.element_id as player_id,
                            COUNT(DISTINCT mp.manager_id) as picked_count,
                            SUM(CASE WHEN mp.is_captain THEN 1 ELSE 0 END) as captain_count,
                            SUM(CASE WHEN mp.is_vice_captain THEN 1 ELSE 0 END) as vice_count,
                            mc.manager_count,
                            (COUNT(DISTINCT mp.manager_id)::float / NULLIF(mc.manager_count, 0) * 100) as top_cohort_ownership_pct,
                            (SUM(CASE WHEN mp.is_captain THEN 1 ELSE 0 END)::float / NULLIF(mc.manager_count, 0) * 100) as top_cohort_captain_pct,
                            (SUM(CASE WHEN mp.is_vice_captain THEN 1 ELSE 0 END)::float / NULLIF(mc.manager_count, 0) * 100) as top_cohort_vice_captain_pct
                        FROM manager_picks mp
                        JOIN manager_counts mc ON mp.event = mc.event
                        GROUP BY mp.event, mp.element_id, mc.manager_count;
                    """)
                    
                    # Index on materialized view for fast lookups
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_player_meta_features_lookup
                        ON player_meta_features(event, player_id);
                    """)
                    
                    conn.commit()
                    self.logger.info("✅ Database schema initialized successfully (manager + FPL data tables)")
                    return True
        except Exception as e:
            self.logger.error(f"Schema initialization failed: {str(e)}")
            return False
    
    def get_manager_by_id(self, manager_id: int) -> Optional[Dict[str, Any]]:
        """Get manager by FPL manager ID"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM fpl_managers WHERE manager_id = %s",
                        (manager_id,)
                    )
                    return cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error fetching manager {manager_id}: {str(e)}")
            return None
    
    def search_managers(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search managers using full-text search"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Use full-text search with trigram fallback
                    cur.execute("""
                        SELECT 
                            manager_id,
                            player_name,
                            team_name,
                            overall_rank,
                            total_points,
                            summary_overall_rank,
                            is_active,
                            last_updated_at
                        FROM fpl_managers
                        WHERE is_active = TRUE
                          AND (
                            to_tsvector('english', COALESCE(player_name, '')) @@ plainto_tsquery('english', %s)
                            OR to_tsvector('english', COALESCE(team_name, '')) @@ plainto_tsquery('english', %s)
                            OR COALESCE(player_name, '') ILIKE %s
                            OR COALESCE(team_name, '') ILIKE %s
                          )
                        ORDER BY 
                          GREATEST(
                            ts_rank(to_tsvector('english', COALESCE(player_name, '')), plainto_tsquery('english', %s)),
                            ts_rank(to_tsvector('english', COALESCE(team_name, '')), plainto_tsquery('english', %s))
                          ) DESC,
                          overall_rank ASC NULLS LAST
                        LIMIT %s;
                    """, (query, query, f'%{query}%', f'%{query}%', query, query, limit))
                    return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error searching managers: {str(e)}")
            return []
    
    def _prepare_json_fields(self, payload: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """Wrap JSON fields with psycopg Json helper"""
        prepared = payload.copy()
        for field in fields:
            if field in prepared and prepared[field] is not None:
                prepared[field] = Json(prepared[field])
        return prepared

    def upsert_manager(self, manager_data: Dict[str, Any]) -> bool:
        """Insert or update manager data"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    values = self._prepare_json_fields(manager_data, ['leagues'])
                    cur.execute("""
                        INSERT INTO fpl_managers (
                            manager_id, player_first_name, player_last_name, player_name,
                            team_name, overall_rank, total_points, region, started_event,
                            favourite_team, player_region_id, summary_overall_points,
                            summary_overall_rank, summary_event_points, current_event,
                            leagues, last_deadline_bank, last_deadline_value,
                            last_deadline_total_transfers, is_active, crawl_status
                        ) VALUES (
                            %(manager_id)s, %(player_first_name)s, %(player_last_name)s, %(player_name)s,
                            %(team_name)s, %(overall_rank)s, %(total_points)s, %(region)s, %(started_event)s,
                            %(favourite_team)s, %(player_region_id)s, %(summary_overall_points)s,
                            %(summary_overall_rank)s, %(summary_event_points)s, %(current_event)s,
                            %(leagues)s, %(last_deadline_bank)s, %(last_deadline_value)s,
                            %(last_deadline_total_transfers)s, %(is_active)s, %(crawl_status)s
                        )
                        ON CONFLICT (manager_id) DO UPDATE SET
                            player_first_name = EXCLUDED.player_first_name,
                            player_last_name = EXCLUDED.player_last_name,
                            player_name = EXCLUDED.player_name,
                            team_name = EXCLUDED.team_name,
                            overall_rank = EXCLUDED.overall_rank,
                            total_points = EXCLUDED.total_points,
                            region = EXCLUDED.region,
                            started_event = EXCLUDED.started_event,
                            favourite_team = EXCLUDED.favourite_team,
                            player_region_id = EXCLUDED.player_region_id,
                            summary_overall_points = EXCLUDED.summary_overall_points,
                            summary_overall_rank = EXCLUDED.summary_overall_rank,
                            summary_event_points = EXCLUDED.summary_event_points,
                            current_event = EXCLUDED.current_event,
                            leagues = EXCLUDED.leagues,
                            last_deadline_bank = EXCLUDED.last_deadline_bank,
                            last_deadline_value = EXCLUDED.last_deadline_value,
                            last_deadline_total_transfers = EXCLUDED.last_deadline_total_transfers,
                            is_active = EXCLUDED.is_active,
                            crawl_status = EXCLUDED.crawl_status,
                            last_updated_at = NOW();
                    """, values)
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error upserting manager {manager_data.get('manager_id')}: {str(e)}")
            return False
    
    def upsert_manager_leagues(self, manager_id: int, leagues: Optional[List[Dict[str, Any]]]) -> bool:
        """Replace manager league memberships"""
        if leagues is None:
            return True
        
        normalized = []
        for league in leagues:
            league_id = league.get('id')
            if not league_id:
                continue
            normalized.append({
                'manager_id': manager_id,
                'league_id': league_id,
                'league_name': league.get('name'),
                'league_type': league.get('type'),
                'league_rank': league.get('rank'),
                'league_points': league.get('total_points')
            })
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM manager_league_memberships WHERE manager_id = %s",
                        (manager_id,)
                    )
                    
                    if normalized:
                        cur.executemany("""
                            INSERT INTO manager_league_memberships (
                                manager_id, league_id, league_name, league_type,
                                league_rank, league_points, last_synced_at
                            ) VALUES (
                                %(manager_id)s, %(league_id)s, %(league_name)s, %(league_type)s,
                                %(league_rank)s, %(league_points)s, NOW()
                            )
                            ON CONFLICT (manager_id, league_id) DO UPDATE SET
                                league_name = EXCLUDED.league_name,
                                league_type = EXCLUDED.league_type,
                                league_rank = EXCLUDED.league_rank,
                                league_points = EXCLUDED.league_points,
                                last_synced_at = NOW();
                        """, normalized)
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error updating leagues for manager {manager_id}: {str(e)}")
            return False
    
    def get_league_progress(self, league_id: int) -> Optional[Dict[str, Any]]:
        """Get crawl progress for a specific league"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT *
                        FROM crawl_progress
                        WHERE crawl_type = 'league'
                          AND metadata ->> 'league_id' = %s
                        ORDER BY started_at DESC
                        LIMIT 1;
                    """, (str(league_id),))
                    return cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error fetching league progress: {str(e)}")
            return None
    
    def get_managers_in_league(self, league_id: int, limit: int = 5000) -> List[Dict[str, Any]]:
        """Get managers stored for a given league"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT m.manager_id, m.player_name, m.team_name,
                               l.league_rank, l.league_points, m.overall_rank
                        FROM manager_league_memberships l
                        JOIN fpl_managers m ON l.manager_id = m.manager_id
                        WHERE l.league_id = %s
                        ORDER BY l.league_rank ASC NULLS LAST
                        LIMIT %s;
                    """, (league_id, limit))
                    return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error getting managers for league {league_id}: {str(e)}")
            return []
    
    def get_league_summaries(self) -> List[Dict[str, Any]]:
        """Get summary (manager counts) for all leagues we have indexed"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            league_id,
                            MAX(league_name) AS league_name,
                            COUNT(*) AS manager_count
                        FROM manager_league_memberships
                        GROUP BY league_id
                        ORDER BY league_id;
                    """)
                    return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error fetching league summaries: {str(e)}")
            return []
    
    def get_league_summary(self, league_id: int) -> Optional[Dict[str, Any]]:
        """Get summary (manager count) for a single league"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            league_id,
                            MAX(league_name) AS league_name,
                            COUNT(*) AS manager_count
                        FROM manager_league_memberships
                        WHERE league_id = %s
                        GROUP BY league_id;
                    """, (league_id,))
                    return cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error fetching league summary for {league_id}: {str(e)}")
            return None
    
    def bulk_upsert_managers(self, managers: List[Dict[str, Any]]) -> int:
        """Bulk insert/update managers using COPY"""
        if not managers:
            return 0
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    prepared = [
                        self._prepare_json_fields(manager, ['leagues'])
                        for manager in managers
                    ]
                    # Use executemany for bulk operations
                    cur.executemany("""
                        INSERT INTO fpl_managers (
                            manager_id, player_first_name, player_last_name, player_name,
                            team_name, overall_rank, total_points, region, started_event,
                            favourite_team, player_region_id, summary_overall_points,
                            summary_overall_rank, summary_event_points, current_event,
                            leagues, last_deadline_bank, last_deadline_value,
                            last_deadline_total_transfers, is_active, crawl_status
                        ) VALUES (
                            %(manager_id)s, %(player_first_name)s, %(player_last_name)s, %(player_name)s,
                            %(team_name)s, %(overall_rank)s, %(total_points)s, %(region)s, %(started_event)s,
                            %(favourite_team)s, %(player_region_id)s, %(summary_overall_points)s,
                            %(summary_overall_rank)s, %(summary_event_points)s, %(current_event)s,
                            %(leagues)s, %(last_deadline_bank)s, %(last_deadline_value)s,
                            %(last_deadline_total_transfers)s, %(is_active)s, %(crawl_status)s
                        )
                        ON CONFLICT (manager_id) DO UPDATE SET
                            player_first_name = EXCLUDED.player_first_name,
                            player_last_name = EXCLUDED.player_last_name,
                            player_name = EXCLUDED.player_name,
                            team_name = EXCLUDED.team_name,
                            overall_rank = EXCLUDED.overall_rank,
                            total_points = EXCLUDED.total_points,
                            region = EXCLUDED.region,
                            started_event = EXCLUDED.started_event,
                            favourite_team = EXCLUDED.favourite_team,
                            player_region_id = EXCLUDED.player_region_id,
                            summary_overall_points = EXCLUDED.summary_overall_points,
                            summary_overall_rank = EXCLUDED.summary_overall_rank,
                            summary_event_points = EXCLUDED.summary_event_points,
                            current_event = EXCLUDED.current_event,
                            leagues = EXCLUDED.leagues,
                            last_deadline_bank = EXCLUDED.last_deadline_bank,
                            last_deadline_value = EXCLUDED.last_deadline_value,
                            last_deadline_total_transfers = EXCLUDED.last_deadline_total_transfers,
                            is_active = EXCLUDED.is_active,
                            crawl_status = EXCLUDED.crawl_status,
                            last_updated_at = NOW();
                    """, prepared)
                    conn.commit()
                    return len(managers)
        except Exception as e:
            self.logger.error(f"Error bulk upserting managers: {str(e)}")
            return 0
    
    def mark_manager_deleted(self, manager_id: int) -> bool:
        """Mark manager as deleted/inactive"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE fpl_managers
                        SET is_active = FALSE, crawl_status = 'deleted', last_updated_at = NOW()
                        WHERE manager_id = %s;
                    """, (manager_id,))
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error marking manager {manager_id} as deleted: {str(e)}")
            return False
    
    def get_crawl_progress(self, crawl_type: str = 'range') -> Optional[Dict[str, Any]]:
        """Get current crawl progress"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT * FROM crawl_progress
                        WHERE crawl_type = %s AND status IN ('running', 'paused')
                        ORDER BY started_at DESC
                        LIMIT 1;
                    """, (crawl_type,))
                    return cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error fetching crawl progress: {str(e)}")
            return None
    
    def create_crawl_progress(self, crawl_data: Dict[str, Any]) -> Optional[int]:
        """Create new crawl progress record"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    values = self._prepare_json_fields(crawl_data, ['metadata'])
                    cur.execute("""
                        INSERT INTO crawl_progress (
                            crawl_type, start_manager_id, end_manager_id, current_manager_id,
                            total_discovered, total_indexed, total_failed, total_deleted,
                            status, metadata
                        ) VALUES (
                            %(crawl_type)s, %(start_manager_id)s, %(end_manager_id)s, %(current_manager_id)s,
                            %(total_discovered)s, %(total_indexed)s, %(total_failed)s, %(total_deleted)s,
                            %(status)s, %(metadata)s
                        )
                        RETURNING id;
                    """, values)
                    result = cur.fetchone()
                    conn.commit()
                    return result['id'] if result else None
        except Exception as e:
            self.logger.error(f"Error creating crawl progress: {str(e)}")
            return None
    
    def update_crawl_progress(self, progress_id: int, updates: Dict[str, Any]) -> bool:
        """Update crawl progress"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    set_clauses = []
                    values = []
                    for key, value in updates.items():
                        if key == 'completed_at' and value is True:
                            set_clauses.append("completed_at = NOW()")
                        elif key == 'completed_at' and value is False:
                            set_clauses.append("completed_at = NULL")
                        elif key == 'metadata' and value is not None:
                            set_clauses.append("metadata = %s")
                            values.append(Json(value))
                        else:
                            set_clauses.append(f"{key} = %s")
                            values.append(value)
                    
                    values.append(progress_id)
                    cur.execute(f"""
                        UPDATE crawl_progress
                        SET {', '.join(set_clauses)}, last_heartbeat = NOW()
                        WHERE id = %s;
                    """, values)
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error updating crawl progress: {str(e)}")
            return False
    
    def log_crawl_error(self, error_data: Dict[str, Any]) -> bool:
        """Log crawl error"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO crawl_errors (
                            manager_id, error_type, error_message, http_status_code, retry_count
                        ) VALUES (
                            %(manager_id)s, %(error_type)s, %(error_message)s, %(http_status_code)s, %(retry_count)s
                        );
                    """, error_data)
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error logging crawl error: {str(e)}")
            return False
    
    def get_manager_count(self) -> int:
        """Get total number of indexed managers"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) as count FROM fpl_managers WHERE is_active = TRUE;")
                    result = cur.fetchone()
                    return result['count'] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting manager count: {str(e)}")
            return 0

    def upsert_manager_gameweeks(self, rows: List[Dict[str, Any]]) -> int:
        """Bulk upsert manager gameweek summary rows."""
        if not rows:
            return 0
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO manager_gameweeks (
                            manager_id, event, gw_points, total_points,
                            overall_rank, event_rank, bank, team_value,
                            event_transfers, event_transfers_cost, chip_played,
                            captain_id, vice_captain_id, points_on_bench
                        ) VALUES (
                            %(manager_id)s, %(event)s, %(gw_points)s, %(total_points)s,
                            %(overall_rank)s, %(event_rank)s, %(bank)s, %(team_value)s,
                            %(event_transfers)s, %(event_transfers_cost)s, %(chip_played)s,
                            %(captain_id)s, %(vice_captain_id)s, %(points_on_bench)s
                        )
                        ON CONFLICT (manager_id, event) DO UPDATE SET
                            gw_points = EXCLUDED.gw_points,
                            total_points = EXCLUDED.total_points,
                            overall_rank = EXCLUDED.overall_rank,
                            event_rank = EXCLUDED.event_rank,
                            bank = EXCLUDED.bank,
                            team_value = EXCLUDED.team_value,
                            event_transfers = EXCLUDED.event_transfers,
                            event_transfers_cost = EXCLUDED.event_transfers_cost,
                            chip_played = EXCLUDED.chip_played,
                            captain_id = EXCLUDED.captain_id,
                            vice_captain_id = EXCLUDED.vice_captain_id,
                            points_on_bench = EXCLUDED.points_on_bench,
                            updated_at = NOW();
                        """,
                        rows,
                    )
            return len(rows)
        except Exception as e:
            self.logger.error(f"Error upserting manager_gameweeks: {str(e)}")
            return 0

    def replace_manager_picks(
        self, manager_id: int, event: int, picks: List[Dict[str, Any]]
    ) -> int:
        """Replace all picks for a manager/event with the provided rows."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM manager_picks WHERE manager_id = %s AND event = %s",
                        (manager_id, event),
                    )
                    if not picks:
                        return 0
                    cur.executemany(
                        """
                        INSERT INTO manager_picks (
                            manager_id, event, element_id, position,
                            is_captain, is_vice_captain, multiplier
                        ) VALUES (
                            %(manager_id)s, %(event)s, %(element_id)s, %(position)s,
                            %(is_captain)s, %(is_vice_captain)s, %(multiplier)s
                        )
                        ON CONFLICT (manager_id, event, element_id) DO UPDATE SET
                            position = EXCLUDED.position,
                            is_captain = EXCLUDED.is_captain,
                            is_vice_captain = EXCLUDED.is_vice_captain,
                            multiplier = EXCLUDED.multiplier,
                            added_at = NOW();
                        """,
                        picks,
                    )
            return len(picks)
        except Exception as e:
            self.logger.error(
                f"Error replacing manager_picks for manager {manager_id}, event {event}: {str(e)}"
            )
            return 0

    def replace_manager_transfers(
        self, manager_id: int, event: int, transfers: List[Dict[str, Any]]
    ) -> int:
        """Replace all transfers for a manager/event with the provided rows."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM manager_transfers WHERE manager_id = %s AND event = %s",
                        (manager_id, event),
                    )
                    if not transfers:
                        return 0
                    cur.executemany(
                        """
                        INSERT INTO manager_transfers (
                            manager_id, event, element_in, element_out,
                            purchase_price, selling_price
                        ) VALUES (
                            %(manager_id)s, %(event)s, %(element_in)s, %(element_out)s,
                            %(purchase_price)s, %(selling_price)s
                        );
                        """,
                        transfers,
                    )
            return len(transfers)
        except Exception as e:
            self.logger.error(
                f"Error replacing manager_transfers for manager {manager_id}, event {event}: {str(e)}"
            )
            return 0

    def upsert_manager_cohort(
        self,
        manager_id: int,
        cohort_name: str,
        event_from: int,
        rank_at_entry: Optional[int] = None,
        event_to: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insert or update manager cohort membership."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO manager_cohorts (
                            manager_id, cohort_name, event_from,
                            event_to, rank_at_entry, meta
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (manager_id, cohort_name, event_from) DO UPDATE SET
                            event_to = EXCLUDED.event_to,
                            rank_at_entry = EXCLUDED.rank_at_entry,
                            meta = COALESCE(EXCLUDED.meta, manager_cohorts.meta),
                            updated_at = NOW();
                        """,
                        (
                            manager_id,
                            cohort_name,
                            event_from,
                            event_to,
                            rank_at_entry,
                            Json(meta) if meta is not None else None,
                        ),
                    )
            return True
        except Exception as e:
            self.logger.error(
                f"Error upserting manager_cohort for manager {manager_id}, cohort {cohort_name}: {str(e)}"
            )
            return False

    def get_top_managers_in_league(
        self, league_id: int, rank_limit: int
    ) -> List[Dict[str, Any]]:
        """Get managers in a league up to a given league_rank."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT m.manager_id,
                               m.player_name,
                               m.team_name,
                               l.league_rank,
                               l.league_points,
                               m.overall_rank
                        FROM manager_league_memberships l
                        JOIN fpl_managers m ON l.manager_id = m.manager_id
                        WHERE l.league_id = %s
                          AND l.league_rank IS NOT NULL
                          AND l.league_rank <= %s
                        ORDER BY l.league_rank ASC;
                        """,
                        (league_id, rank_limit),
                    )
                    return cur.fetchall()
        except Exception as e:
            self.logger.error(
                f"Error getting top managers for league {league_id} up to rank {rank_limit}: {str(e)}"
            )
            return []

    def fix_league_status(self, league_id: int, status: str) -> int:
        """Force-update crawl status for a league in crawl_progress.

        Returns the number of rows updated.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE crawl_progress
                        SET status = %s,
                            completed_at = NOW()
                        WHERE crawl_type = 'league'
                          AND metadata ->> 'league_id' = %s
                          AND status IN ('running', 'paused', 'stopped', 'failed');
                        """,
                        (status, str(league_id)),
                    )
                    return cur.rowcount
        except Exception as e:
            self.logger.error(f"Error fixing league status for {league_id}: {str(e)}")
            return 0

    # ==========================================
    # FPL DATA METHODS
    # ==========================================
    
    def upsert_teams(self, teams_df) -> int:
        """Upsert teams data"""
        if teams_df is None or len(teams_df) == 0:
            return 0
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    teams_data = teams_df.to_dict('records')
                    cur.executemany("""
                        INSERT INTO teams (
                            id, name, short_name, strength,
                            strength_overall_home, strength_overall_away,
                            strength_attack_home, strength_attack_away,
                            strength_defence_home, strength_defence_away,
                            pulse_id
                        ) VALUES (
                            %(id)s, %(name)s, %(short_name)s, %(strength)s,
                            %(strength_overall_home)s, %(strength_overall_away)s,
                            %(strength_attack_home)s, %(strength_attack_away)s,
                            %(strength_defence_home)s, %(strength_defence_away)s,
                            %(pulse_id)s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            short_name = EXCLUDED.short_name,
                            strength = EXCLUDED.strength,
                            strength_overall_home = EXCLUDED.strength_overall_home,
                            strength_overall_away = EXCLUDED.strength_overall_away,
                            strength_attack_home = EXCLUDED.strength_attack_home,
                            strength_attack_away = EXCLUDED.strength_attack_away,
                            strength_defence_home = EXCLUDED.strength_defence_home,
                            strength_defence_away = EXCLUDED.strength_defence_away,
                            pulse_id = EXCLUDED.pulse_id,
                            updated_at = NOW();
                    """, teams_data)
                    return len(teams_data)
        except Exception as e:
            self.logger.error(f"Error upserting teams: {str(e)}")
            return 0
    
    def upsert_players(self, players_df) -> int:
        """Upsert players data"""
        if players_df is None or len(players_df) == 0:
            return 0
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    players_data = players_df.to_dict('records')
                    cur.executemany("""
                        INSERT INTO players (
                            id, web_name, first_name, second_name, team_id, element_type,
                            now_cost, cost_change_start, cost_change_event, total_points,
                            points_per_game, selected_by_percent, form,
                            transfers_in, transfers_out, transfers_in_event, transfers_out_event,
                            minutes, goals_scored, assists, clean_sheets, goals_conceded,
                            own_goals, penalties_saved, penalties_missed, yellow_cards, red_cards,
                            saves, bonus, bps, influence, creativity, threat, ict_index,
                            expected_goals, expected_assists, expected_goal_involvements,
                            expected_goals_conceded, starts, news, news_added,
                            chance_of_playing_this_round, chance_of_playing_next_round, status
                        ) VALUES (
                            %(id)s, %(web_name)s, %(first_name)s, %(second_name)s, %(team)s, %(element_type)s,
                            %(now_cost)s, %(cost_change_start)s, %(cost_change_event)s, %(total_points)s,
                            %(points_per_game)s, %(selected_by_percent)s, %(form)s,
                            %(transfers_in)s, %(transfers_out)s, %(transfers_in_event)s, %(transfers_out_event)s,
                            %(minutes)s, %(goals_scored)s, %(assists)s, %(clean_sheets)s, %(goals_conceded)s,
                            %(own_goals)s, %(penalties_saved)s, %(penalties_missed)s, %(yellow_cards)s, %(red_cards)s,
                            %(saves)s, %(bonus)s, %(bps)s, %(influence)s, %(creativity)s, %(threat)s, %(ict_index)s,
                            %(expected_goals)s, %(expected_assists)s, %(expected_goal_involvements)s,
                            %(expected_goals_conceded)s, %(starts)s, %(news)s, %(news_added)s,
                            %(chance_of_playing_this_round)s, %(chance_of_playing_next_round)s, %(status)s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            web_name = EXCLUDED.web_name,
                            team_id = EXCLUDED.team_id,
                            now_cost = EXCLUDED.now_cost,
                            cost_change_event = EXCLUDED.cost_change_event,
                            total_points = EXCLUDED.total_points,
                            points_per_game = EXCLUDED.points_per_game,
                            selected_by_percent = EXCLUDED.selected_by_percent,
                            form = EXCLUDED.form,
                            transfers_in = EXCLUDED.transfers_in,
                            transfers_out = EXCLUDED.transfers_out,
                            transfers_in_event = EXCLUDED.transfers_in_event,
                            transfers_out_event = EXCLUDED.transfers_out_event,
                            minutes = EXCLUDED.minutes,
                            goals_scored = EXCLUDED.goals_scored,
                            assists = EXCLUDED.assists,
                            clean_sheets = EXCLUDED.clean_sheets,
                            goals_conceded = EXCLUDED.goals_conceded,
                            expected_goals = EXCLUDED.expected_goals,
                            expected_assists = EXCLUDED.expected_assists,
                            expected_goal_involvements = EXCLUDED.expected_goal_involvements,
                            expected_goals_conceded = EXCLUDED.expected_goals_conceded,
                            news = EXCLUDED.news,
                            news_added = EXCLUDED.news_added,
                            chance_of_playing_this_round = EXCLUDED.chance_of_playing_this_round,
                            chance_of_playing_next_round = EXCLUDED.chance_of_playing_next_round,
                            status = EXCLUDED.status,
                            updated_at = NOW();
                    """, players_data)
                    return len(players_data)
        except Exception as e:
            self.logger.error(f"Error upserting players: {str(e)}")
            return 0
    
    def upsert_gameweeks(self, gameweeks_df) -> int:
        """Upsert gameweeks data"""
        if gameweeks_df is None or len(gameweeks_df) == 0:
            return 0
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    gameweeks_data = gameweeks_df.to_dict('records')
                    cur.executemany("""
                        INSERT INTO gameweeks (
                            id, name, deadline_time, finished, is_current, is_next, is_previous,
                            average_entry_score, highest_score, most_selected, most_transferred_in,
                            most_captained, most_vice_captained, top_element
                        ) VALUES (
                            %(id)s, %(name)s, %(deadline_time)s, %(finished)s, %(is_current)s, %(is_next)s, %(is_previous)s,
                            %(average_entry_score)s, %(highest_score)s, %(most_selected)s, %(most_transferred_in)s,
                            %(most_captained)s, %(most_vice_captained)s, %(top_element)s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            deadline_time = EXCLUDED.deadline_time,
                            finished = EXCLUDED.finished,
                            is_current = EXCLUDED.is_current,
                            is_next = EXCLUDED.is_next,
                            is_previous = EXCLUDED.is_previous,
                            average_entry_score = EXCLUDED.average_entry_score,
                            highest_score = EXCLUDED.highest_score,
                            most_selected = EXCLUDED.most_selected,
                            most_transferred_in = EXCLUDED.most_transferred_in,
                            most_captained = EXCLUDED.most_captained,
                            most_vice_captained = EXCLUDED.most_vice_captained,
                            top_element = EXCLUDED.top_element,
                            updated_at = NOW();
                    """, gameweeks_data)
                    return len(gameweeks_data)
        except Exception as e:
            self.logger.error(f"Error upserting gameweeks: {str(e)}")
            return 0
    
    def upsert_fixtures(self, fixtures_df) -> int:
        """Upsert fixtures data"""
        if fixtures_df is None or len(fixtures_df) == 0:
            return 0
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    fixtures_data = fixtures_df.to_dict('records')
                    cur.executemany("""
                        INSERT INTO fixtures (
                            id, event, team_h, team_a, team_h_score, team_a_score,
                            kickoff_time, finished, finished_provisional, started,
                            team_h_difficulty, team_a_difficulty, pulse_id
                        ) VALUES (
                            %(id)s, %(event)s, %(team_h)s, %(team_a)s, %(team_h_score)s, %(team_a_score)s,
                            %(kickoff_time)s, %(finished)s, %(finished_provisional)s, %(started)s,
                            %(team_h_difficulty)s, %(team_a_difficulty)s, %(pulse_id)s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            team_h_score = EXCLUDED.team_h_score,
                            team_a_score = EXCLUDED.team_a_score,
                            kickoff_time = EXCLUDED.kickoff_time,
                            finished = EXCLUDED.finished,
                            finished_provisional = EXCLUDED.finished_provisional,
                            started = EXCLUDED.started,
                            team_h_difficulty = EXCLUDED.team_h_difficulty,
                            team_a_difficulty = EXCLUDED.team_a_difficulty,
                            updated_at = NOW();
                    """, fixtures_data)
                    return len(fixtures_data)
        except Exception as e:
            self.logger.error(f"Error upserting fixtures: {str(e)}")
            return 0
    
    def upsert_player_history(self, history_df) -> int:
        """Upsert player history data"""
        if history_df is None or len(history_df) == 0:
            return 0
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    history_data = history_df.to_dict('records')
                    cur.executemany("""
                        INSERT INTO player_history (
                            player_id, event, fixture, opponent_team, total_points, was_home,
                            minutes, goals_scored, assists, clean_sheets, goals_conceded,
                            own_goals, penalties_saved, penalties_missed, yellow_cards, red_cards,
                            saves, bonus, bps, influence, creativity, threat, ict_index,
                            value, transfers_balance, selected, transfers_in, transfers_out,
                            expected_goals, expected_assists, expected_goal_involvements,
                            expected_goals_conceded
                        ) VALUES (
                            %(player_id)s, %(round)s, %(fixture)s, %(opponent_team)s, %(total_points)s, %(was_home)s,
                            %(minutes)s, %(goals_scored)s, %(assists)s, %(clean_sheets)s, %(goals_conceded)s,
                            %(own_goals)s, %(penalties_saved)s, %(penalties_missed)s, %(yellow_cards)s, %(red_cards)s,
                            %(saves)s, %(bonus)s, %(bps)s, %(influence)s, %(creativity)s, %(threat)s, %(ict_index)s,
                            %(value)s, %(transfers_balance)s, %(selected)s, %(transfers_in)s, %(transfers_out)s,
                            %(expected_goals)s, %(expected_assists)s, %(expected_goal_involvements)s,
                            %(expected_goals_conceded)s
                        )
                        ON CONFLICT (player_id, event) DO UPDATE SET
                            total_points = EXCLUDED.total_points,
                            minutes = EXCLUDED.minutes,
                            goals_scored = EXCLUDED.goals_scored,
                            assists = EXCLUDED.assists,
                            bonus = EXCLUDED.bonus,
                            bps = EXCLUDED.bps,
                            expected_goals = EXCLUDED.expected_goals,
                            expected_assists = EXCLUDED.expected_assists;
                    """, history_data)
                    return len(history_data)
        except Exception as e:
            self.logger.error(f"Error upserting player history: {str(e)}")
            return 0
    
    def get_players_with_stats(self):
        """Get all players with their stats"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT p.*, t.name as team_name, t.short_name as team_short_name
                        FROM players p
                        LEFT JOIN teams t ON p.team_id = t.id
                        ORDER BY p.total_points DESC;
                    """)
                    rows = cur.fetchall()
                    # Convert to DataFrame for compatibility with existing API code
                    import pandas as pd
                    return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching players with stats: {str(e)}")
            import pandas as pd
            return pd.DataFrame()
    
    def get_current_gameweek(self) -> Optional[int]:
        """Get current gameweek number"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM gameweeks WHERE is_current = TRUE LIMIT 1;")
                    result = cur.fetchone()
                    return result['id'] if result else None
        except Exception as e:
            self.logger.error(f"Error fetching current gameweek: {str(e)}")
            return None
    
    def get_player_history_data(self, player_id: int = None, limit_gws: int = 10):
        """Get player historical performance data"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if player_id:
                        cur.execute("""
                            SELECT * FROM player_history
                            WHERE player_id = %s
                            ORDER BY event DESC
                            LIMIT %s;
                        """, (player_id, limit_gws))
                    else:
                        cur.execute("""
                            SELECT * FROM player_history
                            ORDER BY event DESC, player_id
                            LIMIT %s;
                        """, (limit_gws * 100,))
                    return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error fetching player history: {str(e)}")
            return []
    
    def save_predictions(self, predictions_df, gameweek: int, model_version: str) -> int:
        """Save model predictions"""
        if predictions_df is None or len(predictions_df) == 0:
            return 0
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    predictions_data = predictions_df.to_dict('records')
                    for pred in predictions_data:
                        pred['gameweek'] = gameweek
                        pred['model_version'] = model_version
                    
                    cur.executemany("""
                        INSERT INTO predictions (
                            player_id, gameweek, predicted_points, predicted_minutes,
                            predicted_goals, predicted_assists, predicted_clean_sheet_prob,
                            confidence_score, lower_bound, upper_bound, model_version
                        ) VALUES (
                            %(player_id)s, %(gameweek)s, %(predicted_points)s, %(predicted_minutes)s,
                            %(predicted_goals)s, %(predicted_assists)s, %(predicted_clean_sheet_prob)s,
                            %(confidence_score)s, %(lower_bound)s, %(upper_bound)s, %(model_version)s
                        )
                        ON CONFLICT (player_id, gameweek) DO UPDATE SET
                            predicted_points = EXCLUDED.predicted_points,
                            predicted_minutes = EXCLUDED.predicted_minutes,
                            predicted_goals = EXCLUDED.predicted_goals,
                            predicted_assists = EXCLUDED.predicted_assists,
                            predicted_clean_sheet_prob = EXCLUDED.predicted_clean_sheet_prob,
                            confidence_score = EXCLUDED.confidence_score,
                            lower_bound = EXCLUDED.lower_bound,
                            upper_bound = EXCLUDED.upper_bound,
                            model_version = EXCLUDED.model_version,
                            created_at = NOW();
                    """, predictions_data)
                    return len(predictions_data)
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            return 0
    
    def save_model_metrics(self, model_name: str, version: str, train_start_gw: int,
                          train_end_gw: int, validation_gw: int, test_gw: int,
                          metrics: Dict[str, float], artifact_path: Optional[str] = None) -> bool:
        """Save model evaluation metrics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    metadata = {k: v for k, v in metrics.items() if k not in ['mae', 'rmse', 'r2_score']}
                    cur.execute("""
                        INSERT INTO model_metrics (
                            model_name, version, train_start_gw, train_end_gw,
                            validation_gw, test_gw, mae, rmse, r2_score,
                            artifact_path, metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        );
                    """, (
                        model_name, version, train_start_gw, train_end_gw,
                        validation_gw, test_gw,
                        metrics.get('mae'), metrics.get('rmse'), metrics.get('r2_score'),
                        artifact_path, Json(metadata) if metadata else None
                    ))
                    return True
        except Exception as e:
            self.logger.error(f"Error saving model metrics: {str(e)}")
            return False
    
    def get_latest_model_metrics(self, model_name: Optional[str] = None):
        """Get latest metrics for models"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if model_name:
                        cur.execute("""
                            SELECT * FROM model_metrics
                            WHERE model_name = %s
                            ORDER BY created_at DESC
                            LIMIT 1;
                        """, (model_name,))
                        return cur.fetchone()
                    else:
                        cur.execute("""
                            SELECT DISTINCT ON (model_name) *
                            FROM model_metrics
                            ORDER BY model_name, created_at DESC;
                        """)
                        return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error fetching model metrics: {str(e)}")
            return None if model_name else []
    
    # ==========================================
    # COMPATIBILITY ALIASES FOR FPLDataCollector
    # ==========================================
    
    # Alias update_* methods to upsert_* for compatibility
    def update_teams(self, teams_df):
        """Alias for upsert_teams"""
        return self.upsert_teams(teams_df)
    
    def update_players(self, players_df):
        """Alias for upsert_players"""
        return self.upsert_players(players_df)
    
    def update_gameweeks(self, gameweeks_df):
        """Alias for upsert_gameweeks"""
        return self.upsert_gameweeks(gameweeks_df)
    
    def update_fixtures(self, fixtures_df):
        """Alias for upsert_fixtures"""
        return self.upsert_fixtures(fixtures_df)
    
    def update_player_history(self, history_df):
        """Alias for upsert_player_history"""
        return self.upsert_player_history(history_df)
    
    def get_trackable_player_ids(self, limit: int = None):
        """Get player IDs that should have trackable history"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT id FROM players
                        WHERE total_points > 0 OR minutes > 0
                        ORDER BY total_points DESC
                    """
                    if limit:
                        query += f" LIMIT {limit}"
                    cur.execute(query)
                    return [row['id'] for row in cur.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting trackable player IDs: {str(e)}")
            return []
    
    def update_player_fixtures(self, fixtures_df):
        """Update player upcoming fixtures (not implemented for PostgreSQL)"""
        # This is a feature specific to SQLite, skip for PostgreSQL
        return 0
    
    def update_player_gameweek_stats(self, stats_df):
        """Update player gameweek stats (not implemented for PostgreSQL)"""
        # This is a feature specific to SQLite, skip for PostgreSQL
        return 0
    
    def upsert_fixture_odds(self, odds_records):
        """Upsert fixture odds data"""
        if not odds_records:
            return False
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany("""
                        INSERT INTO fixture_odds (
                            fixture_id, event, home_team, away_team, bookmaker, market,
                            home_implied_goals, away_implied_goals, raw_odds
                        ) VALUES (
                            %(fixture_id)s, %(event)s, %(home_team)s, %(away_team)s, 
                            %(bookmaker)s, %(market)s, %(home_implied_goals)s, 
                            %(away_implied_goals)s, %(raw_odds)s
                        )
                        ON CONFLICT (fixture_id, bookmaker, market) DO UPDATE SET
                            home_implied_goals = EXCLUDED.home_implied_goals,
                            away_implied_goals = EXCLUDED.away_implied_goals,
                            raw_odds = EXCLUDED.raw_odds,
                            updated_at = NOW();
                    """, odds_records)
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error upserting fixture odds: {str(e)}")
            return False
    
    def mark_last_update(self):
        """Mark the last data update time"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO data_updates (update_type, updated_at)
                        VALUES ('full_update', NOW())
                        ON CONFLICT (update_type) DO UPDATE SET
                            updated_at = NOW();
                    """)
                    conn.commit()
                    return True
        except Exception as e:
            # Table might not exist, that's okay
            self.logger.debug(f"Could not mark last update: {str(e)}")
            return False
    
    def refresh_player_meta_features(self, incremental: bool = True) -> bool:
        """Refresh materialized view for player meta features
        
        Args:
            incremental: Use CONCURRENTLY for non-blocking refresh (default: True)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if incremental:
                        # CONCURRENTLY allows reads during refresh but requires unique index
                        cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY player_meta_features")
                    else:
                        # Full refresh (blocks reads)
                        cur.execute("REFRESH MATERIALIZED VIEW player_meta_features")
                    
                    # Update cache metadata
                    cur.execute("""
                        INSERT INTO cache_metadata (cache_key, last_updated, record_count)
                        VALUES ('player_meta_features', NOW(), 
                                (SELECT COUNT(*) FROM player_meta_features))
                        ON CONFLICT (cache_key) DO UPDATE SET
                            last_updated = NOW(),
                            record_count = EXCLUDED.record_count
                    """)
                conn.commit()
                self.logger.info("✅ Refreshed player_meta_features materialized view")
                return True
        except Exception as e:
            self.logger.error(f"Error refreshing materialized view: {str(e)}")
            return False
    
    def get_cache_metadata(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cache metadata for a given key
        
        Args:
            cache_key: The cache key to look up
        
        Returns:
            Dict with cache metadata or None if not found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM cache_metadata WHERE cache_key = %s",
                        (cache_key,)
                    )
                    return cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error fetching cache metadata: {str(e)}")
            return None

    
    def get_fixtures(self, gameweek: int = None):
        """Get fixtures, optionally filtered by gameweek"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if gameweek:
                        cur.execute("""
                            SELECT * FROM fixtures
                            WHERE event = %s
                            ORDER BY kickoff_time;
                        """, (gameweek,))
                    else:
                        cur.execute("""
                            SELECT * FROM fixtures
                            ORDER BY event, kickoff_time;
                        """)
                    rows = cur.fetchall()
                    import pandas as pd
                    return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting fixtures: {str(e)}")
            import pandas as pd
            return pd.DataFrame()
    
    def get_teams(self):
        """Get all teams"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM teams ORDER BY id;")
                    rows = cur.fetchall()
                    import pandas as pd
                    return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting teams: {str(e)}")
            import pandas as pd
            return pd.DataFrame()
    
    # ==========================================
    # ADAPTIVE LEARNING METHODS
    # ==========================================
    
    def store_prediction_feedback(self, feedback_records: List[Dict[str, Any]]) -> bool:
        """Store prediction feedback for multiple players"""
        if not feedback_records:
            return True
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany("""
                        INSERT INTO prediction_feedback (
                            player_id, gameweek, predicted_points, actual_points,
                            prediction_error, absolute_error, squared_error,
                            model_version, player_position, player_price,
                            player_team, was_home
                        ) VALUES (
                            %(player_id)s, %(gameweek)s, %(predicted_points)s, %(actual_points)s,
                            %(prediction_error)s, %(absolute_error)s, %(squared_error)s,
                            %(model_version)s, %(player_position)s, %(player_price)s,
                            %(player_team)s, %(was_home)s
                        )
                        ON CONFLICT (player_id, gameweek, model_version) DO UPDATE SET
                            predicted_points = EXCLUDED.predicted_points,
                            actual_points = EXCLUDED.actual_points,
                            prediction_error = EXCLUDED.prediction_error,
                            absolute_error = EXCLUDED.absolute_error,
                            squared_error = EXCLUDED.squared_error;
                    """, feedback_records)
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error storing prediction feedback: {str(e)}")
            return False
    
    def get_prediction_accuracy(self, gameweek: int = None, model_version: str = None) -> Dict[str, Any]:
        """Get prediction accuracy metrics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    where_clauses = []
                    params = []
                    
                    if gameweek:
                        where_clauses.append("gameweek = %s")
                        params.append(gameweek)
                    if model_version:
                        where_clauses.append("model_version = %s")
                        params.append(model_version)
                    
                    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                    
                    cur.execute(f"""
                        SELECT 
                            COUNT(*) as total_predictions,
                            AVG(absolute_error) as mae,
                            SQRT(AVG(squared_error)) as rmse,
                            AVG(prediction_error) as mean_error,
                            STDDEV(prediction_error) as std_error,
                            MAX(absolute_error) as max_error,
                            MIN(absolute_error) as min_error
                        FROM prediction_feedback
                        {where_sql};
                    """, params)
                    
                    result = cur.fetchone()
                    return result if result else {}
        except Exception as e:
            self.logger.error(f"Error getting prediction accuracy: {str(e)}")
            return {}
    
    def get_prediction_accuracy_by_position(self, gameweek: int = None) -> List[Dict[str, Any]]:
        """Get prediction accuracy broken down by position"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    where_sql = "WHERE gameweek = %s" if gameweek else ""
                    params = [gameweek] if gameweek else []
                    
                    cur.execute(f"""
                        SELECT 
                            player_position,
                            COUNT(*) as count,
                            AVG(absolute_error) as mae,
                            SQRT(AVG(squared_error)) as rmse,
                            AVG(prediction_error) as mean_error
                        FROM prediction_feedback
                        {where_sql}
                        GROUP BY player_position
                        ORDER BY player_position;
                    """, params)
                    
                    return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error getting accuracy by position: {str(e)}")
            return []
    
    def store_transfer_outcome(self, outcome: Dict[str, Any]) -> bool:
        """Store a transfer outcome"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO transfer_outcomes (
                            user_id, gameweek, player_out_id, player_in_id,
                            predicted_points_gain, actual_points_out, actual_points_in,
                            actual_points_gain, transfer_cost, net_gain,
                            was_successful, was_recommended, recommendation_rank
                        ) VALUES (
                            %(user_id)s, %(gameweek)s, %(player_out_id)s, %(player_in_id)s,
                            %(predicted_points_gain)s, %(actual_points_out)s, %(actual_points_in)s,
                            %(actual_points_gain)s, %(transfer_cost)s, %(net_gain)s,
                            %(was_successful)s, %(was_recommended)s, %(recommendation_rank)s
                        );
                    """, outcome)
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error storing transfer outcome: {str(e)}")
            return False
    
    def get_transfer_success_rate(self, user_id: int = None, recent_gws: int = 5) -> Dict[str, Any]:
        """Get transfer success rate statistics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    where_clauses = []
                    params = []
                    
                    if user_id:
                        where_clauses.append("user_id = %s")
                        params.append(user_id)
                    
                    if recent_gws:
                        where_clauses.append("""
                            gameweek >= (SELECT MAX(id) FROM gameweeks) - %s
                        """)
                        params.append(recent_gws)
                    
                    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                    
                    cur.execute(f"""
                        SELECT 
                            COUNT(*) as total_transfers,
                            SUM(CASE WHEN was_successful THEN 1 ELSE 0 END) as successful_transfers,
                            AVG(CASE WHEN was_successful THEN 1.0 ELSE 0.0 END) as success_rate,
                            AVG(predicted_points_gain) as avg_predicted_gain,
                            AVG(actual_points_gain) as avg_actual_gain,
                            AVG(net_gain) as avg_net_gain,
                            SUM(net_gain) as total_net_gain
                        FROM transfer_outcomes
                        {where_sql};
                    """, params)
                    
                    result = cur.fetchone()
                    return result if result else {}
        except Exception as e:
            self.logger.error(f"Error getting transfer success rate: {str(e)}")
            return {}
    
    def track_user_decision(self, decision: Dict[str, Any]) -> bool:
        """Track a user decision (accept/reject recommendation)"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Prepare JSONB context
                    context = decision.get('context', {})
                    if context:
                        context = Json(context)
                    
                    cur.execute("""
                        INSERT INTO user_decisions (
                            user_id, gameweek, decision_type,
                            recommended_option_id, recommended_option_name,
                            actual_choice_id, actual_choice_name,
                            user_accepted, recommended_points, actual_points,
                            confidence_score, context
                        ) VALUES (
                            %(user_id)s, %(gameweek)s, %(decision_type)s,
                            %(recommended_option_id)s, %(recommended_option_name)s,
                            %(actual_choice_id)s, %(actual_choice_name)s,
                            %(user_accepted)s, %(recommended_points)s, %(actual_points)s,
                            %(confidence_score)s, %(context)s
                        );
                    """, {**decision, 'context': context})
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error tracking user decision: {str(e)}")
            return False
    
    def get_user_preferences(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user preferences"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT * FROM user_preferences WHERE user_id = %s;
                    """, (user_id,))
                    return cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error getting user preferences: {str(e)}")
            return None
    
    def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """Update or create user preferences"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Prepare JSONB fields
                    favorite_teams = preferences.get('favorite_teams')
                    avoided_teams = preferences.get('avoided_teams')
                    
                    if favorite_teams:
                        favorite_teams = Json(favorite_teams)
                    if avoided_teams:
                        avoided_teams = Json(avoided_teams)
                    
                    cur.execute("""
                        INSERT INTO user_preferences (
                            user_id, risk_tolerance, prefers_differentials,
                            differential_threshold, budget_strategy, preferred_formation,
                            favorite_teams, avoided_teams, captain_strategy,
                            transfer_aggressiveness, learned_from_decisions,
                            confidence_level
                        ) VALUES (
                            %(user_id)s, %(risk_tolerance)s, %(prefers_differentials)s,
                            %(differential_threshold)s, %(budget_strategy)s, %(preferred_formation)s,
                            %(favorite_teams)s, %(avoided_teams)s, %(captain_strategy)s,
                            %(transfer_aggressiveness)s, %(learned_from_decisions)s,
                            %(confidence_level)s
                        )
                        ON CONFLICT (user_id) DO UPDATE SET
                            risk_tolerance = EXCLUDED.risk_tolerance,
                            prefers_differentials = EXCLUDED.prefers_differentials,
                            differential_threshold = EXCLUDED.differential_threshold,
                            budget_strategy = EXCLUDED.budget_strategy,
                            preferred_formation = EXCLUDED.preferred_formation,
                            favorite_teams = EXCLUDED.favorite_teams,
                            avoided_teams = EXCLUDED.avoided_teams,
                            captain_strategy = EXCLUDED.captain_strategy,
                            transfer_aggressiveness = EXCLUDED.transfer_aggressiveness,
                            learned_from_decisions = EXCLUDED.learned_from_decisions,
                            confidence_level = EXCLUDED.confidence_level,
                            last_updated = NOW();
                    """, {
                        'user_id': user_id,
                        **preferences,
                        'favorite_teams': favorite_teams,
                        'avoided_teams': avoided_teams
                    })
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error updating user preferences: {str(e)}")
            return False
    
    def store_feature_importance(self, importance_records: List[Dict[str, Any]]) -> bool:
        """Store feature importance data"""
        if not importance_records:
            return True
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany("""
                        INSERT INTO feature_importance_history (
                            model_name, model_version, gameweek, feature_name,
                            importance_score, rank, feature_type
                        ) VALUES (
                            %(model_name)s, %(model_version)s, %(gameweek)s, %(feature_name)s,
                            %(importance_score)s, %(rank)s, %(feature_type)s
                        );
                    """, importance_records)
                    conn.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Error storing feature importance: {str(e)}")
            return False
    
    def get_feature_importance_trends(self, feature_name: str = None, recent_gws: int = 10) -> List[Dict[str, Any]]:
        """Get feature importance trends over time"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    where_clauses = []
                    params = []
                    
                    if feature_name:
                        where_clauses.append("feature_name = %s")
                        params.append(feature_name)
                    
                    if recent_gws:
                        where_clauses.append("""
                            gameweek >= (SELECT MAX(id) FROM gameweeks) - %s
                        """)
                        params.append(recent_gws)
                    
                    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                    
                    cur.execute(f"""
                        SELECT 
                            feature_name,
                            gameweek,
                            AVG(importance_score) as avg_importance,
                            AVG(rank) as avg_rank,
                            COUNT(*) as model_count
                        FROM feature_importance_history
                        {where_sql}
                        GROUP BY feature_name, gameweek
                        ORDER BY gameweek DESC, avg_importance DESC;
                    """, params)
                    
                    return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error getting feature importance trends: {str(e)}")
            return []

