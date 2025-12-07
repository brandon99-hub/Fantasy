"""Manager-level training dataset utilities built on Postgres manager tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

import pandas as pd

from backend.src.core.postgres_db import PostgresManagerDB

logger = logging.getLogger(__name__)


@dataclass
class ManagerCohortConfig:
    """Configuration for selecting manager cohorts."""

    league_id: int = 314
    rank_limit: int = 10000
    cohort_name: str = "top10k_overall"


class ManagerTrainingDataBuilder:
    """Builds manager-derived aggregate features from Postgres history tables.

    This is a read-only helper that never mutates state; it simply projects
    manager- and league-level behaviour into per-player, per-gameweek metrics.
    """

    def __init__(self, db: Optional[PostgresManagerDB] = None):
        self.db = db or PostgresManagerDB()

    def _load_manager_gameweeks(self) -> pd.DataFrame:
        logger.info("Loading manager gameweek data from database...")
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT manager_id, event,
                       gw_points, total_points, overall_rank,
                       event_transfers, event_transfers_cost, chip_played
                FROM manager_gameweeks
                """,
                conn,
            )
        logger.info(f"✅ Loaded {len(df):,} manager gameweek records")
        return df

    def build_player_meta_features(self, force_refresh: bool = False, use_materialized_view: bool = True) -> pd.DataFrame:
        """Aggregate manager picks into player-level meta features.
        
        Args:
            force_refresh: Force refresh of materialized view
            use_materialized_view: Use pre-computed materialized view (default: True)
        
        Returns a DataFrame with columns like:
          - player_id
          - event
          - top_cohort_ownership_pct
          - top_cohort_captain_pct
          - top_cohort_vice_captain_pct
        """
        logger.info("Building player meta features from manager data...")
        
        # Check if we should use materialized view
        if use_materialized_view:
            try:
                # Check cache freshness
                cache_info = self.db.get_cache_metadata('player_meta_features')
                
                # Refresh if forced or cache doesn't exist
                if force_refresh or cache_info is None:
                    logger.info("Refreshing materialized view...")
                    self.db.refresh_player_meta_features(incremental=not force_refresh)
                else:
                    logger.info(f"Using cached player meta features (last updated: {cache_info.get('last_updated')})")
                
                # Load from materialized view
                with self.db.get_connection() as conn:
                    df = pd.read_sql_query(
                        """
                        SELECT 
                            player_id,
                            event,
                            top_cohort_ownership_pct,
                            top_cohort_captain_pct,
                            top_cohort_vice_captain_pct
                        FROM player_meta_features
                        ORDER BY event, player_id
                        """,
                        conn
                    )
                
                logger.info(f"✅ Loaded {len(df):,} player-gameweek combinations from materialized view")
                return df
                
            except Exception as e:
                logger.warning(f"Failed to use materialized view: {str(e)}. Falling back to SQL aggregation.")
                use_materialized_view = False
        
        # Fallback: SQL-based aggregation (still much faster than pandas)
        if not use_materialized_view:
            logger.info("Using SQL-based aggregation...")
            with self.db.get_connection() as conn:
                df = pd.read_sql_query(
                    """
                    WITH manager_counts AS (
                        SELECT event, COUNT(DISTINCT manager_id) as manager_count
                        FROM manager_picks
                        GROUP BY event
                    )
                    SELECT 
                        mp.event,
                        mp.element_id as player_id,
                        (COUNT(DISTINCT mp.manager_id)::float / NULLIF(mc.manager_count, 0) * 100) as top_cohort_ownership_pct,
                        (SUM(CASE WHEN mp.is_captain THEN 1 ELSE 0 END)::float / NULLIF(mc.manager_count, 0) * 100) as top_cohort_captain_pct,
                        (SUM(CASE WHEN mp.is_vice_captain THEN 1 ELSE 0 END)::float / NULLIF(mc.manager_count, 0) * 100) as top_cohort_vice_captain_pct
                    FROM manager_picks mp
                    JOIN manager_counts mc ON mp.event = mc.event
                    GROUP BY mp.event, mp.element_id, mc.manager_count
                    ORDER BY mp.event, mp.element_id
                    """,
                    conn
                )
            
            logger.info(f"✅ Built meta features for {len(df):,} player-gameweek combinations via SQL")
            return df



