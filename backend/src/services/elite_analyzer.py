"""
Elite Manager Analyzer Service

Analyzes patterns from top-performing managers to identify winning strategies.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd

from backend.src.core.postgres_db import PostgresManagerDB


logger = logging.getLogger(__name__)


class EliteAnalyzer:
    """Analyze elite manager patterns for insights"""
    
    def __init__(self):
        self.postgres_db = PostgresManagerDB()
        self.logger = logging.getLogger(__name__)
    
    def get_elite_transfers(self, gameweek: int, cohort: str = 'top10k_overall', min_count: int = 50) -> List[Dict[str, Any]]:
        """
        Get most popular transfers among elite managers
        
        Args:
            gameweek: Gameweek to analyze
            cohort: Manager cohort (e.g., 'top10k_overall')
            min_count: Minimum number of managers making the transfer
        
        Returns:
            List of popular transfers with counts
        """
        try:
            with self.postgres_db.get_connection() as conn:
                query = """
                    WITH elite_managers AS (
                        SELECT DISTINCT manager_id
                        FROM manager_cohorts
                        WHERE cohort_name = %s
                          AND event_from <= %s
                          AND (event_to IS NULL OR event_to >= %s)
                    )
                    SELECT 
                        mt.element_in as player_id,
                        p.web_name,
                        p.team_id,
                        t.name as team_name,
                        p.element_type as position,
                        p.now_cost as price,
                        COUNT(*) as transfer_count,
                        COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT manager_id) FROM elite_managers) as elite_ownership_pct
                    FROM manager_transfers mt
                    JOIN elite_managers em ON mt.manager_id = em.manager_id
                    JOIN players p ON mt.element_in = p.id
                    JOIN teams t ON p.team_id = t.id
                    WHERE mt.event = %s
                    GROUP BY mt.element_in, p.web_name, p.team_id, t.name, p.element_type, p.now_cost
                    HAVING COUNT(*) >= %s
                    ORDER BY transfer_count DESC
                    LIMIT 20;
                """
                
                return pd.read_sql(query, conn, params=(cohort, gameweek, gameweek, gameweek, min_count)).to_dict('records')
                
        except Exception as e:
            self.logger.error(f"Error getting elite transfers: {str(e)}")
            return []
    
    def get_elite_captains(self, gameweek: int, cohort: str = 'top10k_overall') -> List[Dict[str, Any]]:
        """Get most popular captain choices among elite managers"""
        try:
            with self.postgres_db.get_connection() as conn:
                query = """
                    WITH elite_managers AS (
                        SELECT DISTINCT manager_id
                        FROM manager_cohorts
                        WHERE cohort_name = %s
                          AND event_from <= %s
                          AND (event_to IS NULL OR event_to >= %s)
                    )
                    SELECT 
                        mp.element_id as player_id,
                        p.web_name,
                        p.team_id,
                        t.name as team_name,
                        COUNT(*) as captain_count,
                        COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT manager_id) FROM elite_managers) as elite_captain_pct,
                        p.selected_by_percent as overall_ownership
                    FROM manager_picks mp
                    JOIN elite_managers em ON mp.manager_id = em.manager_id
                    JOIN players p ON mp.element_id = p.id
                    JOIN teams t ON p.team_id = t.id
                    WHERE mp.event = %s
                      AND mp.is_captain = TRUE
                    GROUP BY mp.element_id, p.web_name, p.team_id, t.name, p.selected_by_percent
                    ORDER BY captain_count DESC
                    LIMIT 10;
                """
                
                return pd.read_sql(query, conn, params=(cohort, gameweek, gameweek, gameweek)).to_dict('records')
                
        except Exception as e:
            self.logger.error(f"Error getting elite captains: {str(e)}")
            return []
    
    def get_elite_differentials(self, gameweek: int, cohort: str = 'top10k_overall', ownership_threshold: float = 10.0) -> List[Dict[str, Any]]:
        """
        Identify differential picks (low overall ownership, high elite ownership)
        
        Args:
            gameweek: Gameweek to analyze
            cohort: Manager cohort
            ownership_threshold: Max overall ownership to be considered differential
        
        Returns:
            List of differential picks
        """
        try:
            with self.postgres_db.get_connection() as conn:
                query = """
                    WITH elite_managers AS (
                        SELECT DISTINCT manager_id
                        FROM manager_cohorts
                        WHERE cohort_name = %s
                          AND event_from <= %s
                          AND (event_to IS NULL OR event_to >= %s)
                    )
                    SELECT 
                        mp.element_id as player_id,
                        p.web_name,
                        p.team_id,
                        t.name as team_name,
                        p.element_type as position,
                        p.now_cost as price,
                        COUNT(DISTINCT mp.manager_id) as elite_picks,
                        COUNT(DISTINCT mp.manager_id) * 100.0 / (SELECT COUNT(DISTINCT manager_id) FROM elite_managers) as elite_ownership_pct,
                        p.selected_by_percent as overall_ownership,
                        (COUNT(DISTINCT mp.manager_id) * 100.0 / (SELECT COUNT(DISTINCT manager_id) FROM elite_managers)) - p.selected_by_percent as differential_score
                    FROM manager_picks mp
                    JOIN elite_managers em ON mp.manager_id = em.manager_id
                    JOIN players p ON mp.element_id = p.id
                    JOIN teams t ON p.team_id = t.id
                    WHERE mp.event = %s
                      AND p.selected_by_percent < %s
                    GROUP BY mp.element_id, p.web_name, p.team_id, t.name, p.element_type, p.now_cost, p.selected_by_percent
                    HAVING COUNT(DISTINCT mp.manager_id) * 100.0 / (SELECT COUNT(DISTINCT manager_id) FROM elite_managers) > 15
                    ORDER BY differential_score DESC
                    LIMIT 15;
                """
                
                return pd.read_sql(query, conn, params=(cohort, gameweek, gameweek, gameweek, ownership_threshold)).to_dict('records')
                
        except Exception as e:
            self.logger.error(f"Error getting elite differentials: {str(e)}")
            return []
    
    def analyze_chip_timing(self, cohort: str = 'top10k_overall') -> Dict[str, Any]:
        """Analyze when elite managers use chips"""
        try:
            with self.postgres_db.get_connection() as conn:
                query = """
                    WITH elite_managers AS (
                        SELECT DISTINCT manager_id
                        FROM manager_cohorts
                        WHERE cohort_name = %s
                    )
                    SELECT 
                        chip_played,
                        event,
                        COUNT(*) as usage_count,
                        COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT manager_id) FROM elite_managers) as usage_pct
                    FROM manager_gameweeks mg
                    JOIN elite_managers em ON mg.manager_id = em.manager_id
                    WHERE chip_played IS NOT NULL
                      AND chip_played != ''
                    GROUP BY chip_played, event
                    ORDER BY chip_played, event;
                """
                
                results = pd.read_sql(query, conn, params=(cohort,))
                
                # Group by chip type
                chip_timing = {}
                for _, row in results.iterrows():
                    chip = row['chip_played']
                    if chip not in chip_timing:
                        chip_timing[chip] = []
                    chip_timing[chip].append({
                        'gameweek': row['event'],
                        'usage_count': row['usage_count'],
                        'usage_pct': float(row['usage_pct'])
                    })
                
                return chip_timing
                
        except Exception as e:
            self.logger.error(f"Error analyzing chip timing: {str(e)}")
            return {}
