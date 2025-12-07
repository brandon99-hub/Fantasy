"""
Expected Goals (xG) and Expected Assists (xA) integration
Fetches xG data from Understat and integrates with player predictions
"""

import requests
import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class XGIntegrator:
    """Integrate xG/xA data from Understat"""
    
    BASE_URL = "https://understat.com"
    
    # Team name mapping (FPL -> Understat)
    TEAM_MAPPING = {
        'Arsenal': 'Arsenal',
        'Aston Villa': 'Aston_Villa',
        'Bournemouth': 'Bournemouth',
        'Brentford': 'Brentford',
        'Brighton': 'Brighton',
        'Chelsea': 'Chelsea',
        'Crystal Palace': 'Crystal_Palace',
        'Everton': 'Everton',
        'Fulham': 'Fulham',
        'Liverpool': 'Liverpool',
        'Luton': 'Luton',
        'Man City': 'Manchester_City',
        'Man Utd': 'Manchester_United',
        'Newcastle': 'Newcastle_United',
        'Nott\'m Forest': 'Nottingham_Forest',
        'Sheffield Utd': 'Sheffield_United',
        'Spurs': 'Tottenham',
        'West Ham': 'West_Ham',
        'Wolves': 'Wolverhampton_Wanderers'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_player_xg_data(self, player_name: str, team_name: str) -> Optional[Dict]:
        """
        Fetch xG data for a specific player
        
        Args:
            player_name: Player's name
            team_name: Team name (FPL format)
        
        Returns:
            Dict with xG/xA stats or None
        """
        try:
            # This is a placeholder - actual implementation would scrape Understat
            # or use their API if available
            
            # For now, return mock data structure
            return {
                'player_name': player_name,
                'team': team_name,
                'xG': 0.0,
                'xA': 0.0,
                'xG_per_90': 0.0,
                'xA_per_90': 0.0,
                'xG_overperformance': 0.0,  # Actual goals - xG
                'xA_overperformance': 0.0,  # Actual assists - xA
                'shots': 0,
                'key_passes': 0,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching xG data for {player_name}: {e}")
            return None
    
    def enrich_players_with_xg(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich players DataFrame with xG/xA data
        
        Args:
            players_df: Players DataFrame
        
        Returns:
            Enhanced DataFrame with xG columns
        """
        self.logger.info("Enriching players with xG data...")
        
        # Add xG columns
        xg_columns = {
            'xG': 0.0,
            'xA': 0.0,
            'xG_per_90': 0.0,
            'xA_per_90': 0.0,
            'xG_overperformance': 0.0,
            'xA_overperformance': 0.0,
            'xGI': 0.0,  # xG + xA (expected goal involvement)
            'xGI_per_90': 0.0
        }
        
        for col, default_val in xg_columns.items():
            if col not in players_df.columns:
                players_df[col] = default_val
        
        # TODO: Implement actual data fetching
        # For now, calculate estimated xG from existing stats
        players_df = self._estimate_xg_from_stats(players_df)
        
        return players_df
    
    def _estimate_xg_from_stats(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate xG/xA from existing FPL stats
        This is a rough approximation until real xG data is integrated
        """
        # Estimate xG (goals are roughly 1.2x xG for good finishers)
        if 'goals_scored' in players_df.columns:
            players_df['xG'] = players_df['goals_scored'] * 0.85
        
        # Estimate xA (assists are roughly 1.1x xA)
        if 'assists' in players_df.columns:
            players_df['xA'] = players_df['assists'] * 0.9
        
        # Calculate per 90
        if 'minutes' in players_df.columns:
            minutes = players_df['minutes'].replace(0, 1)  # Avoid division by zero
            players_df['xG_per_90'] = (players_df['xG'] / minutes) * 90
            players_df['xA_per_90'] = (players_df['xA'] / minutes) * 90
        
        # Calculate xGI (expected goal involvement)
        players_df['xGI'] = players_df['xG'] + players_df['xA']
        players_df['xGI_per_90'] = players_df['xG_per_90'] + players_df['xA_per_90']
        
        # Calculate over/underperformance
        if 'goals_scored' in players_df.columns:
            players_df['xG_overperformance'] = players_df['goals_scored'] - players_df['xG']
        if 'assists' in players_df.columns:
            players_df['xA_overperformance'] = players_df['assists'] - players_df['xA']
        
        return players_df
    
    def calculate_xg_features(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate xG-based features for ML models
        
        Args:
            players_df: Players DataFrame with xG data
        
        Returns:
            DataFrame with additional xG features
        """
        # xG efficiency (goals per xG)
        players_df['xG_efficiency'] = players_df.apply(
            lambda x: x['goals_scored'] / x['xG'] if x['xG'] > 0 else 1.0,
            axis=1
        )
        
        # xA efficiency
        players_df['xA_efficiency'] = players_df.apply(
            lambda x: x['assists'] / x['xA'] if x['xA'] > 0 else 1.0,
            axis=1
        )
        
        # Quality of chances (xG per shot)
        if 'shots' in players_df.columns:
            players_df['xG_per_shot'] = players_df.apply(
                lambda x: x['xG'] / x['shots'] if x.get('shots', 0) > 0 else 0.0,
                axis=1
            )
        
        # Expected points from xG/xA
        # Goals = 4-6 pts, Assists = 3 pts
        players_df['expected_points_from_xGI'] = (
            players_df['xG'] * 5 +  # Average points per goal
            players_df['xA'] * 3    # Points per assist
        )
        
        return players_df
    
    def identify_xg_overperformers(
        self,
        players_df: pd.DataFrame,
        threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Identify players significantly over/underperforming xG
        
        Args:
            players_df: Players DataFrame with xG data
            threshold: Minimum difference to be considered over/underperformer
        
        Returns:
            DataFrame of over/underperformers
        """
        # Overperformers (scoring more than expected)
        overperformers = players_df[
            players_df['xG_overperformance'] >= threshold
        ].copy()
        
        # Underperformers (scoring less than expected)
        underperformers = players_df[
            players_df['xG_overperformance'] <= -threshold
        ].copy()
        
        return {
            'overperformers': overperformers.sort_values('xG_overperformance', ascending=False),
            'underperformers': underperformers.sort_values('xG_overperformance')
        }
    
    def get_xg_summary(self, players_df: pd.DataFrame) -> Dict:
        """Get summary statistics for xG data"""
        return {
            'total_xG': float(players_df['xG'].sum()),
            'total_xA': float(players_df['xA'].sum()),
            'avg_xG_per_90': float(players_df['xG_per_90'].mean()),
            'avg_xA_per_90': float(players_df['xA_per_90'].mean()),
            'top_xG_players': players_df.nlargest(10, 'xG')[['web_name', 'xG', 'goals_scored']].to_dict('records'),
            'top_xA_players': players_df.nlargest(10, 'xA')[['web_name', 'xA', 'assists']].to_dict('records'),
            'biggest_overperformers': players_df.nlargest(5, 'xG_overperformance')[['web_name', 'xG_overperformance']].to_dict('records'),
            'biggest_underperformers': players_df.nsmallest(5, 'xG_overperformance')[['web_name', 'xG_overperformance']].to_dict('records')
        }
