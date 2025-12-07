"""
xG Estimator - Fallback estimation when external sources unavailable

Estimates expected goals and assists from FPL statistics using
empirical formulas based on shots, chances, and historical data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)


class XGEstimator:
    """Estimate xG/xA from FPL stats when external sources unavailable"""
    
    # Position-specific conversion rates (empirically derived)
    POSITION_CONVERSION_RATES = {
        'FWD': {'shot_conversion': 0.35, 'big_chance_conversion': 0.45},
        'MID': {'shot_conversion': 0.30, 'big_chance_conversion': 0.40},
        'DEF': {'shot_conversion': 0.25, 'big_chance_conversion': 0.35},
        'GKP': {'shot_conversion': 0.10, 'big_chance_conversion': 0.20}
    }
    
    def __init__(self):
        logger.info("xG Estimator initialized")
    
    def estimate_xg(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate xG for players based on FPL stats
        
        Args:
            players_df: DataFrame with FPL player stats
        
        Returns:
            DataFrame with estimated xG columns added
        """
        df = players_df.copy()
        
        # Ensure required columns exist
        required_cols = ['position', 'total_points', 'minutes']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for xG estimation")
            return df
        
        # Estimate xG
        df['estimated_xg'] = df.apply(self._estimate_player_xg, axis=1)
        df['estimated_xa'] = df.apply(self._estimate_player_xa, axis=1)
        
        # Calculate per 90
        df['estimated_xg_per_90'] = df.apply(
            lambda row: (row['estimated_xg'] / row['minutes'] * 90) if row['minutes'] > 0 else 0,
            axis=1
        )
        df['estimated_xa_per_90'] = df.apply(
            lambda row: (row['estimated_xa'] / row['minutes'] * 90) if row['minutes'] > 0 else 0,
            axis=1
        )
        
        # Round to 2 decimals
        df['estimated_xg'] = df['estimated_xg'].round(2)
        df['estimated_xa'] = df['estimated_xa'].round(2)
        df['estimated_xg_per_90'] = df['estimated_xg_per_90'].round(2)
        df['estimated_xa_per_90'] = df['estimated_xa_per_90'].round(2)
        
        # Add source marker
        df['xg_source'] = 'estimated'
        
        logger.info(f"Estimated xG for {len(df)} players")
        
        return df
    
    def _estimate_player_xg(self, player: pd.Series) -> float:
        """
        Estimate xG for a single player
        
        Formula based on:
        - Goals scored (actual conversion)
        - Shots on target (proxy for quality chances)
        - Bonus points (proxy for overall attacking threat)
        - Position-specific conversion rates
        """
        position = player.get('position', 'MID')
        conversion_rates = self.POSITION_CONVERSION_RATES.get(
            position, 
            self.POSITION_CONVERSION_RATES['MID']
        )
        
        # Base xG from goals (actual goals are evidence of chances)
        goals = player.get('goals_scored', 0)
        
        # Estimate additional xG from non-converted chances
        # Assume each goal came from ~2.5 chances on average
        estimated_chances = goals * 2.5
        
        # Add estimate from bonus points (proxy for attacking involvement)
        bonus = player.get('bonus', 0)
        bonus_xg = bonus * 0.1  # Each bonus point suggests ~0.1 xG
        
        # Add estimate from ICT threat
        threat = player.get('threat', 0)
        threat_xg = threat / 100 * 0.5  # Normalize threat to xG
        
        # Combine estimates
        total_xg = goals + (estimated_chances * 0.3) + bonus_xg + threat_xg
        
        # Apply position adjustment
        position_multiplier = {
            'FWD': 1.0,
            'MID': 0.8,
            'DEF': 0.3,
            'GKP': 0.0
        }.get(position, 0.8)
        
        return max(0, total_xg * position_multiplier)
    
    def _estimate_player_xa(self, player: pd.Series) -> float:
        """
        Estimate xA for a single player
        
        Formula based on:
        - Assists (actual conversion)
        - Creativity (ICT index component)
        - Key passes proxy
        - Position-specific rates
        """
        position = player.get('position', 'MID')
        
        # Base xA from assists
        assists = player.get('assists', 0)
        
        # Estimate from creativity
        creativity = player.get('creativity', 0)
        creativity_xa = creativity / 100 * 0.6
        
        # Estimate from bonus (assists often lead to bonus)
        bonus = player.get('bonus', 0)
        bonus_xa = bonus * 0.08
        
        # Combine estimates
        total_xa = assists + (assists * 1.5) + creativity_xa + bonus_xa
        
        # Apply position adjustment
        position_multiplier = {
            'FWD': 0.6,
            'MID': 1.0,
            'DEF': 0.4,
            'GKP': 0.0
        }.get(position, 0.8)
        
        return max(0, total_xa * position_multiplier)
    
    def validate_estimates(
        self, 
        estimated_df: pd.DataFrame, 
        actual_xg_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate estimation accuracy against actual xG data
        
        Args:
            estimated_df: DataFrame with estimated xG
            actual_xg_df: DataFrame with actual xG from external source
        
        Returns:
            Dictionary with validation metrics
        """
        # Merge on player_id
        merged = estimated_df.merge(
            actual_xg_df[['player_id', 'xg', 'xa']],
            on='player_id',
            how='inner',
            suffixes=('_est', '_actual')
        )
        
        if len(merged) == 0:
            logger.warning("No players to validate")
            return {}
        
        # Calculate metrics
        xg_mae = np.mean(np.abs(merged['estimated_xg'] - merged['xg']))
        xa_mae = np.mean(np.abs(merged['estimated_xa'] - merged['xa']))
        
        xg_correlation = merged['estimated_xg'].corr(merged['xg'])
        xa_correlation = merged['estimated_xa'].corr(merged['xa'])
        
        metrics = {
            'xg_mae': round(xg_mae, 2),
            'xa_mae': round(xa_mae, 2),
            'xg_correlation': round(xg_correlation, 2),
            'xa_correlation': round(xa_correlation, 2),
            'sample_size': len(merged)
        }
        
        logger.info(f"Validation metrics: {metrics}")
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'player_id': [1, 2, 3],
        'position': ['FWD', 'MID', 'DEF'],
        'goals_scored': [15, 8, 2],
        'assists': [3, 10, 1],
        'bonus': [20, 15, 5],
        'threat': [1200, 800, 200],
        'creativity': [400, 1000, 300],
        'minutes': [2500, 2800, 2600]
    })
    
    estimator = XGEstimator()
    result = estimator.estimate_xg(sample_data)
    
    print("Estimated xG/xA:")
    print(result[['player_id', 'position', 'estimated_xg', 'estimated_xa', 
                  'estimated_xg_per_90', 'estimated_xa_per_90']])
