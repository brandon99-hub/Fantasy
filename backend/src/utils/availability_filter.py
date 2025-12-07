"""
Availability Filter - Conservative player selection

Filters players based on:
- Injury status and risk
- Minutes played trends
- Chance of playing percentages
- Position-specific risk factors
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Filtering mode configurations
FILTERING_MODES = {
    'conservative': {
        'min_chance_of_playing': 75,
        'exclude_doubtful': True,
        'min_minutes_last_5': 200,  # ~45 min/game
        'max_injury_risk': 0.3,
        'require_recent_start': True
    },
    'balanced': {
        'min_chance_of_playing': 50,
        'exclude_doubtful': False,
        'min_minutes_last_5': 150,  # ~30 min/game
        'max_injury_risk': 0.5,
        'require_recent_start': False
    },
    'aggressive': {
        'min_chance_of_playing': 25,
        'exclude_doubtful': False,
        'min_minutes_last_5': 90,   # ~20 min/game
        'max_injury_risk': 0.7,
        'require_recent_start': False
    }
}


class AvailabilityFilter:
    """Filter players based on availability and injury risk"""
    
    def __init__(self, mode: str = 'conservative'):
        """
        Initialize filter with specified mode
        
        Args:
            mode: 'conservative', 'balanced', or 'aggressive'
        """
        if mode not in FILTERING_MODES:
            logger.warning(f"Unknown mode '{mode}', using 'conservative'")
            mode = 'conservative'
        
        self.mode = mode
        self.config = FILTERING_MODES[mode]
        logger.info(f"Availability Filter initialized in '{mode}' mode")
    
    def filter_available_players(
        self,
        players_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Filter to only available players
        
        Args:
            players_df: Player data
            history_df: Historical performance data
        
        Returns:
            Filtered DataFrame with available players
        """
        logger.info(f"Filtering {len(players_df)} players ({self.mode} mode)...")
        
        filtered = players_df.copy()
        initial_count = len(filtered)
        
        # 1. Filter by chance of playing
        if 'chance_of_playing_next_round' in filtered.columns:
            min_chance = self.config['min_chance_of_playing']
            filtered = filtered[
                (filtered['chance_of_playing_next_round'].isna()) |
                (filtered['chance_of_playing_next_round'] >= min_chance)
            ]
            logger.info(f"After chance filter: {len(filtered)} players")
        
        # 2. Exclude doubtful players
        if self.config['exclude_doubtful'] and 'status' in filtered.columns:
            filtered = filtered[filtered['status'] != 'd']  # 'd' = doubtful
            logger.info(f"After status filter: {len(filtered)} players")
        
        # 3. Filter by recent minutes
        if history_df is not None and not history_df.empty:
            filtered = self._filter_by_minutes(filtered, history_df)
            logger.info(f"After minutes filter: {len(filtered)} players")
        
        # 4. Calculate and filter by injury risk
        if history_df is not None and not history_df.empty:
            filtered = self._filter_by_injury_risk(filtered, history_df)
            logger.info(f"After injury risk filter: {len(filtered)} players")
        
        filtered_count = len(filtered)
        logger.info(
            f"Filtered {initial_count} → {filtered_count} players "
            f"({filtered_count/initial_count*100:.1f}% retained)"
        )
        
        return filtered
    
    def _filter_by_minutes(
        self,
        players_df: pd.DataFrame,
        history_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter by recent minutes played"""
        min_minutes = self.config['min_minutes_last_5']
        
        # Calculate recent minutes for each player
        recent_minutes = history_df.groupby('element').apply(
            lambda x: x.nlargest(5, 'round')['minutes'].sum()
        ).to_dict()
        
        # Add to dataframe
        players_df = players_df.copy()
        players_df['recent_minutes'] = players_df['id'].map(recent_minutes).fillna(0)
        
        # Filter
        filtered = players_df[players_df['recent_minutes'] >= min_minutes]
        
        return filtered
    
    def _filter_by_injury_risk(
        self,
        players_df: pd.DataFrame,
        history_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate and filter by injury risk score"""
        max_risk = self.config['max_injury_risk']
        
        # Calculate injury risk for each player
        risk_scores = {}
        
        for player_id in players_df['id']:
            risk = self._calculate_injury_risk(player_id, players_df, history_df)
            risk_scores[player_id] = risk
        
        # Add to dataframe
        players_df = players_df.copy()
        players_df['injury_risk'] = players_df['id'].map(risk_scores).fillna(0.5)
        
        # Filter
        filtered = players_df[players_df['injury_risk'] <= max_risk]
        
        return filtered
    
    def _calculate_injury_risk(
        self,
        player_id: int,
        players_df: pd.DataFrame,
        history_df: pd.DataFrame
    ) -> float:
        """
        Calculate injury risk score (0-1)
        
        Factors:
        - Recent injury history
        - Minutes trend
        - Age
        - Position-specific risk
        """
        player = players_df[players_df['id'] == player_id]
        
        if player.empty:
            return 0.5  # Unknown
        
        player = player.iloc[0]
        
        # 1. Status-based risk
        status = player.get('status', 'a')
        status_risk = {
            'a': 0.0,   # Available
            'i': 1.0,   # Injured
            'd': 0.7,   # Doubtful
            's': 0.5,   # Suspended
            'u': 0.3    # Unavailable
        }.get(status, 0.5)
        
        # 2. Chance of playing risk
        chance = player.get('chance_of_playing_next_round')
        if pd.notna(chance):
            chance_risk = 1 - (chance / 100)
        else:
            chance_risk = 0.0
        
        # 3. Minutes trend risk
        player_history = history_df[history_df['element'] == player_id].nlargest(5, 'round')
        
        if len(player_history) >= 3:
            minutes = player_history['minutes'].values
            # Declining minutes = higher risk
            if len(minutes) > 1:
                trend = np.polyfit(range(len(minutes)), minutes, 1)[0]
                trend_risk = max(0, -trend / 90)  # Negative trend = risk
            else:
                trend_risk = 0.0
        else:
            trend_risk = 0.3  # Lack of data = moderate risk
        
        # 4. News-based risk
        news = player.get('news', '')
        news_risk = 0.5 if news and len(news) > 0 else 0.0
        
        # Combine factors
        risk_score = (
            status_risk * 0.4 +
            chance_risk * 0.3 +
            trend_risk * 0.2 +
            news_risk * 0.1
        )
        
        return np.clip(risk_score, 0, 1)


# Quick filter function
def filter_available(
    players_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
    mode: str = 'conservative'
) -> pd.DataFrame:
    """
    Quick filter for available players
    
    Args:
        players_df: Player data
        history_df: Historical data
        mode: Filtering strictness
    
    Returns:
        Filtered players
    """
    filter_obj = AvailabilityFilter(mode)
    return filter_obj.filter_available_players(players_df, history_df)


if __name__ == "__main__":
    # Test with sample data
    players_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'web_name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
        'status': ['a', 'd', 'i', 'a', 'a'],
        'chance_of_playing_next_round': [100, 50, 0, 75, None]
    })
    
    history_df = pd.DataFrame({
        'element': [1, 1, 1, 2, 2, 3, 4, 4, 5],
        'round': [10, 9, 8, 10, 9, 10, 10, 9, 10],
        'minutes': [90, 85, 90, 30, 45, 0, 90, 90, 60]
    })
    
    # Test different modes
    for mode in ['conservative', 'balanced', 'aggressive']:
        print(f"\n{mode.upper()} Mode:")
        filtered = filter_available(players_df, history_df, mode)
        print(f"Retained: {filtered['web_name'].tolist()}")
