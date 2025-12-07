"""
Set piece takers data collector
Tracks and updates penalty, free kick, and corner takers
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SetPieceCollector:
    """Collect and manage set piece taker data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def update_set_piece_data(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update set piece data from FPL API
        
        Args:
            players_df: Players DataFrame
        
        Returns:
            Updated DataFrame with set piece information
        """
        self.logger.info("Updating set piece taker data...")
        
        # FPL API provides these fields:
        # - corners_and_indirect_freekicks_order
        # - corners_and_indirect_freekicks_text
        # - direct_freekicks_order
        # - direct_freekicks_text
        # - penalties_order
        # - penalties_text
        
        # Add boolean flags for easier filtering
        players_df['is_penalty_taker'] = players_df['penalties_order'].notna() & (players_df['penalties_order'] == 1)
        players_df['is_corner_taker'] = players_df['corners_and_indirect_freekicks_order'].notna() & (players_df['corners_and_indirect_freekicks_order'] <= 2)
        players_df['is_freekick_taker'] = players_df['direct_freekicks_order'].notna() & (players_df['direct_freekicks_order'] <= 2)
        
        # Calculate set piece bonus score
        players_df['set_piece_score'] = self._calculate_set_piece_score(players_df)
        
        return players_df
    
    def _calculate_set_piece_score(self, players_df: pd.DataFrame) -> pd.Series:
        """
        Calculate set piece importance score
        
        Penalties are most valuable, then free kicks, then corners
        """
        score = pd.Series(0.0, index=players_df.index)
        
        # Penalty taker: +3 points
        score += players_df['is_penalty_taker'].astype(int) * 3
        
        # Free kick taker: +2 points
        score += players_df['is_freekick_taker'].astype(int) * 2
        
        # Corner taker: +1 point
        score += players_df['is_corner_taker'].astype(int) * 1
        
        return score
    
    def get_set_piece_takers(
        self,
        players_df: pd.DataFrame,
        team_name: Optional[str] = None,
        set_piece_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get set piece takers with optional filtering
        
        Args:
            players_df: Players DataFrame
            team_name: Filter by team (optional)
            set_piece_type: 'penalties', 'corners', or 'freekicks' (optional)
        
        Returns:
            Filtered DataFrame of set piece takers
        """
        result = players_df.copy()
        
        # Filter by team
        if team_name:
            result = result[result['team_name'] == team_name]
        
        # Filter by set piece type
        if set_piece_type == 'penalties':
            result = result[result['is_penalty_taker']]
        elif set_piece_type == 'corners':
            result = result[result['is_corner_taker']]
        elif set_piece_type == 'freekicks':
            result = result[result['is_freekick_taker']]
        else:
            # Any set piece taker
            result = result[
                result['is_penalty_taker'] |
                result['is_corner_taker'] |
                result['is_freekick_taker']
            ]
        
        return result[['web_name', 'team_name', 'position', 'now_cost',
                       'is_penalty_taker', 'is_corner_taker', 'is_freekick_taker',
                       'set_piece_score', 'penalties_text', 'corners_and_indirect_freekicks_text',
                       'direct_freekicks_text']]
    
    def analyze_set_piece_value(self, players_df: pd.DataFrame) -> Dict:
        """
        Analyze the value of set piece takers
        
        Args:
            players_df: Players DataFrame with set piece data
        
        Returns:
            Analysis of set piece takers' performance
        """
        penalty_takers = players_df[players_df['is_penalty_taker']]
        corner_takers = players_df[players_df['is_corner_taker']]
        freekick_takers = players_df[players_df['is_freekick_taker']]
        
        analysis = {
            'penalty_takers': {
                'count': len(penalty_takers),
                'avg_points': float(penalty_takers['total_points'].mean()) if not penalty_takers.empty else 0,
                'top_5': penalty_takers.nlargest(5, 'total_points')[['web_name', 'team_name', 'total_points']].to_dict('records')
            },
            'corner_takers': {
                'count': len(corner_takers),
                'avg_points': float(corner_takers['total_points'].mean()) if not corner_takers.empty else 0,
                'top_5': corner_takers.nlargest(5, 'total_points')[['web_name', 'team_name', 'total_points']].to_dict('records')
            },
            'freekick_takers': {
                'count': len(freekick_takers),
                'avg_points': float(freekick_takers['total_points'].mean()) if not freekick_takers.empty else 0,
                'top_5': freekick_takers.nlargest(5, 'total_points')[['web_name', 'team_name', 'total_points']].to_dict('records')
            }
        }
        
        return analysis
    
    def weight_predictions_for_set_pieces(
        self,
        players_df: pd.DataFrame,
        penalty_boost: float = 0.5,
        freekick_boost: float = 0.3,
        corner_boost: float = 0.2
    ) -> pd.DataFrame:
        """
        Adjust predicted points based on set piece duties
        
        Args:
            players_df: Players DataFrame
            penalty_boost: Points boost for penalty takers
            freekick_boost: Points boost for free kick takers
            corner_boost: Points boost for corner takers
        
        Returns:
            DataFrame with adjusted predictions
        """
        players_df = players_df.copy()
        
        # Add set piece bonus to predictions
        set_piece_bonus = (
            players_df['is_penalty_taker'].astype(int) * penalty_boost +
            players_df['is_freekick_taker'].astype(int) * freekick_boost +
            players_df['is_corner_taker'].astype(int) * corner_boost
        )
        
        if 'predicted_points' in players_df.columns:
            players_df['predicted_points_with_set_pieces'] = (
                players_df['predicted_points'] + set_piece_bonus
            )
        
        return players_df
    
    def get_team_set_piece_summary(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get set piece takers summary by team
        
        Args:
            players_df: Players DataFrame
        
        Returns:
            Summary DataFrame by team
        """
        summary_data = []
        
        for team in players_df['team_name'].unique():
            team_players = players_df[players_df['team_name'] == team]
            
            penalty_taker = team_players[team_players['is_penalty_taker']]
            corner_taker = team_players[team_players['is_corner_taker']]
            freekick_taker = team_players[team_players['is_freekick_taker']]
            
            summary_data.append({
                'team': team,
                'penalty_taker': penalty_taker['web_name'].iloc[0] if not penalty_taker.empty else 'None',
                'corner_taker': ', '.join(corner_taker['web_name'].head(2).tolist()) if not corner_taker.empty else 'None',
                'freekick_taker': ', '.join(freekick_taker['web_name'].head(2).tolist()) if not freekick_taker.empty else 'None',
                'total_set_piece_players': len(team_players[team_players['set_piece_score'] > 0])
            })
        
        return pd.DataFrame(summary_data).sort_values('team')
    
    def identify_set_piece_changes(
        self,
        old_data: pd.DataFrame,
        new_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Identify changes in set piece takers between updates
        
        Args:
            old_data: Previous players data
            new_data: Current players data
        
        Returns:
            List of changes detected
        """
        changes = []
        
        # Check for penalty taker changes
        old_pens = set(old_data[old_data['is_penalty_taker']]['id'])
        new_pens = set(new_data[new_data['is_penalty_taker']]['id'])
        
        # New penalty takers
        for player_id in new_pens - old_pens:
            player = new_data[new_data['id'] == player_id].iloc[0]
            changes.append({
                'type': 'penalty_taker_added',
                'player': player['web_name'],
                'team': player['team_name'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Removed penalty takers
        for player_id in old_pens - new_pens:
            player = old_data[old_data['id'] == player_id].iloc[0]
            changes.append({
                'type': 'penalty_taker_removed',
                'player': player['web_name'],
                'team': player['team_name'],
                'timestamp': datetime.now().isoformat()
            })
        
        return changes
