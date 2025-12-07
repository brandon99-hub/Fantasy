"""
Differential finder - identifies low-owned, high-value players
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DifferentialFinder:
    """Find differential players for rank climbing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def find_differentials(
        self,
        players_df: pd.DataFrame,
        ownership_threshold: float = 10.0,
        min_predicted_points: float = 4.0,
        risk_tolerance: str = 'medium',
        position: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Find differential players
        
        Args:
            players_df: All players with predictions
            ownership_threshold: Max ownership % to be considered differential
            min_predicted_points: Minimum predicted points
            risk_tolerance: 'low', 'medium', or 'high'
            position: Filter by position (optional)
        
        Returns:
            DataFrame of differential players sorted by score
        """
        self.logger.info(f"🔍 Finding differentials (ownership <{ownership_threshold}%)...")
        
        # Filter available players
        differentials = players_df[
            (players_df['selected_by_percent'] < ownership_threshold) &
            (players_df['predicted_points'] >= min_predicted_points) &
            (players_df['status'] == 'a')  # Available
        ].copy()
        
        if position:
            differentials = differentials[differentials['position'] == position]
        
        if differentials.empty:
            return pd.DataFrame()
        
        # Calculate differential score
        differentials['differential_score'] = self._calculate_differential_score(
            differentials, risk_tolerance
        )
        
        # Add risk rating
        differentials['risk_rating'] = differentials.apply(
            lambda x: self._calculate_risk_rating(x), axis=1
        )
        
        # Add recommendation reason
        differentials['recommendation_reason'] = differentials.apply(
            lambda x: self._generate_reason(x), axis=1
        )
        
        # Sort by differential score
        differentials = differentials.sort_values('differential_score', ascending=False)
        
        return differentials[[
            'web_name', 'team_name', 'position', 'now_cost', 'predicted_points',
            'selected_by_percent', 'differential_score', 'risk_rating',
            'form', 'recommendation_reason'
        ]]
    
    def _calculate_differential_score(
        self,
        players: pd.DataFrame,
        risk_tolerance: str
    ) -> pd.Series:
        """
        Calculate differential score
        Score = (predicted_points / ownership) * fixture_quality * form_factor
        """
        # Base score: points per ownership percentage
        base_score = players['predicted_points'] / (players['selected_by_percent'] + 0.1)
        
        # Form factor (recent form boost)
        form_factor = 1 + (players['form'].fillna(0) / 10)
        
        # Fixture quality (if available)
        if 'fixture_difficulty' in players.columns:
            fixture_factor = 1 + ((5 - players['fixture_difficulty']) / 10)
        else:
            fixture_factor = 1.0
        
        # Risk adjustment based on tolerance
        if risk_tolerance == 'low':
            # Penalize high variance
            variance_penalty = 1 - (players.get('points_variance', 0) / 20)
            variance_penalty = np.clip(variance_penalty, 0.7, 1.0)
        elif risk_tolerance == 'high':
            # Reward high variance (high ceiling)
            variance_bonus = 1 + (players.get('points_variance', 0) / 30)
            variance_bonus = np.clip(variance_bonus, 1.0, 1.3)
            variance_penalty = variance_bonus
        else:  # medium
            variance_penalty = 1.0
        
        differential_score = base_score * form_factor * fixture_factor * variance_penalty
        
        return differential_score
    
    def _calculate_risk_rating(self, player: pd.Series) -> str:
        """Calculate risk rating for player"""
        risk_factors = 0
        
        # Low ownership = higher risk
        if player['selected_by_percent'] < 3:
            risk_factors += 2
        elif player['selected_by_percent'] < 7:
            risk_factors += 1
        
        # Low minutes = higher risk
        if player.get('minutes', 0) < 500:
            risk_factors += 1
        
        # Poor form = higher risk
        if player.get('form', 0) < 3:
            risk_factors += 1
        
        # High price = lower risk (usually)
        if player['now_cost'] > 90:
            risk_factors -= 1
        
        if risk_factors >= 3:
            return '🔴 High Risk'
        elif risk_factors >= 1:
            return '🟡 Medium Risk'
        else:
            return '🟢 Low Risk'
    
    def _generate_reason(self, player: pd.Series) -> str:
        """Generate recommendation reason"""
        reasons = []
        
        # Ownership
        reasons.append(f"{player['selected_by_percent']:.1f}% owned")
        
        # Predicted points
        reasons.append(f"{player['predicted_points']:.1f} pts expected")
        
        # Form
        if player.get('form', 0) > 5:
            reasons.append("🔥 Hot form")
        
        # Fixtures
        if player.get('fixture_difficulty', 3) < 3:
            reasons.append("✅ Good fixtures")
        
        return " | ".join(reasons)
    
    def calculate_template_balance(
        self,
        team: List[int],
        players_df: pd.DataFrame,
        target_differential_count: int = 4
    ) -> Dict:
        """
        Analyze template vs differential balance
        
        Args:
            team: List of 15 player IDs
            players_df: All players
            target_differential_count: Target number of differentials
        
        Returns:
            Dict with balance analysis
        """
        team_players = players_df[players_df['id'].isin(team)]
        
        # Define template (>20% owned)
        template_players = team_players[team_players['selected_by_percent'] > 20]
        differential_players = team_players[team_players['selected_by_percent'] <= 10]
        
        analysis = {
            'template_count': len(template_players),
            'differential_count': len(differential_players),
            'target_differential_count': target_differential_count,
            'balance_score': 0.0,
            'recommendation': '',
            'template_players': template_players[['web_name', 'selected_by_percent']].to_dict('records'),
            'differential_players': differential_players[['web_name', 'selected_by_percent']].to_dict('records')
        }
        
        # Calculate balance score (1.0 = perfect)
        diff_ratio = len(differential_players) / target_differential_count
        analysis['balance_score'] = min(diff_ratio, 1.0)
        
        # Generate recommendation
        if len(differential_players) < target_differential_count:
            analysis['recommendation'] = f"Consider adding {target_differential_count - len(differential_players)} more differentials"
        elif len(differential_players) > target_differential_count + 2:
            analysis['recommendation'] = "Too many differentials - consider some template players for safety"
        else:
            analysis['recommendation'] = "Good template/differential balance"
        
        return analysis
    
    def rank_strategy_recommendation(
        self,
        current_rank: int,
        target_rank: int,
        gameweeks_remaining: int
    ) -> Dict:
        """
        Recommend differential strategy based on rank goals
        
        Args:
            current_rank: Current overall rank
            target_rank: Target rank
            gameweeks_remaining: Gameweeks left in season
        
        Returns:
            Strategy recommendation
        """
        rank_gap = current_rank - target_rank
        
        if rank_gap <= 0:
            # Ahead of target - protect rank
            strategy = {
                'approach': 'conservative',
                'differential_count': 2,
                'risk_tolerance': 'low',
                'recommendation': '🛡️  Protect your rank with template players and safe differentials'
            }
        elif rank_gap < 50000:
            # Close to target - balanced
            strategy = {
                'approach': 'balanced',
                'differential_count': 4,
                'risk_tolerance': 'medium',
                'recommendation': '⚖️  Balanced approach with mix of template and differentials'
            }
        else:
            # Far from target - aggressive
            differential_count = min(6, 3 + (rank_gap // 100000))
            strategy = {
                'approach': 'aggressive',
                'differential_count': differential_count,
                'risk_tolerance': 'high',
                'recommendation': f'🚀 Aggressive differential strategy needed to climb {rank_gap:,} ranks'
            }
        
        # Adjust for time remaining
        if gameweeks_remaining < 5:
            strategy['differential_count'] += 1
            strategy['recommendation'] += ' - Time running out, increase risk!'
        
        return strategy
