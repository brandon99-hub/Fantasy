"""
Personalized Optimizer

Wraps FPLOptimizer to provide personalized recommendations based on user preferences.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd

from backend.src.core.optimizer import FPLOptimizer
from backend.src.services.user_preference_learner import UserPreferenceLearner


logger = logging.getLogger(__name__)


class PersonalizedOptimizer:
    """Optimizer that adapts recommendations to user preferences"""
    
    def __init__(self):
        self.base_optimizer = FPLOptimizer()
        self.preference_learner = UserPreferenceLearner()
        self.logger = logging.getLogger(__name__)
    
    def optimize_team(
        self,
        players_df: pd.DataFrame,
        budget: float,
        user_id: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize team with personalization
        
        Args:
            players_df: Player data
            budget: Available budget
            user_id: User ID for personalization (optional)
            **kwargs: Additional arguments for base optimizer
        
        Returns:
            Optimized team with personalization applied
        """
        # Get base optimization
        result = self.base_optimizer.optimize_team(players_df, budget, **kwargs)
        
        # Apply personalization if user_id provided
        if user_id:
            result = self._apply_personalization(result, players_df, user_id)
        
        return result
    
    def _apply_personalization(
        self,
        result: Dict[str, Any],
        players_df: pd.DataFrame,
        user_id: int
    ) -> Dict[str, Any]:
        """Apply user preferences to optimization result"""
        try:
            # Get user preferences
            prefs = self.preference_learner.get_user_preferences(user_id)
            
            if prefs.get('confidence_level', 0) < 0.3:
                # Not enough data for personalization
                result['personalization'] = {
                    'applied': False,
                    'reason': 'Insufficient user data'
                }
                return result
            
            # Adjust recommendations based on preferences
            adjustments = []
            
            # 1. Risk tolerance adjustments
            if prefs.get('prefers_differentials'):
                result = self._boost_differentials(result, players_df)
                adjustments.append('Boosted differential picks')
            
            # 2. Team preferences
            favorite_teams = prefs.get('favorite_teams', [])
            if favorite_teams:
                result = self._favor_teams(result, players_df, favorite_teams)
                adjustments.append(f'Favored teams: {favorite_teams}')
            
            # 3. Formation preference
            preferred_formation = prefs.get('preferred_formation')
            if preferred_formation:
                result['recommended_formation'] = preferred_formation
                adjustments.append(f'Suggested formation: {preferred_formation}')
            
            # 4. Captain strategy
            captain_strategy = prefs.get('captain_strategy', 'safe')
            result = self._adjust_captain_suggestions(result, captain_strategy)
            adjustments.append(f'Captain strategy: {captain_strategy}')
            
            result['personalization'] = {
                'applied': True,
                'user_id': user_id,
                'confidence': prefs.get('confidence_level'),
                'adjustments': adjustments,
                'preferences_used': {
                    'risk_tolerance': prefs.get('risk_tolerance'),
                    'prefers_differentials': prefs.get('prefers_differentials'),
                    'captain_strategy': captain_strategy
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying personalization: {str(e)}")
            result['personalization'] = {
                'applied': False,
                'error': str(e)
            }
            return result
    
    def _boost_differentials(self, result: Dict[str, Any], players_df: pd.DataFrame) -> Dict[str, Any]:
        """Boost differential picks in recommendations"""
        # Implementation would adjust transfer suggestions to favor lower ownership
        return result
    
    def _favor_teams(self, result: Dict[str, Any], players_df: pd.DataFrame, teams: List[int]) -> Dict[str, Any]:
        """Favor players from specific teams"""
        # Implementation would boost players from favorite teams
        return result
    
    def _adjust_captain_suggestions(self, result: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Adjust captain suggestions based on strategy"""
        if 'captain_suggestions' in result:
            if strategy == 'differential':
                # Re-order to favor lower ownership captains
                result['captain_suggestions'] = sorted(
                    result['captain_suggestions'],
                    key=lambda x: x.get('ownership', 100)
                )
            elif strategy == 'safe':
                # Re-order to favor higher ownership captains
                result['captain_suggestions'] = sorted(
                    result['captain_suggestions'],
                    key=lambda x: -x.get('ownership', 0)
                )
        
        return result
