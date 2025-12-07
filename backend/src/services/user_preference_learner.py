"""
User Preference Learner Service

Infers user preferences from their decisions (accepted/rejected recommendations)
and builds a personalized profile for each user.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

from backend.src.core.postgres_db import PostgresManagerDB


logger = logging.getLogger(__name__)


class UserPreferenceLearner:
    """Learn and infer user preferences from decision history"""
    
    def __init__(self):
        self.postgres_db = PostgresManagerDB()
        self.logger = logging.getLogger(__name__)
    
    def learn_preferences(self, user_id: int, min_decisions: int = 5) -> Dict[str, Any]:
        """
        Analyze user decisions and infer preferences
        
        Args:
            user_id: User ID to analyze
            min_decisions: Minimum number of decisions needed for learning
        
        Returns:
            Dictionary with inferred preferences
        """
        self.logger.info(f"Learning preferences for user {user_id}")
        
        try:
            # Get user decisions
            decisions = self._get_user_decisions(user_id)
            
            if len(decisions) < min_decisions:
                self.logger.warning(f"Not enough decisions for user {user_id}: {len(decisions)} < {min_decisions}")
                return self._get_default_preferences()
            
            # Analyze decisions
            preferences = {
                'user_id': user_id,
                'risk_tolerance': self._calculate_risk_tolerance(decisions),
                'prefers_differentials': self._prefers_differentials(decisions),
                'differential_threshold': self._calculate_differential_threshold(decisions),
                'budget_strategy': self._infer_budget_strategy(decisions),
                'preferred_formation': self._infer_formation_preference(decisions),
                'favorite_teams': self._identify_favorite_teams(decisions),
                'avoided_teams': self._identify_avoided_teams(decisions),
                'captain_strategy': self._infer_captain_strategy(decisions),
                'transfer_aggressiveness': self._calculate_transfer_aggressiveness(decisions),
                'learned_from_decisions': len(decisions),
                'confidence_level': self._calculate_confidence(len(decisions))
            }
            
            # Store preferences
            success = self.postgres_db.update_user_preferences(user_id, preferences)
            
            if success:
                self.logger.info(f"Preferences learned for user {user_id}: risk={preferences['risk_tolerance']:.2f}")
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error learning preferences for user {user_id}: {str(e)}")
            return self._get_default_preferences()
    
    def _get_user_decisions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all decisions for a user"""
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT *
                        FROM user_decisions
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT 100;
                    """, (user_id,))
                    return cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error getting user decisions: {str(e)}")
            return []
    
    def _calculate_risk_tolerance(self, decisions: List[Dict[str, Any]]) -> float:
        """
        Calculate risk tolerance (0-1 scale)
        Higher = more willing to take risks (differentials, hits, etc.)
        """
        if not decisions:
            return 0.5
        
        risk_scores = []
        
        for decision in decisions:
            if decision['decision_type'] == 'transfer':
                # Did they accept risky transfers (differentials)?
                context = decision.get('context', {})
                if isinstance(context, dict):
                    ownership = context.get('ownership', 50)
                    if ownership < 10 and decision['user_accepted']:
                        risk_scores.append(0.8)  # Accepted differential
                    elif ownership > 30 and not decision['user_accepted']:
                        risk_scores.append(0.3)  # Rejected safe pick
                    else:
                        risk_scores.append(0.5)
            
            elif decision['decision_type'] == 'captain':
                # Did they captain differentials?
                context = decision.get('context', {})
                if isinstance(context, dict):
                    ownership = context.get('ownership', 50)
                    if ownership < 20 and decision['user_accepted']:
                        risk_scores.append(0.9)  # Differential captain
                    else:
                        risk_scores.append(0.4)
        
        return np.mean(risk_scores) if risk_scores else 0.5
    
    def _prefers_differentials(self, decisions: List[Dict[str, Any]]) -> bool:
        """Determine if user prefers differential picks"""
        differential_accepts = 0
        total_differential_decisions = 0
        
        for decision in decisions:
            context = decision.get('context', {})
            if isinstance(context, dict):
                ownership = context.get('ownership', 50)
                if ownership < 15:  # Differential threshold
                    total_differential_decisions += 1
                    if decision['user_accepted']:
                        differential_accepts += 1
        
        if total_differential_decisions == 0:
            return False
        
        return (differential_accepts / total_differential_decisions) > 0.6
    
    def _calculate_differential_threshold(self, decisions: List[Dict[str, Any]]) -> float:
        """Calculate ownership threshold for what user considers a differential"""
        accepted_ownerships = []
        
        for decision in decisions:
            if decision['user_accepted']:
                context = decision.get('context', {})
                if isinstance(context, dict):
                    ownership = context.get('ownership')
                    if ownership is not None:
                        accepted_ownerships.append(ownership)
        
        if not accepted_ownerships:
            return 10.0
        
        # Use 25th percentile of accepted ownerships
        return np.percentile(accepted_ownerships, 25)
    
    def _infer_budget_strategy(self, decisions: List[Dict[str, Any]]) -> str:
        """Infer budget strategy: 'premium_heavy', 'balanced', or 'budget_heavy'"""
        premium_accepts = 0
        budget_accepts = 0
        total_transfer_decisions = 0
        
        for decision in decisions:
            if decision['decision_type'] == 'transfer':
                context = decision.get('context', {})
                if isinstance(context, dict):
                    price = context.get('price', 0)
                    if price > 0:
                        total_transfer_decisions += 1
                        if price >= 100:  # 10.0m+
                            if decision['user_accepted']:
                                premium_accepts += 1
                        elif price <= 60:  # 6.0m or less
                            if decision['user_accepted']:
                                budget_accepts += 1
        
        if total_transfer_decisions == 0:
            return 'balanced'
        
        premium_rate = premium_accepts / total_transfer_decisions
        budget_rate = budget_accepts / total_transfer_decisions
        
        if premium_rate > 0.4:
            return 'premium_heavy'
        elif budget_rate > 0.4:
            return 'budget_heavy'
        else:
            return 'balanced'
    
    def _infer_formation_preference(self, decisions: List[Dict[str, Any]]) -> Optional[str]:
        """Infer preferred formation from decisions"""
        formations = {}
        
        for decision in decisions:
            context = decision.get('context', {})
            if isinstance(context, dict):
                formation = context.get('formation')
                if formation and decision['user_accepted']:
                    formations[formation] = formations.get(formation, 0) + 1
        
        if not formations:
            return None
        
        # Return most common formation
        return max(formations, key=formations.get)
    
    def _identify_favorite_teams(self, decisions: List[Dict[str, Any]]) -> List[int]:
        """Identify teams user prefers to have players from"""
        team_accepts = {}
        team_total = {}
        
        for decision in decisions:
            context = decision.get('context', {})
            if isinstance(context, dict):
                team_id = context.get('team_id')
                if team_id:
                    team_total[team_id] = team_total.get(team_id, 0) + 1
                    if decision['user_accepted']:
                        team_accepts[team_id] = team_accepts.get(team_id, 0) + 1
        
        # Teams with >70% accept rate and at least 3 decisions
        favorites = []
        for team_id, total in team_total.items():
            if total >= 3:
                accept_rate = team_accepts.get(team_id, 0) / total
                if accept_rate > 0.7:
                    favorites.append(team_id)
        
        return favorites
    
    def _identify_avoided_teams(self, decisions: List[Dict[str, Any]]) -> List[int]:
        """Identify teams user tends to avoid"""
        team_rejects = {}
        team_total = {}
        
        for decision in decisions:
            context = decision.get('context', {})
            if isinstance(context, dict):
                team_id = context.get('team_id')
                if team_id:
                    team_total[team_id] = team_total.get(team_id, 0) + 1
                    if not decision['user_accepted']:
                        team_rejects[team_id] = team_rejects.get(team_id, 0) + 1
        
        # Teams with >70% reject rate and at least 3 decisions
        avoided = []
        for team_id, total in team_total.items():
            if total >= 3:
                reject_rate = team_rejects.get(team_id, 0) / total
                if reject_rate > 0.7:
                    avoided.append(team_id)
        
        return avoided
    
    def _infer_captain_strategy(self, decisions: List[Dict[str, Any]]) -> str:
        """Infer captain strategy: 'safe', 'differential', or 'form_based'"""
        captain_decisions = [d for d in decisions if d['decision_type'] == 'captain']
        
        if not captain_decisions:
            return 'safe'
        
        differential_count = 0
        safe_count = 0
        
        for decision in captain_decisions:
            if decision['user_accepted']:
                context = decision.get('context', {})
                if isinstance(context, dict):
                    ownership = context.get('ownership', 50)
                    if ownership < 20:
                        differential_count += 1
                    elif ownership > 40:
                        safe_count += 1
        
        total = len(captain_decisions)
        if differential_count / total > 0.5:
            return 'differential'
        elif safe_count / total > 0.6:
            return 'safe'
        else:
            return 'form_based'
    
    def _calculate_transfer_aggressiveness(self, decisions: List[Dict[str, Any]]) -> float:
        """
        Calculate transfer aggressiveness (0-1 scale)
        Higher = more willing to take hits
        """
        transfer_decisions = [d for d in decisions if d['decision_type'] == 'transfer']
        
        if not transfer_decisions:
            return 0.5
        
        hit_accepts = 0
        hit_decisions = 0
        
        for decision in transfer_decisions:
            context = decision.get('context', {})
            if isinstance(context, dict):
                is_hit = context.get('is_hit', False)
                if is_hit:
                    hit_decisions += 1
                    if decision['user_accepted']:
                        hit_accepts += 1
        
        if hit_decisions == 0:
            return 0.5
        
        return hit_accepts / hit_decisions
    
    def _calculate_confidence(self, num_decisions: int) -> float:
        """Calculate confidence in learned preferences based on sample size"""
        # Sigmoid function: more decisions = higher confidence
        # Reaches ~0.9 confidence at 50 decisions
        return min(1.0, 1 / (1 + np.exp(-0.1 * (num_decisions - 25))))
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default preferences for new users"""
        return {
            'risk_tolerance': 0.5,
            'prefers_differentials': False,
            'differential_threshold': 10.0,
            'budget_strategy': 'balanced',
            'preferred_formation': None,
            'favorite_teams': [],
            'avoided_teams': [],
            'captain_strategy': 'safe',
            'transfer_aggressiveness': 0.5,
            'learned_from_decisions': 0,
            'confidence_level': 0.0
        }
    
    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get stored preferences for a user"""
        prefs = self.postgres_db.get_user_preferences(user_id)
        
        if not prefs:
            # No stored preferences, return defaults
            return self._get_default_preferences()
        
        return prefs
    
    def update_preferences_from_decision(self, user_id: int, decision: Dict[str, Any]):
        """
        Update user preferences after a new decision
        (Incremental learning)
        """
        try:
            # Track the decision
            self.postgres_db.track_user_decision(decision)
            
            # Re-learn preferences if we have enough decisions
            current_prefs = self.get_user_preferences(user_id)
            decision_count = current_prefs.get('learned_from_decisions', 0) + 1
            
            # Re-learn every 5 decisions
            if decision_count % 5 == 0:
                self.learn_preferences(user_id)
            
        except Exception as e:
            self.logger.error(f"Error updating preferences from decision: {str(e)}")
