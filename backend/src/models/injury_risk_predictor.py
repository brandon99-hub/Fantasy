"""
Injury risk prediction model
Predicts probability of injury based on player workload and history
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class InjuryRiskPredictor:
    """Predict injury risk for FPL players"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def calculate_injury_risk_features(self, player_history: pd.DataFrame) -> Dict:
        """
        Calculate injury risk features for a player
        
        Args:
            player_history: Player's gameweek history
        
        Returns:
            Dict of injury risk features
        """
        if player_history.empty:
            return self._get_default_features()
        
        # Sort by gameweek
        history = player_history.sort_values('round')
        recent_games = history.tail(10)
        
        features = {}
        
        # 1. Workload features
        features['total_minutes'] = history['minutes'].sum()
        features['minutes_last_5'] = recent_games.tail(5)['minutes'].sum()
        features['minutes_last_10'] = recent_games['minutes'].sum()
        features['avg_minutes_per_game'] = history['minutes'].mean()
        
        # 2. Fatigue indicators
        # Consecutive 90-minute games
        full_games = (history['minutes'] >= 85).astype(int)
        features['consecutive_full_games'] = self._max_consecutive(full_games)
        
        # Games without rest
        features['games_played_last_30_days'] = len(recent_games)
        
        # 3. Physical intensity
        if 'tackles' in history.columns:
            features['tackles_per_90'] = (history['tackles'].sum() / (history['minutes'].sum() / 90)) if history['minutes'].sum() > 0 else 0
        
        if 'recoveries' in history.columns:
            features['recoveries_per_90'] = (history['recoveries'].sum() / (history['minutes'].sum() / 90)) if history['minutes'].sum() > 0 else 0
        
        # 4. Age factor (if available)
        # Older players have higher injury risk
        features['age_factor'] = 0  # Placeholder - would need player age data
        
        # 5. Recent injury history
        # Check for games with 0 minutes (potential injury)
        zero_minute_games = (history['minutes'] == 0).sum()
        features['recent_absences'] = zero_minute_games
        
        # 6. Rotation risk
        # Variance in minutes indicates rotation
        features['minutes_variance'] = history['minutes'].var()
        
        # 7. Position-specific risk
        # Defenders and forwards have different injury profiles
        features['position_risk'] = 0  # Placeholder
        
        return features
    
    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values"""
        max_streak = 0
        current_streak = 0
        
        for val in series:
            if val:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _get_default_features(self) -> Dict:
        """Default features for players with no history"""
        return {
            'total_minutes': 0,
            'minutes_last_5': 0,
            'minutes_last_10': 0,
            'avg_minutes_per_game': 0,
            'consecutive_full_games': 0,
            'games_played_last_30_days': 0,
            'tackles_per_90': 0,
            'recoveries_per_90': 0,
            'age_factor': 0,
            'recent_absences': 0,
            'minutes_variance': 0,
            'position_risk': 0
        }
    
    def predict_injury_risk(self, player_history: pd.DataFrame) -> Dict:
        """
        Predict injury risk for a player
        
        Args:
            player_history: Player's gameweek history
        
        Returns:
            Dict with injury risk assessment
        """
        features = self.calculate_injury_risk_features(player_history)
        
        # Calculate risk score (0-1)
        risk_score = self._calculate_risk_score(features)
        
        # Categorize risk
        if risk_score < 0.2:
            risk_level = '🟢 Low Risk'
            recommendation = 'Safe to select'
        elif risk_score < 0.5:
            risk_level = '🟡 Medium Risk'
            recommendation = 'Monitor closely'
        elif risk_score < 0.75:
            risk_level = '🟠 High Risk'
            recommendation = 'Consider alternatives'
        else:
            risk_level = '🔴 Very High Risk'
            recommendation = 'Avoid or transfer out'
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'factors': self._identify_risk_factors(features),
            'features': features
        }
    
    def _calculate_risk_score(self, features: Dict) -> float:
        """
        Calculate injury risk score from features
        Simple heuristic-based approach (can be replaced with ML model)
        """
        score = 0.0
        
        # High workload
        if features['minutes_last_5'] >= 450:  # All 90 mins
            score += 0.2
        elif features['minutes_last_5'] >= 400:
            score += 0.1
        
        # Consecutive full games (fatigue)
        if features['consecutive_full_games'] >= 5:
            score += 0.25
        elif features['consecutive_full_games'] >= 3:
            score += 0.15
        
        # High intensity
        if features['tackles_per_90'] > 3:
            score += 0.1
        
        # Recent absences (injury history)
        if features['recent_absences'] > 2:
            score += 0.2
        elif features['recent_absences'] > 0:
            score += 0.1
        
        # High variance (rotation/fitness concerns)
        if features['minutes_variance'] > 1000:
            score += 0.15
        
        return min(score, 1.0)
    
    def _identify_risk_factors(self, features: Dict) -> List[str]:
        """Identify specific risk factors"""
        factors = []
        
        if features['consecutive_full_games'] >= 3:
            factors.append(f"Playing every game ({features['consecutive_full_games']} consecutive)")
        
        if features['minutes_last_5'] >= 450:
            factors.append("High recent workload (450+ mins in last 5)")
        
        if features['recent_absences'] > 0:
            factors.append(f"Recent injury history ({features['recent_absences']} absences)")
        
        if features['tackles_per_90'] > 3:
            factors.append("High physical intensity")
        
        if features['minutes_variance'] > 1000:
            factors.append("Rotation concerns")
        
        if not factors:
            factors.append("No significant risk factors identified")
        
        return factors
    
    def batch_predict_injury_risk(
        self,
        players_df: pd.DataFrame,
        history_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict injury risk for multiple players
        
        Args:
            players_df: Players DataFrame
            history_df: All player histories
        
        Returns:
            players_df with injury risk columns
        """
        self.logger.info("Calculating injury risk for all players...")
        
        injury_data = []
        
        for _, player in players_df.iterrows():
            player_id = player['id']
            player_hist = history_df[history_df['element'] == player_id]
            
            risk = self.predict_injury_risk(player_hist)
            
            injury_data.append({
                'id': player_id,
                'injury_risk_score': risk['risk_score'],
                'injury_risk_level': risk['risk_level'],
                'injury_recommendation': risk['recommendation'],
                'injury_risk_factors': ' | '.join(risk['factors'])
            })
        
        injury_df = pd.DataFrame(injury_data)
        return players_df.merge(injury_df, on='id', how='left')
    
    def train_ml_model(self, training_data: pd.DataFrame):
        """
        Train ML model for injury prediction
        
        Args:
            training_data: Historical data with injury labels
        
        Note: This requires historical injury data which we don't have yet
        """
        # Placeholder for future ML model training
        self.logger.info("ML model training not yet implemented - using heuristic approach")
        pass
