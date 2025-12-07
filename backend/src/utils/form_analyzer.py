"""
Advanced form momentum analysis for FPL players
Detects hot/cold streaks, trend analysis, and breakout predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FormAnalyzer:
    """Analyze player form momentum and trends"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_momentum(self, player_history: pd.DataFrame, window: int = 5) -> Dict[str, float]:
        """
        Calculate form momentum score
        
        Args:
            player_history: DataFrame with 'total_points' column
            window: Number of recent games to analyze
        
        Returns:
            Dict with momentum metrics
        """
        if player_history.empty or len(player_history) < 3:
            return {
                'momentum_score': 0.0,
                'trend': 'stable',
                'hot_streak_length': 0,
                'cold_streak_length': 0,
                'volatility': 0.0
            }
        
        # Sort by gameweek
        history = player_history.sort_values('round').tail(window)
        points = history['total_points'].values
        
        # Calculate exponential moving average
        weights = np.exp(np.linspace(-1., 0., len(points)))
        weights /= weights.sum()
        ema = np.average(points, weights=weights)
        
        # Calculate season average
        season_avg = player_history['total_points'].mean()
        
        # Momentum score: -1 (declining) to +1 (improving)
        if season_avg > 0:
            momentum_score = (ema - season_avg) / (season_avg + 1)
            momentum_score = np.clip(momentum_score, -1, 1)
        else:
            momentum_score = 0.0
        
        # Detect trend using linear regression
        if len(points) >= 3:
            x = np.arange(len(points))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, points)
            
            if p_value < 0.1:  # Significant trend
                if slope > 0.5:
                    trend = 'improving'
                elif slope < -0.5:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # Detect streaks
        hot_streak = self._detect_hot_streak(points, season_avg)
        cold_streak = self._detect_cold_streak(points, season_avg)
        
        # Calculate volatility
        volatility = np.std(points) if len(points) > 1 else 0.0
        
        return {
            'momentum_score': float(momentum_score),
            'trend': trend,
            'hot_streak_length': hot_streak,
            'cold_streak_length': cold_streak,
            'volatility': float(volatility),
            'recent_avg': float(ema),
            'season_avg': float(season_avg)
        }
    
    def _detect_hot_streak(self, points: np.ndarray, threshold: float) -> int:
        """Detect consecutive above-average performances"""
        streak = 0
        for p in reversed(points):
            if p >= threshold:
                streak += 1
            else:
                break
        return streak
    
    def _detect_cold_streak(self, points: np.ndarray, threshold: float) -> int:
        """Detect consecutive below-average performances"""
        streak = 0
        for p in reversed(points):
            if p < threshold:
                streak += 1
            else:
                break
        return streak
    
    def detect_breakout(self, player_history: pd.DataFrame, lookback: int = 10) -> Dict[str, any]:
        """
        Detect if player is breaking out (sustained improvement)
        
        Args:
            player_history: Player's gameweek history
            lookback: Games to look back
        
        Returns:
            Dict with breakout analysis
        """
        if len(player_history) < lookback:
            return {
                'is_breakout': False,
                'confidence': 0.0,
                'reason': 'Insufficient data'
            }
        
        history = player_history.sort_values('round').tail(lookback)
        points = history['total_points'].values
        
        # Split into first half and second half
        mid = len(points) // 2
        first_half = points[:mid]
        second_half = points[mid:]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        # Check for significant improvement
        if second_avg > first_avg * 1.5 and second_avg > 4:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(second_half, first_half)
            
            if p_value < 0.1 and t_stat > 0:
                confidence = 1 - p_value
                return {
                    'is_breakout': True,
                    'confidence': float(confidence),
                    'first_half_avg': float(first_avg),
                    'second_half_avg': float(second_avg),
                    'improvement': float((second_avg - first_avg) / (first_avg + 1)),
                    'reason': f'Significant improvement: {first_avg:.1f} → {second_avg:.1f} pts'
                }
        
        return {
            'is_breakout': False,
            'confidence': 0.0,
            'first_half_avg': float(first_avg),
            'second_half_avg': float(second_avg),
            'reason': 'No significant improvement detected'
        }
    
    def predict_form_continuation(self, player_history: pd.DataFrame) -> Dict[str, float]:
        """
        Predict probability of maintaining current form
        
        Args:
            player_history: Player's gameweek history
        
        Returns:
            Dict with continuation probabilities
        """
        momentum = self.calculate_momentum(player_history)
        
        # Base probability on momentum and volatility
        if momentum['trend'] == 'improving':
            base_prob = 0.7
        elif momentum['trend'] == 'declining':
            base_prob = 0.3
        else:
            base_prob = 0.5
        
        # Adjust for volatility (high volatility = less predictable)
        volatility_factor = 1 - (momentum['volatility'] / 10)  # Normalize
        volatility_factor = np.clip(volatility_factor, 0.5, 1.0)
        
        # Adjust for streak length
        if momentum['hot_streak_length'] >= 3:
            streak_boost = 0.15
        elif momentum['cold_streak_length'] >= 3:
            streak_boost = -0.15
        else:
            streak_boost = 0.0
        
        continuation_prob = base_prob * volatility_factor + streak_boost
        continuation_prob = np.clip(continuation_prob, 0.0, 1.0)
        
        return {
            'continuation_probability': float(continuation_prob),
            'confidence': float(volatility_factor),
            'factors': {
                'trend': momentum['trend'],
                'volatility': momentum['volatility'],
                'streak': max(momentum['hot_streak_length'], momentum['cold_streak_length'])
            }
        }
    
    def analyze_player_form(self, player_history: pd.DataFrame) -> Dict[str, any]:
        """
        Comprehensive form analysis for a player
        
        Args:
            player_history: Player's gameweek history
        
        Returns:
            Complete form analysis
        """
        momentum = self.calculate_momentum(player_history)
        breakout = self.detect_breakout(player_history)
        continuation = self.predict_form_continuation(player_history)
        
        return {
            'momentum': momentum,
            'breakout': breakout,
            'continuation': continuation,
            'summary': self._generate_summary(momentum, breakout, continuation)
        }
    
    def _generate_summary(self, momentum: Dict, breakout: Dict, continuation: Dict) -> str:
        """Generate human-readable form summary"""
        parts = []
        
        # Trend
        if momentum['trend'] == 'improving':
            parts.append("📈 Improving form")
        elif momentum['trend'] == 'declining':
            parts.append("📉 Declining form")
        else:
            parts.append("➡️  Stable form")
        
        # Streak
        if momentum['hot_streak_length'] >= 3:
            parts.append(f"🔥 {momentum['hot_streak_length']}-game hot streak")
        elif momentum['cold_streak_length'] >= 3:
            parts.append(f"❄️  {momentum['cold_streak_length']}-game cold streak")
        
        # Breakout
        if breakout['is_breakout']:
            parts.append(f"⭐ Breaking out ({breakout['confidence']:.0%} confidence)")
        
        # Continuation
        if continuation['continuation_probability'] > 0.7:
            parts.append(f"✅ Likely to continue ({continuation['continuation_probability']:.0%})")
        elif continuation['continuation_probability'] < 0.3:
            parts.append(f"⚠️  Form may not last ({continuation['continuation_probability']:.0%})")
        
        return " | ".join(parts) if parts else "No significant form pattern"
    
    def batch_analyze_players(self, players_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze form for multiple players
        
        Args:
            players_df: Players DataFrame
            history_df: All player histories
        
        Returns:
            players_df with added form columns
        """
        form_data = []
        
        for _, player in players_df.iterrows():
            player_id = player['id']
            player_hist = history_df[history_df['element'] == player_id]
            
            if not player_hist.empty:
                analysis = self.analyze_player_form(player_hist)
                form_data.append({
                    'id': player_id,
                    'form_momentum_score': analysis['momentum']['momentum_score'],
                    'form_trend': analysis['momentum']['trend'],
                    'hot_streak_length': analysis['momentum']['hot_streak_length'],
                    'form_volatility': analysis['momentum']['volatility'],
                    'is_breakout': analysis['breakout']['is_breakout'],
                    'breakout_confidence': analysis['breakout']['confidence'],
                    'form_continuation_prob': analysis['continuation']['continuation_probability'],
                    'form_summary': analysis['summary']
                })
            else:
                form_data.append({
                    'id': player_id,
                    'form_momentum_score': 0.0,
                    'form_trend': 'unknown',
                    'hot_streak_length': 0,
                    'form_volatility': 0.0,
                    'is_breakout': False,
                    'breakout_confidence': 0.0,
                    'form_continuation_prob': 0.5,
                    'form_summary': 'No data'
                })
        
        form_df = pd.DataFrame(form_data)
        return players_df.merge(form_df, on='id', how='left')
