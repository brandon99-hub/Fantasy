"""
Advanced Feature Engineering for FPL Predictions

Creates sophisticated features that capture:
- Time-weighted rolling averages
- Opponent-adjusted metrics
- Fixture congestion
- Home/away splits
- Momentum indicators
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self):
        self.decay_factor = 0.9  # Time decay for rolling averages
        logger.info("Advanced Feature Engineer initialized")
    
    def create_features(
        self,
        players_df: pd.DataFrame,
        history_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create all advanced features
        
        Args:
            players_df: Current player data
            history_df: Player historical performance
            fixtures_df: Upcoming fixtures
            teams_df: Team strength data
        
        Returns:
            DataFrame with advanced features added
        """
        logger.info("Creating advanced features...")
        
        features_df = players_df.copy()
        
        # Time-weighted features
        logger.info("Calculating time-weighted features...")
        tw_features = self._create_time_weighted_features(history_df)
        features_df = features_df.merge(tw_features, on='id', how='left')
        
        # Opponent-adjusted features
        logger.info("Calculating opponent-adjusted features...")
        opp_features = self._create_opponent_adjusted_features(
            features_df, fixtures_df, teams_df
        )
        features_df = features_df.merge(opp_features, on='id', how='left')
        
        # Fixture congestion
        logger.info("Calculating fixture congestion...")
        congestion_features = self._create_congestion_features(
            features_df, fixtures_df, history_df
        )
        features_df = features_df.merge(congestion_features, on='id', how='left')
        
        # Home/away splits
        logger.info("Calculating home/away splits...")
        ha_features = self._create_home_away_features(history_df)
        features_df = features_df.merge(ha_features, on='id', how='left')
        
        # Momentum indicators
        logger.info("Calculating momentum indicators...")
        momentum_features = self._create_momentum_features(history_df)
        features_df = features_df.merge(momentum_features, on='id', how='left')
        
        # Fill NaN values
        feature_cols = [col for col in features_df.columns if col.startswith(('tw_', 'opp_', 'cong_', 'ha_', 'mom_'))]
        features_df[feature_cols] = features_df[feature_cols].fillna(0)
        
        logger.info(f"Created {len(feature_cols)} advanced features")
        
        return features_df
    
    def _create_time_weighted_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-weighted rolling averages
        
        More recent games weighted higher using exponential decay
        """
        if history_df.empty:
            return pd.DataFrame()
        
        features = []
        
        # Group by player
        for player_id, player_history in history_df.groupby('element'):
            # Sort by gameweek (most recent first)
            player_history = player_history.sort_values('round', ascending=False)
            
            # Limit to last 10 games
            player_history = player_history.head(10)
            
            if len(player_history) == 0:
                continue
            
            # Calculate weights (exponential decay)
            n_games = len(player_history)
            weights = np.array([self.decay_factor ** i for i in range(n_games)])
            weights = weights / weights.sum()  # Normalize
            
            # Time-weighted averages
            tw_points = np.average(player_history['total_points'], weights=weights)
            tw_minutes = np.average(player_history['minutes'], weights=weights)
            tw_xg = np.average(player_history.get('xG', player_history['total_points'] * 0.1), weights=weights)
            tw_xa = np.average(player_history.get('xA', player_history['assists'] * 1.5), weights=weights)
            tw_bonus = np.average(player_history['bonus'], weights=weights)
            tw_bps = np.average(player_history['bps'], weights=weights)
            
            features.append({
                'id': player_id,
                'tw_points': round(tw_points, 2),
                'tw_minutes': round(tw_minutes, 1),
                'tw_xg': round(tw_xg, 2),
                'tw_xa': round(tw_xa, 2),
                'tw_bonus': round(tw_bonus, 2),
                'tw_bps': round(tw_bps, 1),
                'tw_games_played': n_games
            })
        
        return pd.DataFrame(features)
    
    def _create_opponent_adjusted_features(
        self,
        players_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adjust player metrics based on opponent strength
        """
        features = []
        
        for _, player in players_df.iterrows():
            player_id = player['id']
            team_id = player.get('team', player.get('team_id'))
            
            # Find next fixture
            next_fixture = fixtures_df[
                ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
                (fixtures_df['finished'] == False)
            ].sort_values('kickoff_time').head(1)
            
            if next_fixture.empty:
                features.append({
                    'id': player_id,
                    'opp_adj_points': player.get('form', 0),
                    'opp_adj_xg': player.get('expected_goals', 0),
                    'opp_difficulty': 3  # Neutral
                })
                continue
            
            fixture = next_fixture.iloc[0]
            is_home = fixture['team_h'] == team_id
            opponent_id = fixture['team_a'] if is_home else fixture['team_h']
            
            # Get opponent strength
            opponent = teams_df[teams_df['id'] == opponent_id]
            if opponent.empty:
                opp_strength = 100  # Neutral
            else:
                opp_strength = opponent.iloc[0].get(
                    'strength_defence_away' if is_home else 'strength_defence_home',
                    100
                )
            
            # Adjustment factor (inverse of opponent strength)
            adjustment = 100 / max(opp_strength, 1)
            
            # Adjusted metrics
            base_points = player.get('form', 0)
            base_xg = player.get('expected_goals', 0)
            
            opp_adj_points = base_points * adjustment
            opp_adj_xg = base_xg * adjustment
            
            features.append({
                'id': player_id,
                'opp_adj_points': round(opp_adj_points, 2),
                'opp_adj_xg': round(opp_adj_xg, 2),
                'opp_difficulty': fixture.get('team_h_difficulty' if is_home else 'team_a_difficulty', 3)
            })
        
        return pd.DataFrame(features)
    
    def _create_congestion_features(
        self,
        players_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        history_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate fixture congestion indicators
        """
        features = []
        
        for _, player in players_df.iterrows():
            player_id = player['id']
            team_id = player.get('team', player.get('team_id'))
            
            # Get team fixtures
            team_fixtures = fixtures_df[
                (fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)
            ].sort_values('kickoff_time')
            
            # Count upcoming fixtures in next 14 days
            now = pd.Timestamp.now()
            upcoming = team_fixtures[
                (team_fixtures['kickoff_time'] >= now) &
                (team_fixtures['kickoff_time'] <= now + pd.Timedelta(days=14))
            ]
            
            fixtures_next_14_days = len(upcoming)
            
            # Get player's recent minutes
            player_history = history_df[history_df['element'] == player_id].sort_values('round', ascending=False).head(5)
            
            if not player_history.empty:
                recent_minutes = player_history['minutes'].sum()
                avg_minutes_per_game = player_history['minutes'].mean()
                
                # Calculate rest days since last game
                if len(player_history) > 0:
                    # Estimate last game was most recent gameweek
                    rest_days = 7  # Default assumption
                else:
                    rest_days = 14
            else:
                recent_minutes = 0
                avg_minutes_per_game = 0
                rest_days = 14
            
            # Congestion score (higher = more congested)
            congestion_score = (
                fixtures_next_14_days * 0.4 +
                (recent_minutes / 450) * 0.4 +  # Normalize to 5 full games
                (1 - rest_days / 14) * 0.2
            )
            
            features.append({
                'id': player_id,
                'cong_fixtures_14d': fixtures_next_14_days,
                'cong_recent_minutes': recent_minutes,
                'cong_rest_days': rest_days,
                'cong_score': round(congestion_score, 2)
            })
        
        return pd.DataFrame(features)
    
    def _create_home_away_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate separate performance metrics for home vs away
        """
        if history_df.empty:
            return pd.DataFrame()
        
        features = []
        
        for player_id, player_history in history_df.groupby('element'):
            # Separate home and away games
            home_games = player_history[player_history['was_home'] == True]
            away_games = player_history[player_history['was_home'] == False]
            
            # Home stats
            if not home_games.empty:
                ha_home_points = home_games['total_points'].mean()
                ha_home_minutes = home_games['minutes'].mean()
                ha_home_games = len(home_games)
            else:
                ha_home_points = 0
                ha_home_minutes = 0
                ha_home_games = 0
            
            # Away stats
            if not away_games.empty:
                ha_away_points = away_games['total_points'].mean()
                ha_away_minutes = away_games['minutes'].mean()
                ha_away_games = len(away_games)
            else:
                ha_away_points = 0
                ha_away_minutes = 0
                ha_away_games = 0
            
            # Home advantage (difference)
            ha_advantage = ha_home_points - ha_away_points
            
            features.append({
                'id': player_id,
                'ha_home_points': round(ha_home_points, 2),
                'ha_away_points': round(ha_away_points, 2),
                'ha_home_minutes': round(ha_home_minutes, 1),
                'ha_away_minutes': round(ha_away_minutes, 1),
                'ha_advantage': round(ha_advantage, 2),
                'ha_home_games': ha_home_games,
                'ha_away_games': ha_away_games
            })
        
        return pd.DataFrame(features)
    
    def _create_momentum_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators (form trends)
        """
        if history_df.empty:
            return pd.DataFrame()
        
        features = []
        
        for player_id, player_history in history_df.groupby('element'):
            # Sort by gameweek
            player_history = player_history.sort_values('round', ascending=False)
            
            # Limit to last 6 games
            recent = player_history.head(6)
            
            if len(recent) < 3:
                # Not enough data
                features.append({
                    'id': player_id,
                    'mom_form_trend': 0,
                    'mom_points_streak': 0,
                    'mom_consistency': 0
                })
                continue
            
            # Form trend (linear regression slope)
            points = recent['total_points'].values
            x = np.arange(len(points))
            if len(points) > 1:
                slope = np.polyfit(x, points, 1)[0]
                form_trend = slope
            else:
                form_trend = 0
            
            # Points streak (consecutive games with points > 0)
            streak = 0
            for pts in points:
                if pts > 0:
                    streak += 1
                else:
                    break
            
            # Consistency (inverse of standard deviation)
            if len(points) > 1:
                std = np.std(points)
                consistency = 1 / (std + 1)  # Add 1 to avoid division by zero
            else:
                consistency = 0
            
            features.append({
                'id': player_id,
                'mom_form_trend': round(form_trend, 2),
                'mom_points_streak': streak,
                'mom_consistency': round(consistency, 2)
            })
        
        return pd.DataFrame(features)


# Example usage
if __name__ == "__main__":
    # Test with sample data
    engineer = AdvancedFeatureEngineer()
    
    # Sample players
    players_df = pd.DataFrame({
        'id': [1, 2, 3],
        'team': [1, 2, 3],
        'form': [5.5, 4.2, 6.1],
        'expected_goals': [0.5, 0.3, 0.7]
    })
    
    # Sample history
    history_df = pd.DataFrame({
        'element': [1, 1, 1, 2, 2, 3],
        'round': [10, 9, 8, 10, 9, 10],
        'total_points': [8, 6, 10, 4, 5, 12],
        'minutes': [90, 85, 90, 60, 75, 90],
        'bonus': [2, 1, 3, 0, 1, 3],
        'bps': [45, 32, 58, 20, 25, 65],
        'assists': [1, 0, 2, 1, 0, 2],
        'was_home': [True, False, True, False, True, True]
    })
    
    # Sample fixtures
    fixtures_df = pd.DataFrame({
        'team_h': [1, 2],
        'team_a': [3, 1],
        'kickoff_time': [pd.Timestamp.now() + pd.Timedelta(days=3), pd.Timestamp.now() + pd.Timedelta(days=10)],
        'finished': [False, False],
        'team_h_difficulty': [3, 4],
        'team_a_difficulty': [3, 2]
    })
    
    # Sample teams
    teams_df = pd.DataFrame({
        'id': [1, 2, 3],
        'strength_defence_home': [100, 90, 110],
        'strength_defence_away': [95, 85, 105]
    })
    
    # Create features
    result = engineer.create_features(players_df, history_df, fixtures_df, teams_df)
    
    print("\nAdvanced Features Created:")
    feature_cols = [col for col in result.columns if col.startswith(('tw_', 'opp_', 'cong_', 'ha_', 'mom_'))]
    print(result[['id'] + feature_cols])
