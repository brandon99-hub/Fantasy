import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

class FeatureEngineer:
    """Create features for FPL prediction models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_form_features(self, player_history: pd.DataFrame, window_sizes: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """Create rolling form features from player history"""
        try:
            features = pd.DataFrame()
            
            if player_history.empty:
                return features
            
            # Sort by player and gameweek
            history = player_history.sort_values(['element', 'round'])
            
            for window in window_sizes:
                # Rolling averages
                history[f'points_last_{window}'] = history.groupby('element')['total_points'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                history[f'minutes_last_{window}'] = history.groupby('element')['minutes'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                history[f'bps_last_{window}'] = history.groupby('element')['bps'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling sums for goals/assists
                history[f'goals_last_{window}'] = history.groupby('element')['goals_scored'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum()
                )
                
                history[f'assists_last_{window}'] = history.groupby('element')['assists'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum()
                )
            
            # Form trends
            history['points_trend'] = history.groupby('element')['total_points'].transform(
                lambda x: x.diff().rolling(window=3, min_periods=1).mean()
            )
            
            # Consistency measures
            for window in [5, 10]:
                history[f'points_std_{window}'] = history.groupby('element')['total_points'].transform(
                    lambda x: x.rolling(window=window, min_periods=2).std().fillna(0)
                )
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error creating form features: {str(e)}")
            return pd.DataFrame()
    
    def create_fixture_features(self, fixtures_df: pd.DataFrame, team_strengths: pd.DataFrame) -> pd.DataFrame:
        """Create fixture difficulty and scheduling features"""
        try:
            features = fixtures_df.copy()
            
            if features.empty or team_strengths.empty:
                return features
            
            # Merge team strengths
            features = features.merge(
                team_strengths[['id', 'strength_overall_home', 'strength_overall_away', 
                              'strength_attack_home', 'strength_attack_away',
                              'strength_defence_home', 'strength_defence_away']],
                left_on='team_h', right_on='id', how='left', suffixes=('', '_home')
            )
            
            features = features.merge(
                team_strengths[['id', 'strength_overall_home', 'strength_overall_away', 
                              'strength_attack_home', 'strength_attack_away',
                              'strength_defence_home', 'strength_defence_away']],
                left_on='team_a', right_on='id', how='left', suffixes=('', '_away')
            )
            
            # Fixture difficulty rating calculations
            features['home_attack_vs_away_defence'] = (
                features['strength_attack_home'] - features['strength_defence_away_away']
            )
            features['away_attack_vs_home_defence'] = (
                features['strength_attack_away_away'] - features['strength_defence_home']
            )
            
            # Overall fixture difficulty
            features['home_difficulty_rating'] = features['team_h_difficulty']
            features['away_difficulty_rating'] = features['team_a_difficulty']
            
            # Double gameweek indicators
            features['is_double_gameweek'] = features.groupby(['team_h', 'event']).size() > 1
            features['is_double_gameweek'] |= features.groupby(['team_a', 'event']).size() > 1
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating fixture features: {str(e)}")
            return fixtures_df
    
    def create_schedule_features(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to fixture scheduling and rotation risk"""
        try:
            features = fixtures_df.copy()
            
            if features.empty:
                return features
            
            # Convert kickoff times
            if 'kickoff_time' in features.columns:
                features['kickoff_time'] = pd.to_datetime(features['kickoff_time'])
                
                # Day of week (midweek games)
                features['day_of_week'] = features['kickoff_time'].dt.dayofweek
                features['is_midweek'] = features['day_of_week'].isin([1, 2, 3])  # Tue, Wed, Thu
                
                # Time between fixtures
                features = features.sort_values(['team_h', 'kickoff_time'])
                features['days_since_last_game_home'] = features.groupby('team_h')['kickoff_time'].diff().dt.days
                
                features = features.sort_values(['team_a', 'kickoff_time'])
                features['days_since_last_game_away'] = features.groupby('team_a')['kickoff_time'].diff().dt.days
                
                # Fixtures in next 7 days
                current_date = features['kickoff_time'].min()
                week_window = current_date + timedelta(days=7)
                
                for team_col in ['team_h', 'team_a']:
                    team_fixtures = features[features['kickoff_time'] <= week_window].groupby(team_col).size()
                    features[f'fixtures_next_7_days_{team_col[-1]}'] = features[team_col].map(team_fixtures).fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating schedule features: {str(e)}")
            return fixtures_df
    
    def create_player_position_features(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features"""
        try:
            features = players_df.copy()
            
            # Position encoding
            features['is_goalkeeper'] = (features['position'] == 'GKP').astype(int)
            features['is_defender'] = (features['position'] == 'DEF').astype(int)
            features['is_midfielder'] = (features['position'] == 'MID').astype(int)
            features['is_forward'] = (features['position'] == 'FWD').astype(int)
            
            # Position-specific performance metrics
            if 'goals_scored' in features.columns and 'assists' in features.columns:
                # Goal involvement
                features['goal_involvement'] = features['goals_scored'] + features['assists']
                
                # Position-adjusted metrics
                features['goals_per_position'] = np.where(
                    features['position'] == 'FWD', features['goals_scored'],
                    np.where(features['position'] == 'MID', features['goals_scored'] * 1.25,
                            np.where(features['position'] == 'DEF', features['goals_scored'] * 1.5, 0))
                )
            
            # Clean sheet relevance
            if 'clean_sheets' in features.columns:
                features['clean_sheet_relevance'] = np.where(
                    features['position'].isin(['GKP', 'DEF']), features['clean_sheets'],
                    np.where(features['position'] == 'MID', features['clean_sheets'] * 0.25, 0)
                )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating position features: {str(e)}")
            return players_df
    
    def create_ownership_features(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on player ownership and transfers"""
        try:
            features = players_df.copy()
            
            # Ownership percentiles
            if 'selected_by_percent' in features.columns:
                features['ownership_percentile'] = features['selected_by_percent'].rank(pct=True)
                
                # Ownership categories
                features['is_highly_owned'] = (features['selected_by_percent'] > 20).astype(int)
                features['is_differential'] = (features['selected_by_percent'] < 5).astype(int)
            
            # Transfer trends
            if 'transfers_in_event' in features.columns and 'transfers_out_event' in features.columns:
                features['net_transfers'] = features['transfers_in_event'] - features['transfers_out_event']
                features['transfer_momentum'] = features['net_transfers'] / (features['selected_by_percent'] + 1)
            
            # Price change indicators
            if 'value_form' in features.columns and 'value_season' in features.columns:
                features['price_change_risk'] = np.where(
                    features['value_form'] > features['value_season'], 1, 0
                )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating ownership features: {str(e)}")
            return players_df
    
    def create_team_features(self, players_df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Create team-level features for players"""
        try:
            features = players_df.copy()
            
            if team_stats.empty:
                return features
            
            # Merge team statistics
            team_cols = ['id', 'strength_overall_home', 'strength_overall_away',
                        'strength_attack_home', 'strength_attack_away',
                        'strength_defence_home', 'strength_defence_away']
            
            available_cols = [col for col in team_cols if col in team_stats.columns]
            
            features = features.merge(
                team_stats[available_cols],
                left_on='team', right_on='id', how='left', suffixes=('', '_team')
            )
            
            # Team quality indicators
            if 'strength_overall_home' in features.columns:
                features['team_quality'] = (
                    features['strength_overall_home'] + features['strength_overall_away']
                ) / 2
                
                features['attacking_team'] = (
                    features['strength_attack_home'] + features['strength_attack_away']
                ) / 2
                
                features['defensive_team'] = (
                    features['strength_defence_home'] + features['strength_defence_away']
                ) / 2
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating team features: {str(e)}")
            return players_df
    
    def create_injury_availability_features(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to player availability and injury risk"""
        try:
            features = players_df.copy()
            
            # Availability percentage
            if 'chance_of_playing_this_round' in features.columns:
                features['availability_pct'] = features['chance_of_playing_this_round'].fillna(100) / 100
                features['injury_risk'] = 1 - features['availability_pct']
                
                # Risk categories
                features['availability_category'] = pd.cut(
                    features['availability_pct'],
                    bins=[0, 0.25, 0.75, 1.0],
                    labels=['High Risk', 'Medium Risk', 'Available'],
                    include_lowest=True
                )
            
            # Status flags
            if 'status' in features.columns:
                features['is_available'] = (features['status'] == 'a').astype(int)
                features['is_injured'] = (features['status'] == 'i').astype(int)
                features['is_suspended'] = (features['status'] == 's').astype(int)
                features['is_unavailable'] = (features['status'] == 'u').astype(int)
            
            # News sentiment (simple keyword matching)
            if 'news' in features.columns:
                injury_keywords = ['injury', 'injured', 'doubt', 'fitness', 'knock', 'strain', 'suspended']
                positive_keywords = ['fit', 'available', 'training', 'ready']
                
                features['news_negative'] = features['news'].fillna('').str.lower().str.contains(
                    '|'.join(injury_keywords), regex=True
                ).astype(int)
                
                features['news_positive'] = features['news'].fillna('').str.lower().str.contains(
                    '|'.join(positive_keywords), regex=True
                ).astype(int)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating availability features: {str(e)}")
            return players_df
    
    def create_all_features(self, 
                           players_df: pd.DataFrame,
                           history_df: pd.DataFrame = None,
                           fixtures_df: pd.DataFrame = None,
                           team_stats: pd.DataFrame = None) -> pd.DataFrame:
        """Create comprehensive feature set for modeling"""
        try:
            self.logger.info("Creating comprehensive feature set...")
            
            features = players_df.copy()
            
            # Position features
            features = self.create_player_position_features(features)
            
            # Ownership features
            features = self.create_ownership_features(features)
            
            # Injury/availability features
            features = self.create_injury_availability_features(features)
            
            # Team features
            if team_stats is not None and not team_stats.empty:
                features = self.create_team_features(features, team_stats)
            
            # Form features (if history available)
            if history_df is not None and not history_df.empty:
                form_features = self.create_form_features(history_df)
                if not form_features.empty:
                    # Get latest form data for each player
                    latest_form = form_features.groupby('element').last().reset_index()
                    features = features.merge(
                        latest_form[['element'] + [col for col in latest_form.columns if col.startswith(('points_', 'minutes_', 'goals_', 'assists_'))]],
                        left_on='id', right_on='element', how='left'
                    )
            
            self.logger.info(f"Created {len(features.columns)} features for {len(features)} players")
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive features: {str(e)}")
            return players_df
