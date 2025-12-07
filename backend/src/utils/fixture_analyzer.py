"""
Advanced fixture analysis for FPL optimization
Handles fixture difficulty, double gameweeks, blank gameweeks, and rotation risk
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

class FixtureAnalyzer:
    """Advanced fixture analysis for FPL optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.head_to_head_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self.max_head_to_head_samples = 5
        
        # Fixture difficulty weights (1-5 scale)
        self.difficulty_weights = {
            1: 0.2,  # Very easy
            2: 0.4,  # Easy
            3: 0.6,  # Medium
            4: 0.8,  # Hard
            5: 1.0   # Very hard
        }
        
        # Position-specific difficulty adjustments
        self.position_difficulty_multipliers = {
            'GKP': {'attack': 0.8, 'defence': 1.2},
            'DEF': {'attack': 0.7, 'defence': 1.3},
            'MID': {'attack': 1.0, 'defence': 1.0},
            'FWD': {'attack': 1.2, 'defence': 0.8}
        }
    
    def _get_upcoming_window(self, fixtures_df: pd.DataFrame, gameweeks: int) -> pd.DataFrame:
        """Return the slice of fixtures covering the next `gameweeks` unfinished rounds."""
        fixtures = fixtures_df.copy()
        if fixtures.empty:
            return fixtures
        
        fixtures['finished'] = fixtures['finished'].astype(int)
        unfinished = fixtures[fixtures['finished'] == 0]
        
        if not unfinished.empty:
            next_event = unfinished['event'].min()
            max_event = next_event + gameweeks - 1
            return fixtures[(fixtures['event'] >= next_event) & (fixtures['event'] <= max_event)]
        
        # All fixtures finished (e.g., during off-season) - fallback to latest window
        max_event = fixtures['event'].max()
        min_event = max(fixtures['event'].min(), max_event - gameweeks + 1)
        return fixtures[(fixtures['event'] >= min_event) & (fixtures['event'] <= max_event)]
    
    def calculate_fixture_difficulty_score(self, 
                                         fixtures_df: pd.DataFrame, 
                                         team_strengths_df: pd.DataFrame,
                                         gameweeks: int = 5) -> pd.DataFrame:
        """Calculate comprehensive fixture difficulty scores"""
        try:
            if fixtures_df.empty or team_strengths_df.empty:
                return pd.DataFrame()
            
            upcoming = self._get_upcoming_window(fixtures_df, gameweeks)
            upcoming = upcoming[upcoming['finished'] == 0].copy()
            
            if upcoming.empty:
                return pd.DataFrame()
            
            # Merge team strengths
            upcoming = upcoming.merge(
                team_strengths_df[['id', 'strength_overall_home', 'strength_overall_away',
                                 'strength_attack_home', 'strength_attack_away',
                                 'strength_defence_home', 'strength_defence_away']],
                left_on='team_h', right_on='id', how='left', suffixes=('', '_home')
            )
            
            upcoming = upcoming.merge(
                team_strengths_df[['id', 'strength_overall_home', 'strength_overall_away',
                                 'strength_attack_home', 'strength_attack_away',
                                 'strength_defence_home', 'strength_defence_away']],
                left_on='team_a', right_on='id', how='left', suffixes=('', '_away')
            )
            
            # Calculate base difficulty scores
            upcoming['home_attack_vs_away_defence'] = (
                upcoming['strength_attack_home'] - upcoming['strength_defence_away_away']
            )
            upcoming['away_attack_vs_home_defence'] = (
                upcoming['strength_attack_away_away'] - upcoming['strength_defence_home']
            )
            
            # Overall difficulty (1-5 scale)
            upcoming['home_difficulty'] = upcoming['team_h_difficulty']
            upcoming['away_difficulty'] = upcoming['team_a_difficulty']
            
            # Calculate weighted difficulty scores
            upcoming['home_difficulty_score'] = upcoming['home_difficulty'].map(self.difficulty_weights)
            upcoming['away_difficulty_score'] = upcoming['away_difficulty'].map(self.difficulty_weights)
            
            # Add fixture timing analysis
            upcoming['kickoff_time'] = pd.to_datetime(upcoming['kickoff_time'])
            upcoming['days_since_last'] = self._calculate_rest_days(upcoming)
            upcoming['is_midweek'] = upcoming['kickoff_time'].dt.dayofweek.isin([1, 2, 3])
            
            # Rotation risk indicators
            upcoming['rotation_risk'] = self._calculate_rotation_risk(upcoming)
            
            return upcoming
            
        except Exception as e:
            self.logger.error(f"Error calculating fixture difficulty: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_rest_days(self, fixtures_df: pd.DataFrame) -> pd.Series:
        """Calculate days since last fixture for each team"""
        try:
            rest_days = []
            
            for team_col in ['team_h', 'team_a']:
                team_rest = []
                for team_id in fixtures_df[team_col].unique():
                    team_fixtures = fixtures_df[fixtures_df[team_col] == team_id].sort_values('kickoff_time')
                    team_rest.extend(team_fixtures['kickoff_time'].diff().dt.days.fillna(7))
                
                rest_days.extend(team_rest)
            
            return pd.Series(rest_days)
            
        except Exception as e:
            self.logger.error(f"Error calculating rest days: {str(e)}")
            return pd.Series([7] * len(fixtures_df))
    
    def _calculate_rotation_risk(self, fixtures_df: pd.DataFrame) -> pd.Series:
        """Calculate rotation risk based on fixture congestion"""
        try:
            risk_factors = []
            
            for _, row in fixtures_df.iterrows():
                risk = 0
                
                # Midweek games increase rotation risk
                if row.get('is_midweek', False):
                    risk += 0.3
                
                # Short rest increases risk
                rest_days = row.get('days_since_last', 7)
                if rest_days < 3:
                    risk += 0.4
                elif rest_days < 5:
                    risk += 0.2
                
                # Multiple games in short period
                team_id = row.get('team_h', 0)
                if team_id:
                    team_games = fixtures_df[
                        (fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)
                    ]
                    if len(team_games) > 1:
                        time_span = (team_games['kickoff_time'].max() - team_games['kickoff_time'].min()).days
                        if time_span < 7:
                            risk += 0.3
                
                risk_factors.append(min(1.0, risk))
            
            return pd.Series(risk_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating rotation risk: {str(e)}")
            return pd.Series([0.1] * len(fixtures_df))
    
    def detect_double_gameweeks(self, fixtures_df: pd.DataFrame) -> Dict[int, List[int]]:
        """Detect teams with double gameweeks"""
        try:
            double_gws = {}
            
            for gw in fixtures_df['event'].unique():
                gw_fixtures = fixtures_df[fixtures_df['event'] == gw]
                
                # Count fixtures per team
                team_counts = {}
                for _, fixture in gw_fixtures.iterrows():
                    for team_col in ['team_h', 'team_a']:
                        team_id = fixture[team_col]
                        team_counts[team_id] = team_counts.get(team_id, 0) + 1
                
                # Teams with 2+ fixtures have double gameweek
                double_teams = [team_id for team_id, count in team_counts.items() if count >= 2]
                if double_teams:
                    double_gws[gw] = double_teams
            
            return double_gws
            
        except Exception as e:
            self.logger.error(f"Error detecting double gameweeks: {str(e)}")
            return {}
    
    def detect_blank_gameweeks(self, fixtures_df: pd.DataFrame) -> List[int]:
        """Detect blank gameweeks (no fixtures)"""
        try:
            blank_gws = []
            
            for gw in fixtures_df['event'].unique():
                gw_fixtures = fixtures_df[fixtures_df['event'] == gw]
                if gw_fixtures.empty:
                    blank_gws.append(gw)
            
            return blank_gws
            
        except Exception as e:
            self.logger.error(f"Error detecting blank gameweeks: {str(e)}")
            return []
    
    def calculate_position_fixture_difficulty(self, 
                                            fixtures_df: pd.DataFrame,
                                            team_strengths_df: pd.DataFrame,
                                            position: str) -> pd.DataFrame:
        """Calculate position-specific fixture difficulty"""
        try:
            difficulty_df = self.calculate_fixture_difficulty_score(fixtures_df, team_strengths_df)
            
            if difficulty_df.empty:
                return pd.DataFrame()
            
            # Apply position-specific multipliers
            multipliers = self.position_difficulty_multipliers.get(position, {'attack': 1.0, 'defence': 1.0})
            
            # Adjust difficulty based on position
            difficulty_df['position_difficulty_home'] = (
                difficulty_df['home_difficulty_score'] * multipliers['attack'] +
                difficulty_df['home_difficulty_score'] * multipliers['defence']
            ) / 2
            
            difficulty_df['position_difficulty_away'] = (
                difficulty_df['away_difficulty_score'] * multipliers['attack'] +
                difficulty_df['away_difficulty_score'] * multipliers['defence']
            ) / 2
            
            return difficulty_df
            
        except Exception as e:
            self.logger.error(f"Error calculating position fixture difficulty: {str(e)}")
            return pd.DataFrame()
    
    def _compute_head_to_head(self, fixtures_df: pd.DataFrame, home_team: int, away_team: int) -> Dict[str, Any]:
        """Compute last N head-to-head stats for the pair."""
        cache_key = (home_team, away_team, self.max_head_to_head_samples)
        if cache_key in self.head_to_head_cache:
            return self.head_to_head_cache[cache_key]
        
        past_meetings = fixtures_df[
            (
                ((fixtures_df['team_h'] == home_team) & (fixtures_df['team_a'] == away_team)) |
                ((fixtures_df['team_h'] == away_team) & (fixtures_df['team_a'] == home_team))
            ) &
            (fixtures_df['finished'] == 1)
        ].sort_values('event', ascending=False).head(self.max_head_to_head_samples)
        
        stats = {
            'samples': len(past_meetings),
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0,
            'home_goals_avg': 0.0,
            'away_goals_avg': 0.0,
            'trend': 'Balanced'
        }
        
        if not past_meetings.empty:
            home_wins = 0
            away_wins = 0
            draws = 0
            home_goals = []
            away_goals = []
            
            for _, row in past_meetings.iterrows():
                home_score = row.get('team_h_score', 0) or 0
                away_score = row.get('team_a_score', 0) or 0
                home_is_home = row['team_h'] == home_team
                
                if home_is_home:
                    home_goals.append(home_score)
                    away_goals.append(away_score)
                    if home_score > away_score:
                        home_wins += 1
                    elif away_score > home_score:
                        away_wins += 1
                    else:
                        draws += 1
                else:
                    # Swap roles
                    home_goals.append(away_score)
                    away_goals.append(home_score)
                    if away_score > home_score:
                        home_wins += 1
                    elif home_score > away_score:
                        away_wins += 1
                    else:
                        draws += 1
            
            stats['home_wins'] = home_wins
            stats['away_wins'] = away_wins
            stats['draws'] = draws
            stats['home_goals_avg'] = round(np.mean(home_goals), 2) if home_goals else 0.0
            stats['away_goals_avg'] = round(np.mean(away_goals), 2) if away_goals else 0.0
            
            if home_wins > away_wins:
                stats['trend'] = 'Home edge'
            elif away_wins > home_wins:
                stats['trend'] = 'Away edge'
            else:
                stats['trend'] = 'Balanced'
        
        self.head_to_head_cache[cache_key] = stats
        return stats
    
    def get_team_fixture_summary(self, 
                                team_id: int, 
                                fixtures_df: pd.DataFrame,
                                team_strengths_df: pd.DataFrame,
                                gameweeks: int = 5) -> Dict:
        """Get comprehensive fixture summary for a team"""
        try:
            upcoming = self._get_upcoming_window(fixtures_df, gameweeks)
            upcoming = upcoming[upcoming['finished'] == 0]
            
            team_fixtures = upcoming[
                (upcoming['team_h'] == team_id) | (upcoming['team_a'] == team_id)
            ]
            
            if team_fixtures.empty:
                return {}
            
            # Calculate difficulty scores
            difficulty_df = self.calculate_fixture_difficulty_score(fixtures_df, team_strengths_df, gameweeks)
            team_difficulty = difficulty_df[
                (difficulty_df['team_h'] == team_id) | (difficulty_df['team_a'] == team_id)
            ]
            
            # Calculate metrics
            avg_difficulty = team_difficulty[['home_difficulty_score', 'away_difficulty_score']].mean().mean() if not team_difficulty.empty else 3.0
            double_gws = self.detect_double_gameweeks(fixtures_df)
            has_double_gw = any(team_id in teams for teams in double_gws.values())
            
            # Rotation risk
            avg_rotation_risk = team_difficulty['rotation_risk'].mean() if not team_difficulty.empty else 0.1
            
            # Home vs away split
            home_games = team_fixtures[team_fixtures['team_h'] == team_id]
            away_games = team_fixtures[team_fixtures['team_a'] == team_id]
            home_ratio = len(home_games) / max(len(team_fixtures), 1)
            
            # Next fixture info
            next_fixture = team_fixtures.sort_values(['event', 'kickoff_time']).iloc[0]
            next_venue = 'home' if next_fixture['team_h'] == team_id else 'away'
            opponent_team = int(next_fixture['team_a'] if next_venue == 'home' else next_fixture['team_h'])
            next_difficulty = next_fixture['team_h_difficulty'] if next_venue == 'home' else next_fixture['team_a_difficulty']
            venue_multiplier = 1.1 if next_venue == 'home' else 0.9
            
            head_to_head = self._compute_head_to_head(fixtures_df, team_id, opponent_team)
            
            return {
                'team_id': team_id,
                'avg_difficulty': round(avg_difficulty, 2),
                'has_double_gameweek': has_double_gw,
                'rotation_risk': round(avg_rotation_risk, 2),
                'fixture_count': len(team_fixtures),
                'home_fixtures': len(home_games),
                'away_fixtures': len(away_games),
                'home_ratio': round(home_ratio, 2),
                'next_fixture': {
                    'venue': next_venue,
                    'difficulty': next_difficulty,
                    'event': int(next_fixture['event']),
                    'opponent_team': opponent_team,
                    'fixture_id': int(next_fixture['id']),
                    'venue_multiplier': venue_multiplier,
                    'head_to_head': head_to_head
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting team fixture summary: {str(e)}")
            return {}
    
    def calculate_fixture_weighted_form(self, 
                                      player_history: pd.DataFrame,
                                      fixtures_df: pd.DataFrame,
                                      team_strengths_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate form weighted by upcoming fixture difficulty"""
        try:
            if player_history.empty or fixtures_df.empty:
                return pd.DataFrame()
            
            # Get recent form (last 5 games)
            recent_form = player_history.groupby('element').tail(5)
            
            # Calculate fixture difficulty for each team
            difficulty_scores = {}
            for team_id in recent_form['team'].unique():
                team_summary = self.get_team_fixture_summary(team_id, fixtures_df, team_strengths_df)
                difficulty_scores[team_id] = team_summary.get('avg_difficulty', 3.0)
            
            # Weight form by fixture difficulty
            recent_form['fixture_difficulty'] = recent_form['team'].map(difficulty_scores)
            recent_form['weighted_points'] = recent_form['total_points'] * (1 / recent_form['fixture_difficulty'])
            
            # Calculate weighted form
            weighted_form = recent_form.groupby('element').agg({
                'weighted_points': 'mean',
                'total_points': 'mean',
                'fixture_difficulty': 'mean'
            }).reset_index()
            
            weighted_form['form_ratio'] = weighted_form['weighted_points'] / weighted_form['total_points']
            
            return weighted_form
            
        except Exception as e:
            self.logger.error(f"Error calculating fixture weighted form: {str(e)}")
            return pd.DataFrame()
