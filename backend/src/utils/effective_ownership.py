"""
Effective Ownership (EO) tracking and template team analysis
Manages risk between template vs differential picks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from backend.src.core.manager_training_data import ManagerTrainingDataBuilder

@dataclass
class EOAnalysis:
    player_id: int
    player_name: str
    position: str
    ownership_percent: float
    effective_ownership: float
    template_status: str
    risk_level: str
    differential_value: float

class EffectiveOwnershipTracker:
    """Track effective ownership and template team analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._manager_meta_builder = ManagerTrainingDataBuilder()
        
        # Ownership thresholds
        self.ownership_thresholds = {
            'template': 20.0,      # >20% = template player
            'high_owned': 10.0,    # 10-20% = high ownership
            'medium_owned': 5.0,   # 5-10% = medium ownership
            'differential': 5.0     # <5% = differential
        }
        
        # Risk levels based on ownership
        self.risk_levels = {
            'template': 'Low Risk',
            'high_owned': 'Low-Medium Risk',
            'medium_owned': 'Medium Risk',
            'differential': 'High Risk'
        }
    
    def calculate_effective_ownership(
        self,
        players_df: pd.DataFrame,
        top_managers_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Calculate effective ownership for players"""
        try:
            if players_df.empty:
                return pd.DataFrame()
            
            # Start with basic ownership
            eo_data = players_df[
                ["id", "web_name", "position", "selected_by_percent"]
            ].copy()

            # Merge top cohort metrics if available
            if top_managers_df is not None and not top_managers_df.empty:
                eo_data = self._calculate_weighted_eo(eo_data, top_managers_df)
            else:
                # Try to derive from manager history tables
                try:
                    meta = self._manager_meta_builder.build_player_meta_features(force_refresh=False)
                    if not meta.empty:
                        # If players_df has a gameweek column, align on it; otherwise, use latest event
                        event_col = None
                        for candidate in ("event", "gameweek", "round"):
                            if candidate in players_df.columns:
                                event_col = candidate
                                break

                        if event_col is not None:
                            latest_meta = (
                                meta.sort_values("event")
                                .groupby(["player_id", "event"])
                                .tail(1)
                            )
                            # Ensure player_id is int64 to match id column
                            if 'player_id' in latest_meta.columns:
                                latest_meta = latest_meta.copy()
                                latest_meta['player_id'] = pd.to_numeric(latest_meta['player_id'], errors='coerce').astype('Int64')
                            eo_data = eo_data.merge(
                                latest_meta[
                                    [
                                        "player_id",
                                        "event",
                                        "top_cohort_ownership_pct",
                                        "top_cohort_captain_pct",
                                    ]
                                ],
                                left_on=["id", event_col],
                                right_on=["player_id", "event"],
                                how="left",
                            )
                        else:
                            latest_meta = (
                                meta.sort_values("event")
                                .groupby("player_id")
                                .tail(1)
                            )
                            # Ensure player_id is int64 to match id column
                            if 'player_id' in latest_meta.columns:
                                latest_meta = latest_meta.copy()
                                latest_meta['player_id'] = pd.to_numeric(latest_meta['player_id'], errors='coerce').astype('Int64')
                            eo_data = eo_data.merge(
                                latest_meta[
                                    [
                                        "player_id",
                                        "top_cohort_ownership_pct",
                                        "top_cohort_captain_pct",
                                    ]
                                ],
                                left_on="id",
                                right_on="player_id",
                                how="left",
                            )

                        eo_data["effective_ownership"] = eo_data[
                            "top_cohort_ownership_pct"
                        ].fillna(eo_data["selected_by_percent"])
                        # Drop helper keys
                        for col in ("player_id", "event"):
                            if col in eo_data.columns:
                                eo_data.drop(columns=[col], inplace=True)
                    else:
                        eo_data["effective_ownership"] = eo_data["selected_by_percent"]
                except Exception as exc:
                    self.logger.error(
                        f"Error deriving EO from manager history tables: {exc}"
                    )
                    eo_data["effective_ownership"] = eo_data["selected_by_percent"]
            
            # Classify ownership levels
            eo_data["template_status"] = eo_data["selected_by_percent"].apply(
                self._classify_ownership
            )
            eo_data["risk_level"] = eo_data["template_status"].map(self.risk_levels)
            
            # Calculate differential value (inverse of ownership)
            eo_data["differential_value"] = 100 - eo_data["selected_by_percent"]
            
            return eo_data
            
        except Exception as e:
            self.logger.error(f"Error calculating effective ownership: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_weighted_eo(self, eo_data: pd.DataFrame, top_managers_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted effective ownership based on top managers"""
        try:
            # This would integrate with top managers data
            # For now, use basic ownership as effective ownership
            eo_data['effective_ownership'] = eo_data['selected_by_percent']
            return eo_data
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted EO: {str(e)}")
            return eo_data
    
    def _classify_ownership(self, ownership: float) -> str:
        """Classify player ownership level"""
        try:
            if ownership >= self.ownership_thresholds['template']:
                return 'template'
            elif ownership >= self.ownership_thresholds['high_owned']:
                return 'high_owned'
            elif ownership >= self.ownership_thresholds['medium_owned']:
                return 'medium_owned'
            else:
                return 'differential'
                
        except Exception as e:
            self.logger.error(f"Error classifying ownership: {str(e)}")
            return 'unknown'
    
    def identify_template_team(self, players_df: pd.DataFrame) -> Dict:
        """Identify the current template team"""
        try:
            if players_df.empty:
                return {}
            
            # Get template players (high ownership)
            template_players = players_df[
                players_df['selected_by_percent'] >= self.ownership_thresholds['template']
            ].copy()
            
            # Group by position
            template_by_position = {}
            for position in ['GKP', 'DEF', 'MID', 'FWD']:
                pos_players = template_players[template_players['position'] == position]
                template_by_position[position] = pos_players.nlargest(5, 'selected_by_percent').to_dict('records')
            
            # Calculate template coverage
            total_template_players = len(template_players)
            template_coverage = (total_template_players / len(players_df)) * 100
            
            return {
                'template_players': template_by_position,
                'total_template_players': total_template_players,
                'template_coverage': template_coverage,
                'avg_template_ownership': template_players['selected_by_percent'].mean() if not template_players.empty else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying template team: {str(e)}")
            return {}
    
    def find_differential_opportunities(self, 
                                      players_df: pd.DataFrame,
                                      min_form: float = 4.0,
                                      max_ownership: float = 10.0) -> List[Dict]:
        """Find differential opportunities"""
        try:
            if players_df.empty:
                return []
            
            # Filter for differential players with good form
            differentials = players_df[
                (players_df['selected_by_percent'] <= max_ownership) &
                (players_df['form'] >= min_form) &
                (players_df['total_points'] >= 20)  # Minimum season points
            ].copy()
            
            # Sort by form and ownership
            differentials = differentials.sort_values(['form', 'selected_by_percent'], ascending=[False, True])
            
            # Add differential analysis
            differentials['differential_score'] = (
                differentials['form'] * 0.6 + 
                (100 - differentials['selected_by_percent']) * 0.4
            )
            
            return differentials.head(10).to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error finding differential opportunities: {str(e)}")
            return []
    
    def calculate_team_risk_profile(self, team_players: List[int], players_df: pd.DataFrame) -> Dict:
        """Calculate risk profile for a team"""
        try:
            if not team_players or players_df.empty:
                return {}
            
            # Get team player data
            team_data = players_df[players_df['id'].isin(team_players)]
            
            if team_data.empty:
                return {}
            
            # Calculate ownership statistics
            ownership_stats = {
                'template_players': len(team_data[team_data['selected_by_percent'] >= self.ownership_thresholds['template']]),
                'differential_players': len(team_data[team_data['selected_by_percent'] <= self.ownership_thresholds['differential']]),
                'avg_ownership': team_data['selected_by_percent'].mean(),
                'ownership_std': team_data['selected_by_percent'].std(),
                'max_ownership': team_data['selected_by_percent'].max(),
                'min_ownership': team_data['selected_by_percent'].min()
            }
            
            # Calculate risk score (0-1, higher = riskier)
            risk_score = (
                (ownership_stats['differential_players'] / len(team_data)) * 0.6 +
                (ownership_stats['ownership_std'] / 100) * 0.4
            )
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = 'High Risk'
            elif risk_score >= 0.4:
                risk_level = 'Medium Risk'
            else:
                risk_level = 'Low Risk'
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'ownership_stats': ownership_stats,
                'team_size': len(team_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating team risk profile: {str(e)}")
            return {}
    
    def get_ownership_insights(self, players_df: pd.DataFrame) -> Dict:
        """Get overall ownership insights"""
        try:
            if players_df.empty:
                return {}
            
            # Calculate ownership distribution
            ownership_distribution = {
                'template': len(players_df[players_df['selected_by_percent'] >= self.ownership_thresholds['template']]),
                'high_owned': len(players_df[
                    (players_df['selected_by_percent'] >= self.ownership_thresholds['high_owned']) &
                    (players_df['selected_by_percent'] < self.ownership_thresholds['template'])
                ]),
                'medium_owned': len(players_df[
                    (players_df['selected_by_percent'] >= self.ownership_thresholds['medium_owned']) &
                    (players_df['selected_by_percent'] < self.ownership_thresholds['high_owned'])
                ]),
                'differential': len(players_df[players_df['selected_by_percent'] < self.ownership_thresholds['differential']])
            }
            
            # Top owned players
            top_owned = players_df.nlargest(10, 'selected_by_percent')[
                ['web_name', 'position', 'selected_by_percent', 'form']
            ].to_dict('records')
            
            # Most differential players with good form
            differentials = self.find_differential_opportunities(players_df)
            
            return {
                'ownership_distribution': ownership_distribution,
                'top_owned_players': top_owned,
                'differential_opportunities': differentials[:5],
                'total_players': len(players_df),
                'avg_ownership': players_df['selected_by_percent'].mean()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting ownership insights: {str(e)}")
            return {}
    
    def recommend_ownership_strategy(self, 
                                   current_team: List[int],
                                   players_df: pd.DataFrame,
                                   risk_tolerance: float = 0.5) -> Dict:
        """Recommend ownership strategy based on risk tolerance"""
        try:
            if not current_team or players_df.empty:
                return {}
            
            # Calculate current team risk profile
            current_risk = self.calculate_team_risk_profile(current_team, players_df)
            
            # Get template team
            template_team = self.identify_template_team(players_df)
            
            # Get differential opportunities
            differentials = self.find_differential_opportunities(players_df)
            
            # Recommend strategy based on risk tolerance
            if risk_tolerance >= 0.7:  # High risk tolerance
                strategy = "Aggressive Differential"
                recommended_ratio = "70% differentials, 30% template"
            elif risk_tolerance >= 0.4:  # Medium risk tolerance
                strategy = "Balanced"
                recommended_ratio = "50% differentials, 50% template"
            else:  # Low risk tolerance
                strategy = "Conservative Template"
                recommended_ratio = "70% template, 30% differentials"
            
            return {
                'strategy': strategy,
                'recommended_ratio': recommended_ratio,
                'current_risk_level': current_risk.get('risk_level', 'Unknown'),
                'current_risk_score': current_risk.get('risk_score', 0),
                'template_players': template_team.get('template_players', {}),
                'differential_opportunities': differentials[:5],
                'risk_tolerance': risk_tolerance
            }
            
        except Exception as e:
            self.logger.error(f"Error recommending ownership strategy: {str(e)}")
            return {}
