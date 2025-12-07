"""
Strategic planning for FPL optimization
Handles chip usage, long-term team building, and manager preferences
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

class ChipType(Enum):
    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"

@dataclass
class ChipUsage:
    chip_type: ChipType
    gameweek: int
    reason: str
    expected_benefit: float

@dataclass
class ManagerPreferences:
    risk_tolerance: float = 0.5  # 0-1 scale
    formation_preference: str = "3-4-3"
    budget_allocation: Dict[str, float] = None  # Position -> budget allocation
    differential_threshold: float = 5.0  # Minimum ownership % for differentials
    captain_strategy: str = "fixture_based"  # "safe", "differential", "fixture_based"
    transfer_frequency: str = "moderate"  # "conservative", "moderate", "aggressive"
    
    def __post_init__(self):
        if self.budget_allocation is None:
            self.budget_allocation = {
                'GKP': 0.08,  # 8% of budget
                'DEF': 0.25,  # 25% of budget
                'MID': 0.40,  # 40% of budget
                'FWD': 0.27   # 27% of budget
            }

class StrategicPlanner:
    """Advanced strategic planning for FPL optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Chip usage limits
        self.chip_limits = {
            ChipType.WILDCARD: 2,
            ChipType.FREE_HIT: 1,
            ChipType.BENCH_BOOST: 1,
            ChipType.TRIPLE_CAPTAIN: 1
        }
        
        # Default manager preferences
        self.default_preferences = ManagerPreferences(
            risk_tolerance=0.5,
            formation_preference="3-4-3",
            budget_allocation={
                'GKP': 0.08,  # 8% of budget
                'DEF': 0.25,  # 25% of budget
                'MID': 0.40,  # 40% of budget
                'FWD': 0.27   # 27% of budget
            },
            differential_threshold=5.0,
            captain_strategy="fixture_based",
            transfer_frequency="moderate"
        )
    
    def analyze_chip_usage_opportunities(self, 
                                       fixtures_df: pd.DataFrame,
                                       players_df: pd.DataFrame,
                                       current_gameweek: int,
                                       used_chips: List[ChipType] = None) -> List[ChipUsage]:
        """Analyze optimal chip usage opportunities"""
        try:
            if used_chips is None:
                used_chips = []
            
            opportunities = []
            
            # Analyze double gameweeks for Bench Boost
            double_gws = self._detect_double_gameweeks(fixtures_df)
            for gw, teams in double_gws.items():
                if gw > current_gameweek and ChipType.BENCH_BOOST not in used_chips:
                    benefit = self._calculate_bench_boost_benefit(teams, players_df, gw)
                    if benefit > 5:  # Minimum 5 point benefit
                        opportunities.append(ChipUsage(
                            chip_type=ChipType.BENCH_BOOST,
                            gameweek=gw,
                            reason=f"Double gameweek for {len(teams)} teams",
                            expected_benefit=benefit
                        ))
            
            # Analyze blank gameweeks for Free Hit
            blank_gws = self._detect_blank_gameweeks(fixtures_df)
            for gw in blank_gws:
                if gw > current_gameweek and ChipType.FREE_HIT not in used_chips:
                    benefit = self._calculate_free_hit_benefit(players_df, gw)
                    if benefit > 8:  # Minimum 8 point benefit
                        opportunities.append(ChipUsage(
                            chip_type=ChipType.FREE_HIT,
                            gameweek=gw,
                            reason=f"Blank gameweek - limited options",
                            expected_benefit=benefit
                        ))
            
            # Analyze captain opportunities for Triple Captain
            captain_opportunities = self._find_triple_captain_opportunities(
                players_df, fixtures_df, current_gameweek
            )
            for opp in captain_opportunities:
                if ChipType.TRIPLE_CAPTAIN not in used_chips:
                    opportunities.append(ChipUsage(
                        chip_type=ChipType.TRIPLE_CAPTAIN,
                        gameweek=opp['gameweek'],
                        reason=f"High-scoring opportunity for {opp['player_name']}",
                        expected_benefit=opp['expected_benefit']
                    ))
            
            # Wildcard opportunities (more complex analysis)
            wildcard_opportunities = self._find_wildcard_opportunities(
                players_df, fixtures_df, current_gameweek
            )
            for opp in wildcard_opportunities:
                if ChipType.WILDCARD not in used_chips:
                    opportunities.append(ChipUsage(
                        chip_type=ChipType.WILDCARD,
                        gameweek=opp['gameweek'],
                        reason=opp['reason'],
                        expected_benefit=opp['expected_benefit']
                    ))
            
            # Sort by expected benefit
            opportunities.sort(key=lambda x: x.expected_benefit, reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing chip opportunities: {str(e)}")
            return []
    
    def _detect_double_gameweeks(self, fixtures_df: pd.DataFrame) -> Dict[int, List[int]]:
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
    
    def _detect_blank_gameweeks(self, fixtures_df: pd.DataFrame) -> List[int]:
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
    
    def _calculate_bench_boost_benefit(self, teams: List[int], players_df: pd.DataFrame, gameweek: int) -> float:
        """Calculate expected benefit of using Bench Boost"""
        try:
            # Find players from double gameweek teams
            double_team_players = players_df[players_df['team'].isin(teams)]
            
            if double_team_players.empty:
                return 0.0
            
            # Calculate expected points for bench players
            bench_players = double_team_players.nsmallest(4, 'now_cost')  # Cheapest 4 as bench
            expected_bench_points = bench_players['form'].sum() * 2  # Double gameweek multiplier
            
            return expected_bench_points
            
        except Exception as e:
            self.logger.error(f"Error calculating bench boost benefit: {str(e)}")
            return 0.0
    
    def _calculate_free_hit_benefit(self, players_df: pd.DataFrame, gameweek: int) -> float:
        """Calculate expected benefit of using Free Hit"""
        try:
            # In blank gameweeks, benefit comes from having more options
            # This is a simplified calculation
            available_players = len(players_df[players_df['status'] == 'a'])
            normal_team_size = 15
            
            # Benefit scales with how many more options you have
            additional_options = max(0, available_players - normal_team_size)
            benefit = additional_options * 0.5  # 0.5 points per additional option
            
            return benefit
            
        except Exception as e:
            self.logger.error(f"Error calculating free hit benefit: {str(e)}")
            return 0.0
    
    def _find_triple_captain_opportunities(self, 
                                         players_df: pd.DataFrame,
                                         fixtures_df: pd.DataFrame,
                                         current_gameweek: int) -> List[Dict]:
        """Find optimal Triple Captain opportunities"""
        try:
            opportunities = []
            
            # Look for players with high form and easy fixtures
            top_players = players_df.nlargest(20, 'form')
            
            for _, player in top_players.iterrows():
                # Check upcoming fixtures for this player's team
                team_fixtures = fixtures_df[
                    ((fixtures_df['team_h'] == player['team']) | 
                     (fixtures_df['team_a'] == player['team'])) &
                    (fixtures_df['event'] > current_gameweek) &
                    (fixtures_df['event'] <= current_gameweek + 5)
                ]
                
                if team_fixtures.empty:
                    continue
                
                # Calculate average fixture difficulty
                avg_difficulty = team_fixtures['team_h_difficulty'].mean() if player['team'] in team_fixtures['team_h'].values else team_fixtures['team_a_difficulty'].mean()
                
                # High form + easy fixtures = good triple captain opportunity
                if player['form'] > 6 and avg_difficulty <= 3:
                    expected_benefit = player['form'] * 1.5  # Triple captain bonus
                    
                    opportunities.append({
                        'player_id': player['id'],
                        'player_name': player['web_name'],
                        'gameweek': team_fixtures['event'].iloc[0],
                        'expected_benefit': expected_benefit,
                        'form': player['form'],
                        'fixture_difficulty': avg_difficulty
                    })
            
            return sorted(opportunities, key=lambda x: x['expected_benefit'], reverse=True)[:3]
            
        except Exception as e:
            self.logger.error(f"Error finding triple captain opportunities: {str(e)}")
            return []
    
    def _find_wildcard_opportunities(self, 
                                   players_df: pd.DataFrame,
                                   fixtures_df: pd.DataFrame,
                                   current_gameweek: int) -> List[Dict]:
        """Find optimal Wildcard opportunities"""
        try:
            opportunities = []
            
            # Look for major fixture swings or price changes
            # This is a simplified analysis
            
            # Check for teams with improving fixture runs
            team_fixture_analysis = self._analyze_team_fixture_runs(fixtures_df, current_gameweek)
            
            for team_id, analysis in team_fixture_analysis.items():
                if analysis['improvement_score'] > 2:  # Significant improvement
                    opportunities.append({
                        'gameweek': current_gameweek + 1,
                        'reason': f"Team {team_id} has improving fixtures",
                        'expected_benefit': analysis['improvement_score'] * 2
                    })
            
            return opportunities[:2]  # Top 2 opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding wildcard opportunities: {str(e)}")
            return []
    
    def _analyze_team_fixture_runs(self, fixtures_df: pd.DataFrame, current_gameweek: int) -> Dict[int, Dict]:
        """Analyze fixture difficulty runs for teams"""
        try:
            team_analysis = {}
            
            for team_id in fixtures_df['team_h'].unique():
                # Get team's upcoming fixtures
                team_fixtures = fixtures_df[
                    ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
                    (fixtures_df['event'] > current_gameweek) &
                    (fixtures_df['event'] <= current_gameweek + 6)
                ]
                
                if team_fixtures.empty:
                    continue
                
                # Calculate difficulty scores
                difficulties = []
                for _, fixture in team_fixtures.iterrows():
                    if fixture['team_h'] == team_id:
                        difficulties.append(fixture['team_h_difficulty'])
                    else:
                        difficulties.append(fixture['team_a_difficulty'])
                
                # Calculate improvement trend
                if len(difficulties) >= 3:
                    early_avg = np.mean(difficulties[:3])
                    later_avg = np.mean(difficulties[-3:])
                    improvement_score = early_avg - later_avg  # Lower is better
                    
                    team_analysis[team_id] = {
                        'improvement_score': improvement_score,
                        'avg_difficulty': np.mean(difficulties),
                        'fixture_count': len(difficulties)
                    }
            
            return team_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing team fixture runs: {str(e)}")
            return {}
    
    def create_long_term_strategy(self, 
                                players_df: pd.DataFrame,
                                fixtures_df: pd.DataFrame,
                                current_gameweek: int,
                                preferences: ManagerPreferences = None) -> Dict:
        """Create a long-term team building strategy"""
        try:
            if preferences is None:
                preferences = self.default_preferences
            
            strategy = {
                'current_gameweek': current_gameweek,
                'preferences': preferences,
                'budget_allocation': preferences.budget_allocation,
                'key_players': [],
                'differential_targets': [],
                'fixture_swings': [],
                'price_watch_list': [],
                'transfer_plan': []
            }
            
            # Identify key players (high ownership, consistent performers)
            key_players = players_df[
                (players_df['selected_by_percent'] > 20) &
                (players_df['form'] > 5) &
                (players_df['total_points'] > 50)
            ].nlargest(8, 'total_points')
            
            strategy['key_players'] = key_players[['id', 'web_name', 'position', 'now_cost', 'form']].to_dict('records')
            
            # Identify differential targets
            differentials = players_df[
                (players_df['selected_by_percent'] < preferences.differential_threshold) &
                (players_df['form'] > 4) &
                (players_df['now_cost'] > 60)  # Not too cheap
            ].nlargest(5, 'form')
            
            strategy['differential_targets'] = differentials[['id', 'web_name', 'position', 'now_cost', 'form', 'selected_by_percent']].to_dict('records')
            
            # Analyze fixture swings
            fixture_swings = self._analyze_fixture_swings(fixtures_df, current_gameweek)
            strategy['fixture_swings'] = fixture_swings
            
            # Price watch list (players likely to rise/fall)
            price_watch = self._create_price_watch_list(players_df)
            strategy['price_watch_list'] = price_watch
            
            # Create transfer plan
            transfer_plan = self._create_transfer_plan(players_df, fixtures_df, current_gameweek, preferences)
            strategy['transfer_plan'] = transfer_plan
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error creating long-term strategy: {str(e)}")
            return {}
    
    def _analyze_fixture_swings(self, fixtures_df: pd.DataFrame, current_gameweek: int) -> List[Dict]:
        """Analyze upcoming fixture difficulty swings"""
        try:
            swings = []
            
            # Get next 6 gameweeks
            upcoming_gws = range(current_gameweek + 1, current_gameweek + 7)
            
            for gw in upcoming_gws:
                gw_fixtures = fixtures_df[fixtures_df['event'] == gw]
                
                if gw_fixtures.empty:
                    continue
                
                # Find teams with very easy or very hard fixtures
                easy_teams = gw_fixtures[
                    (gw_fixtures['team_h_difficulty'] <= 2) | 
                    (gw_fixtures['team_a_difficulty'] <= 2)
                ]['team_h'].tolist()
                
                hard_teams = gw_fixtures[
                    (gw_fixtures['team_h_difficulty'] >= 4) | 
                    (gw_fixtures['team_a_difficulty'] >= 4)
                ]['team_h'].tolist()
                
                if easy_teams or hard_teams:
                    swings.append({
                        'gameweek': gw,
                        'easy_fixtures': easy_teams,
                        'hard_fixtures': hard_teams
                    })
            
            return swings
            
        except Exception as e:
            self.logger.error(f"Error analyzing fixture swings: {str(e)}")
            return []
    
    def _create_price_watch_list(self, players_df: pd.DataFrame) -> List[Dict]:
        """Create price watch list based on transfer trends"""
        try:
            # Players with high transfer activity
            high_transfer_activity = players_df[
                (players_df['transfers_in_event'] > 10000) |
                (players_df['transfers_out_event'] > 10000)
            ].nlargest(10, 'transfers_in_event')
            
            watch_list = []
            for _, player in high_transfer_activity.iterrows():
                watch_list.append({
                    'player_id': player['id'],
                    'player_name': player['web_name'],
                    'current_price': player['now_cost'] / 10.0,
                    'transfers_in': player['transfers_in_event'],
                    'transfers_out': player['transfers_out_event'],
                    'net_transfers': player['transfers_in_event'] - player['transfers_out_event'],
                    'price_trend': 'rising' if player['transfers_in_event'] > player['transfers_out_event'] else 'falling'
                })
            
            return watch_list
            
        except Exception as e:
            self.logger.error(f"Error creating price watch list: {str(e)}")
            return []
    
    def _create_transfer_plan(self, 
                            players_df: pd.DataFrame,
                            fixtures_df: pd.DataFrame,
                            current_gameweek: int,
                            preferences: ManagerPreferences) -> List[Dict]:
        """Create a multi-gameweek transfer plan"""
        try:
            transfer_plan = []
            
            # Look ahead 4-6 gameweeks
            planning_horizon = 6
            
            for gw in range(current_gameweek + 1, current_gameweek + planning_horizon + 1):
                # Analyze fixture difficulty for this gameweek
                gw_fixtures = fixtures_df[fixtures_df['event'] == gw]
                
                if gw_fixtures.empty:
                    continue
                
                # Find teams with easy fixtures
                easy_teams = []
                for _, fixture in gw_fixtures.iterrows():
                    if fixture['team_h_difficulty'] <= 2:
                        easy_teams.append(fixture['team_h'])
                    if fixture['team_a_difficulty'] <= 2:
                        easy_teams.append(fixture['team_a'])
                
                # Find players from easy fixture teams
                if easy_teams:
                    easy_fixture_players = players_df[players_df['team'].isin(easy_teams)]
                    
                    # Get top performers from easy fixture teams
                    top_performers = easy_fixture_players.nlargest(3, 'form')
                    
                    if not top_performers.empty:
                        transfer_plan.append({
                            'gameweek': gw,
                            'priority': 'high' if len(easy_teams) > 3 else 'medium',
                            'target_players': top_performers[['id', 'web_name', 'position', 'form']].to_dict('records'),
                            'reason': f"Easy fixtures for {len(easy_teams)} teams"
                        })
            
            return transfer_plan
            
        except Exception as e:
            self.logger.error(f"Error creating transfer plan: {str(e)}")
            return []
    
    def optimize_captain_strategy(self, 
                                players_df: pd.DataFrame,
                                fixtures_df: pd.DataFrame,
                                current_gameweek: int,
                                preferences: ManagerPreferences = None) -> Dict:
        """Optimize captain selection strategy"""
        try:
            if preferences is None:
                preferences = self.default_preferences
            
            captain_strategy = {
                'current_gameweek': current_gameweek,
                'strategy_type': preferences.captain_strategy,
                'next_5_captains': [],
                'differential_captains': [],
                'safe_captains': []
            }
            
            # Get next 5 gameweeks
            next_5_gws = range(current_gameweek + 1, current_gameweek + 6)
            
            for gw in next_5_gws:
                gw_fixtures = fixtures_df[fixtures_df['event'] == gw]
                
                if gw_fixtures.empty:
                    continue
                
                # Find teams with home fixtures (generally better for captaincy)
                home_teams = gw_fixtures['team_h'].tolist()
                home_team_players = players_df[players_df['team'].isin(home_teams)]
                
                # Get top captain options for this gameweek
                captain_options = home_team_players.nlargest(5, 'form')
                
                if not captain_options.empty:
                    records = captain_options[['id', 'web_name', 'position', 'form', 'selected_by_percent']].to_dict('records')
                    # If effective ownership columns are present, include them for richer UI
                    if 'effective_ownership' in captain_options.columns:
                        for rec in records:
                            row = captain_options[captain_options['id'] == rec['id']].iloc[0]
                            rec['effective_ownership'] = float(row.get('effective_ownership', rec['selected_by_percent']))
                            top_captain_share = row.get('top_cohort_captain_pct')
                            if pd.notna(top_captain_share):
                                rec['top10k_captain_pct'] = float(top_captain_share)
                    captain_strategy['next_5_captains'].append({
                        'gameweek': gw,
                        'top_options': records
                    })
                
                # Identify differential captains
                differentials = home_team_players[
                    (home_team_players['selected_by_percent'] < preferences.differential_threshold) &
                    (home_team_players['form'] > 4)
                ].nlargest(3, 'form')
                
                if not differentials.empty:
                    captain_strategy['differential_captains'].extend(
                        differentials[['id', 'web_name', 'position', 'form', 'selected_by_percent']].to_dict('records')
                    )
                
                # Identify safe captains (high ownership, consistent)
                safe_captains = home_team_players[
                    (home_team_players['selected_by_percent'] > 15) &
                    (home_team_players['form'] > 5)
                ].nlargest(3, 'total_points')
                
                if not safe_captains.empty:
                    captain_strategy['safe_captains'].extend(
                        safe_captains[['id', 'web_name', 'position', 'form', 'selected_by_percent']].to_dict('records')
                    )
            
            return captain_strategy
            
        except Exception as e:
            self.logger.error(f"Error optimizing captain strategy: {str(e)}")
            return {}
