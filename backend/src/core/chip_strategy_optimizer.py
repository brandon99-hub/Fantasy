"""
Chip strategy optimizer
Plans optimal timing for Wildcard, Free Hit, Bench Boost, and Triple Captain
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ChipStrategyOptimizer:
    """Optimize chip usage across the season"""
    
    CHIPS = ['wildcard', 'free_hit', 'bench_boost', 'triple_captain']
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_season_chip_plan(
        self,
        fixtures_df: pd.DataFrame,
        players_df: pd.DataFrame,
        current_gameweek: int,
        used_chips: List[str] = None,
        team_ids: List[int] = None
    ) -> Dict:
        """
        Create optimal chip usage plan for rest of season
        
        Args:
            fixtures_df: All fixtures
            players_df: All players with predictions
            current_gameweek: Current gameweek number
            used_chips: List of already used chips
            team_ids: Current team player IDs
        
        Returns:
            Dict with chip recommendations for each gameweek
        """
        self.logger.info(f"Creating chip strategy from GW{current_gameweek}...")
        
        used_chips = used_chips or []
        remaining_chips = [chip for chip in self.CHIPS if chip not in used_chips]
        
        # Detect special gameweeks
        dgw_weeks = self._detect_double_gameweeks(fixtures_df)
        bgw_weeks = self._detect_blank_gameweeks(fixtures_df)
        
        chip_plan = {
            'current_gameweek': current_gameweek,
            'remaining_chips': remaining_chips,
            'double_gameweeks': dgw_weeks,
            'blank_gameweeks': bgw_weeks,
            'recommendations': []
        }
        
        # Analyze each remaining chip
        if 'bench_boost' in remaining_chips:
            bb_rec = self._recommend_bench_boost(dgw_weeks, current_gameweek, team_ids, players_df)
            chip_plan['recommendations'].append(bb_rec)
        
        if 'free_hit' in remaining_chips:
            fh_rec = self._recommend_free_hit(bgw_weeks, dgw_weeks, current_gameweek)
            chip_plan['recommendations'].append(fh_rec)
        
        if 'triple_captain' in remaining_chips:
            tc_rec = self._recommend_triple_captain(dgw_weeks, players_df, current_gameweek)
            chip_plan['recommendations'].append(tc_rec)
        
        if 'wildcard' in remaining_chips:
            wc_rec = self._recommend_wildcard(fixtures_df, current_gameweek, dgw_weeks)
            chip_plan['recommendations'].append(wc_rec)
        
        # Sort by priority
        chip_plan['recommendations'].sort(key=lambda x: x['priority'], reverse=True)
        
        # Create timeline
        chip_plan['timeline'] = self._create_chip_timeline(chip_plan['recommendations'])
        
        return chip_plan
    
    def _detect_double_gameweeks(self, fixtures_df: pd.DataFrame) -> List[int]:
        """Detect gameweeks where teams have 2+ fixtures"""
        if fixtures_df.empty:
            return []
        
        # Count fixtures per team per gameweek
        fixture_counts = fixtures_df.groupby(['event', 'team_h']).size().reset_index(name='count')
        
        # Find gameweeks with teams having 2+ fixtures
        dgw_data = fixture_counts[fixture_counts['count'] >= 2]
        dgw_weeks = sorted(dgw_data['event'].unique().tolist())
        
        return dgw_weeks
    
    def _detect_blank_gameweeks(self, fixtures_df: pd.DataFrame) -> List[int]:
        """Detect gameweeks with fewer fixtures than normal"""
        if fixtures_df.empty:
            return []
        
        # Count total fixtures per gameweek
        fixtures_per_gw = fixtures_df.groupby('event').size()
        
        # Normal is 10 fixtures (20 teams)
        # Blank gameweek has significantly fewer
        bgw_weeks = fixtures_per_gw[fixtures_per_gw < 7].index.tolist()
        
        return sorted(bgw_weeks)
    
    def _recommend_bench_boost(
        self,
        dgw_weeks: List[int],
        current_gw: int,
        team_ids: Optional[List[int]],
        players_df: pd.DataFrame
    ) -> Dict:
        """Recommend when to use Bench Boost"""
        future_dgws = [gw for gw in dgw_weeks if gw > current_gw]
        
        if not future_dgws:
            return {
                'chip': 'bench_boost',
                'recommended_gameweek': None,
                'priority': 3,
                'reasoning': 'No double gameweeks detected - use before season end',
                'expected_benefit': 0
            }
        
        # Best DGW for BB
        best_dgw = future_dgws[0] if future_dgws else None
        
        # Calculate potential benefit
        if team_ids and not players_df.empty:
            team_players = players_df[players_df['id'].isin(team_ids)]
            bench_players = team_players.tail(4)  # Last 4 are bench
            expected_bench_points = bench_players['predicted_points'].sum() * 2  # DGW multiplier
        else:
            expected_bench_points = 20  # Estimate
        
        return {
            'chip': 'bench_boost',
            'recommended_gameweek': best_dgw,
            'priority': 8 if best_dgw else 3,
            'reasoning': f'Use in DGW{best_dgw} to maximize bench points' if best_dgw else 'No DGW available',
            'expected_benefit': float(expected_bench_points)
        }
    
    def _recommend_free_hit(
        self,
        bgw_weeks: List[int],
        dgw_weeks: List[int],
        current_gw: int
    ) -> Dict:
        """Recommend when to use Free Hit"""
        future_bgws = [gw for gw in bgw_weeks if gw > current_gw]
        future_dgws = [gw for gw in dgw_weeks if gw > current_gw]
        
        if future_bgws:
            # Use in blank gameweek
            best_gw = future_bgws[0]
            reasoning = f'Use in BGW{best_gw} to field full team when many teams blank'
            priority = 9
            benefit = 40  # Significant benefit in BGW
        elif future_dgws:
            # Use in DGW if no BGW
            best_gw = future_dgws[0]
            reasoning = f'Use in DGW{best_gw} to maximize double gameweek players'
            priority = 7
            benefit = 30
        else:
            best_gw = None
            reasoning = 'Save for unexpected fixture changes or end of season'
            priority = 4
            benefit = 0
        
        return {
            'chip': 'free_hit',
            'recommended_gameweek': best_gw,
            'priority': priority,
            'reasoning': reasoning,
            'expected_benefit': benefit
        }
    
    def _recommend_triple_captain(
        self,
        dgw_weeks: List[int],
        players_df: pd.DataFrame,
        current_gw: int
    ) -> Dict:
        """Recommend when to use Triple Captain"""
        future_dgws = [gw for gw in dgw_weeks if gw > current_gw]
        
        if not future_dgws:
            # No DGW - use on best single gameweek
            if not players_df.empty:
                best_captain = players_df.nlargest(1, 'predicted_points').iloc[0]
                benefit = best_captain['predicted_points'] * 2  # Triple instead of double
                reasoning = f"Use on {best_captain['web_name']} in favorable fixture"
            else:
                benefit = 10
                reasoning = "Use on premium captain in good fixture"
            
            return {
                'chip': 'triple_captain',
                'recommended_gameweek': None,
                'priority': 5,
                'reasoning': reasoning,
                'expected_benefit': float(benefit)
            }
        
        # Use in DGW
        best_dgw = future_dgws[0]
        
        # Find best captain for DGW
        if not players_df.empty:
            # Premium players likely to play both games
            premium = players_df[players_df['now_cost'] >= 100]
            if not premium.empty:
                best_captain = premium.nlargest(1, 'predicted_points').iloc[0]
                benefit = best_captain['predicted_points'] * 4  # DGW + Triple
                captain_name = best_captain['web_name']
            else:
                benefit = 30
                captain_name = "premium player"
        else:
            benefit = 30
            captain_name = "premium player"
        
        return {
            'chip': 'triple_captain',
            'recommended_gameweek': best_dgw,
            'priority': 10,  # Highest priority - most impactful
            'reasoning': f'Use on {captain_name} in DGW{best_dgw} for maximum points',
            'expected_benefit': float(benefit)
        }
    
    def _recommend_wildcard(
        self,
        fixtures_df: pd.DataFrame,
        current_gw: int,
        dgw_weeks: List[int]
    ) -> Dict:
        """Recommend when to use Wildcard"""
        # Wildcard timing depends on:
        # 1. Upcoming fixture swings
        # 2. Before DGWs to optimize team
        # 3. Team value and structure issues
        
        future_dgws = [gw for gw in dgw_weeks if gw > current_gw]
        
        if future_dgws:
            # Use 1-2 GWs before DGW to prepare
            target_gw = max(current_gw + 1, future_dgws[0] - 1)
            reasoning = f'Use in GW{target_gw} to prepare team for DGW{future_dgws[0]}'
            priority = 7
        else:
            # Use around GW20-25 for fixture swing
            if current_gw < 20:
                target_gw = 20
                reasoning = 'Use around GW20 for mid-season fixture swing'
                priority = 6
            else:
                target_gw = current_gw + 5
                reasoning = 'Use when team structure needs major overhaul'
                priority = 5
        
        return {
            'chip': 'wildcard',
            'recommended_gameweek': target_gw,
            'priority': priority,
            'reasoning': reasoning,
            'expected_benefit': 50  # Significant long-term benefit
        }
    
    def _create_chip_timeline(self, recommendations: List[Dict]) -> List[Dict]:
        """Create timeline of chip usage"""
        timeline = []
        
        for rec in recommendations:
            if rec['recommended_gameweek']:
                timeline.append({
                    'gameweek': rec['recommended_gameweek'],
                    'chip': rec['chip'],
                    'reasoning': rec['reasoning'],
                    'priority': rec['priority']
                })
        
        # Sort by gameweek
        timeline.sort(key=lambda x: x['gameweek'])
        
        return timeline
    
    def evaluate_chip_combinations(
        self,
        fixtures_df: pd.DataFrame,
        current_gw: int
    ) -> Dict:
        """Evaluate synergies between chips"""
        dgw_weeks = self._detect_double_gameweeks(fixtures_df)
        bgw_weeks = self._detect_blank_gameweeks(fixtures_df)
        
        combinations = []
        
        # Wildcard + Bench Boost in DGW
        if dgw_weeks:
            combinations.append({
                'chips': ['wildcard', 'bench_boost'],
                'gameweeks': [dgw_weeks[0] - 1, dgw_weeks[0]],
                'synergy': 'High',
                'reasoning': 'Wildcard before DGW to build strong bench, then Bench Boost in DGW',
                'expected_benefit': 70
            })
        
        # Free Hit in BGW, Triple Captain in DGW
        if bgw_weeks and dgw_weeks:
            combinations.append({
                'chips': ['free_hit', 'triple_captain'],
                'gameweeks': [bgw_weeks[0], dgw_weeks[0]],
                'synergy': 'Medium',
                'reasoning': 'Free Hit to navigate BGW, Triple Captain in DGW',
                'expected_benefit': 60
            })
        
        return {
            'combinations': combinations,
            'recommended_strategy': combinations[0] if combinations else None
        }
