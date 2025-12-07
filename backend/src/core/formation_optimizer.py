"""
Formation flexibility optimizer
Suggests optimal formation based on squad composition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class FormationOptimizer:
    """Optimize formation selection for squad"""
    
    VALID_FORMATIONS = [
        (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1),
        (5, 3, 2), (5, 4, 1)
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def suggest_formation(
        self,
        squad: pd.DataFrame,
        include_autosub_analysis: bool = True
    ) -> Dict:
        """
        Suggest optimal formation for squad
        
        Args:
            squad: DataFrame of 15 players with predictions
            include_autosub_analysis: Whether to simulate autosubs
        
        Returns:
            Dict with formation recommendation
        """
        self.logger.info("🎯 Analyzing optimal formation...")
        
        # Separate by position
        gkp = squad[squad['position'] == 'GKP'].sort_values('predicted_points', ascending=False)
        defs = squad[squad['position'] == 'DEF'].sort_values('predicted_points', ascending=False)
        mids = squad[squad['position'] == 'MID'].sort_values('predicted_points', ascending=False)
        fwds = squad[squad['position'] == 'FWD'].sort_values('predicted_points', ascending=False)
        
        best_formation = None
        best_score = -1
        best_xi = None
        formation_analysis = []
        
        for formation in self.VALID_FORMATIONS:
            n_def, n_mid, n_fwd = formation
            
            # Check if formation is possible
            if len(defs) < n_def or len(mids) < n_mid or len(fwds) < n_fwd:
                continue
            
            # Select best XI for this formation
            xi = pd.concat([
                gkp.head(1),
                defs.head(n_def),
                mids.head(n_mid),
                fwds.head(n_fwd)
            ])
            
            # Calculate expected points
            base_points = xi['predicted_points'].sum()
            
            # Calculate bench strength
            bench = squad[~squad['id'].isin(xi['id'])]
            bench_strength = self._calculate_bench_strength(bench)
            
            # Simulate autosubs if requested
            if include_autosub_analysis:
                autosub_value = self._simulate_autosubs(xi, bench)
            else:
                autosub_value = 0
            
            # Total score
            total_score = base_points + (bench_strength * 0.1) + autosub_value
            
            formation_analysis.append({
                'formation': f"{n_def}-{n_mid}-{n_fwd}",
                'base_points': float(base_points),
                'bench_strength': float(bench_strength),
                'autosub_value': float(autosub_value),
                'total_score': float(total_score),
                'xi_ids': xi['id'].tolist()
            })
            
            if total_score > best_score:
                best_score = total_score
                best_formation = formation
                best_xi = xi
        
        if best_formation is None:
            return {'error': 'No valid formation found for this squad'}
        
        # Sort formations by score
        formation_analysis.sort(key=lambda x: x['total_score'], reverse=True)
        
        return {
            'recommended_formation': f"{best_formation[0]}-{best_formation[1]}-{best_formation[2]}",
            'expected_points': float(best_score),
            'starting_xi': best_xi[['web_name', 'position', 'predicted_points']].to_dict('records'),
            'all_formations': formation_analysis[:3],  # Top 3
            'reasoning': self._generate_reasoning(best_formation, formation_analysis[0])
        }
    
    def _calculate_bench_strength(self, bench: pd.DataFrame) -> float:
        """Calculate bench strength score"""
        if bench.empty:
            return 0.0
        
        # Weight by position (defenders more likely to come on)
        weights = {
            'GKP': 0.1,  # Rarely plays
            'DEF': 0.4,  # Often first sub
            'MID': 0.3,
            'FWD': 0.2
        }
        
        bench_score = 0.0
        for _, player in bench.iterrows():
            position = player['position']
            points = player.get('predicted_points', 0)
            bench_score += points * weights.get(position, 0.25)
        
        return bench_score
    
    def _simulate_autosubs(self, xi: pd.DataFrame, bench: pd.DataFrame) -> float:
        """
        Simulate autosub scenarios
        
        Estimates expected points from automatic substitutions
        """
        if bench.empty:
            return 0.0
        
        # Probability each player doesn't play
        non_play_prob = 0.15  # 15% chance on average
        
        # Expected autosub value
        autosub_value = 0.0
        
        # For each starting player, calculate expected replacement
        for _, starter in xi.iterrows():
            if starter['position'] == 'GKP':
                continue  # GKP rarely subbed
            
            # Find eligible bench replacements
            eligible_bench = bench[
                (bench['position'] == starter['position']) |
                (bench['position'] == 'DEF')  # DEF can replace anyone
            ]
            
            if not eligible_bench.empty:
                # Best bench option
                best_bench = eligible_bench.nlargest(1, 'predicted_points').iloc[0]
                
                # Expected value = prob(starter doesn't play) * bench points
                expected_value = non_play_prob * best_bench['predicted_points']
                autosub_value += expected_value
        
        return autosub_value
    
    def _generate_reasoning(self, formation: Tuple, analysis: Dict) -> str:
        """Generate human-readable reasoning"""
        n_def, n_mid, n_fwd = formation
        
        reasons = []
        
        # Formation type
        if n_mid >= 5:
            reasons.append("Midfield-heavy for consistent points")
        elif n_def >= 5:
            reasons.append("Defensive formation for clean sheet potential")
        elif n_fwd >= 3:
            reasons.append("Attacking formation for high ceiling")
        else:
            reasons.append("Balanced formation")
        
        # Bench strength
        if analysis['bench_strength'] > 10:
            reasons.append("Strong bench for autosub coverage")
        
        # Expected points
        reasons.append(f"{analysis['base_points']:.1f} expected points")
        
        return " | ".join(reasons)
    
    def analyze_formation_flexibility(self, squad: pd.DataFrame) -> Dict:
        """
        Analyze how many formations the squad can play
        
        Args:
            squad: 15-player squad
        
        Returns:
            Flexibility analysis
        """
        position_counts = squad['position'].value_counts().to_dict()
        
        valid_formations = []
        for formation in self.VALID_FORMATIONS:
            n_def, n_mid, n_fwd = formation
            
            if (position_counts.get('DEF', 0) >= n_def and
                position_counts.get('MID', 0) >= n_mid and
                position_counts.get('FWD', 0) >= n_fwd):
                valid_formations.append(f"{n_def}-{n_mid}-{n_fwd}")
        
        flexibility_score = len(valid_formations) / len(self.VALID_FORMATIONS)
        
        return {
            'valid_formations': valid_formations,
            'formation_count': len(valid_formations),
            'flexibility_score': flexibility_score,
            'rating': self._get_flexibility_rating(flexibility_score),
            'recommendation': self._get_flexibility_recommendation(flexibility_score)
        }
    
    def _get_flexibility_rating(self, score: float) -> str:
        """Get flexibility rating"""
        if score >= 0.7:
            return '⭐⭐⭐ Excellent'
        elif score >= 0.5:
            return '⭐⭐ Good'
        elif score >= 0.3:
            return '⭐ Fair'
        else:
            return '❌ Limited'
    
    def _get_flexibility_recommendation(self, score: float) -> str:
        """Get flexibility recommendation"""
        if score >= 0.7:
            return "Squad has excellent formation flexibility"
        elif score >= 0.5:
            return "Good flexibility - can adapt to most situations"
        elif score >= 0.3:
            return "Consider balancing positions for more options"
        else:
            return "⚠️  Limited flexibility - squad structure needs improvement"
