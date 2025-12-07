"""
Multi-gameweek transfer planning optimizer
Optimizes transfer sequence across multiple gameweeks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from ortools.linear_solver import pywraplp
import logging

from backend.src.core.optimizer import FPLOptimizer
from backend.src.utils.price_predictor import PricePredictor

logger = logging.getLogger(__name__)


class MultiGWOptimizer:
    """Optimize transfers across multiple gameweeks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimizer = FPLOptimizer()
        self.price_predictor = PricePredictor()
    
    def optimize_transfer_sequence(
        self,
        players_df: pd.DataFrame,
        current_team: List[int],
        horizon: int = 5,
        free_transfers: int = 1,
        bank: float = 0.0,
        budget: float = 100.0
    ) -> Dict:
        """
        Optimize transfer sequence over multiple gameweeks
        
        Args:
            players_df: All players with predictions
            current_team: Current 15 player IDs
            horizon: Number of gameweeks to plan ahead
            free_transfers: Current free transfers
            bank: Money in bank (millions)
            budget: Total budget (millions)
        
        Returns:
            Dict with transfer plan for each gameweek
        """
        self.logger.info(f"📅 Planning transfers for next {horizon} gameweeks...")
        
        transfer_plan = {
            'gameweeks': [],
            'total_expected_points': 0.0,
            'total_transfer_cost': 0,
            'final_bank': bank,
            'recommendations': []
        }
        
        current_squad = current_team.copy()
        current_ft = free_transfers
        current_bank = bank
        
        # Get price change predictions
        price_predictions = self.price_predictor.predict_price_changes(players_df)
        
        for gw in range(1, horizon + 1):
            self.logger.info(f"Planning GW +{gw}...")
            
            # Get predictions for this gameweek
            gw_predictions = self._get_gw_predictions(players_df, gw)
            
            # Optimize team for this gameweek
            result = self.optimizer.optimize_team(
                players_df=gw_predictions,
                budget=budget,
                current_team=current_squad,
                free_transfers=current_ft,
                use_wildcard=False
            )
            
            if not result or 'team' not in result:
                self.logger.warning(f"Optimization failed for GW +{gw}")
                continue
            
            # Extract transfers
            transfers_in = result.get('transfers_in', [])
            transfers_out = result.get('transfers_out', [])
            transfer_cost = result.get('transfer_cost', 0)
            
            # Calculate expected points
            expected_points = result.get('total_predicted_points', 0)
            
            # Check for price changes
            price_changes = self._check_price_changes(
                transfers_in, transfers_out, price_predictions, gw
            )
            
            # Determine if transfers should be made this week or banked
            should_transfer = self._should_make_transfer(
                expected_points_gain=result.get('points_gain', 0),
                transfer_cost=transfer_cost,
                free_transfers=current_ft,
                price_changes=price_changes
            )
            
            gw_plan = {
                'gameweek': gw,
                'transfers_in': transfers_in,
                'transfers_out': transfers_out,
                'transfer_cost': transfer_cost if should_transfer else 0,
                'expected_points': expected_points,
                'should_transfer': should_transfer,
                'free_transfers_before': current_ft,
                'free_transfers_after': self._calculate_ft_after(current_ft, len(transfers_in), should_transfer),
                'bank_before': current_bank,
                'bank_after': current_bank,  # Will be updated
                'price_change_risk': price_changes,
                'recommendation': self._generate_recommendation(
                    should_transfer, transfers_in, transfers_out, price_changes
                )
            }
            
            # Update state for next gameweek
            if should_transfer:
                current_squad = result['team']
                current_ft = gw_plan['free_transfers_after']
                # Update bank based on transfers
                transfer_value = self._calculate_transfer_value(
                    transfers_in, transfers_out, players_df
                )
                current_bank += transfer_value
            else:
                # Bank the free transfer (max 2)
                current_ft = min(current_ft + 1, 2)
            
            gw_plan['bank_after'] = current_bank
            transfer_plan['gameweeks'].append(gw_plan)
            transfer_plan['total_expected_points'] += expected_points
            transfer_plan['total_transfer_cost'] += gw_plan['transfer_cost']
        
        transfer_plan['final_bank'] = current_bank
        transfer_plan['summary'] = self._generate_summary(transfer_plan)
        
        return transfer_plan
    
    def _get_gw_predictions(self, players_df: pd.DataFrame, gw_offset: int) -> pd.DataFrame:
        """Get predictions adjusted for future gameweek"""
        # In practice, this would adjust predictions based on fixture difficulty
        # For now, return current predictions
        return players_df.copy()
    
    def _check_price_changes(
        self,
        transfers_in: List[Dict],
        transfers_out: List[Dict],
        price_predictions: pd.DataFrame,
        gw: int
    ) -> Dict:
        """Check if any transfer targets are likely to change price"""
        risks = {
            'rise_soon': [],
            'fall_soon': []
        }
        
        for player in transfers_in:
            player_id = player.get('id')
            pred = price_predictions[price_predictions['id'] == player_id]
            
            if not pred.empty:
                rise_prob = pred.iloc[0].get('rise_probability', 0)
                if rise_prob > 0.7:
                    risks['rise_soon'].append({
                        'player': player.get('web_name'),
                        'probability': rise_prob
                    })
        
        for player in transfers_out:
            player_id = player.get('id')
            pred = price_predictions[price_predictions['id'] == player_id]
            
            if not pred.empty:
                fall_prob = pred.iloc[0].get('fall_probability', 0)
                if fall_prob > 0.7:
                    risks['fall_soon'].append({
                        'player': player.get('web_name'),
                        'probability': fall_prob
                    })
        
        return risks
    
    def _should_make_transfer(
        self,
        expected_points_gain: float,
        transfer_cost: int,
        free_transfers: int,
        price_changes: Dict
    ) -> bool:
        """Decide if transfer should be made this week"""
        # Don't take hits unless significant gain
        if transfer_cost > 0:
            if expected_points_gain < transfer_cost + 2:
                return False
        
        # Make transfer if price rise imminent
        if price_changes['rise_soon']:
            return True
        
        # Make transfer if using free transfers
        if transfer_cost == 0:
            return expected_points_gain > 1.0
        
        return expected_points_gain > transfer_cost + 3
    
    def _calculate_ft_after(self, current_ft: int, num_transfers: int, made_transfer: bool) -> int:
        """Calculate free transfers after this gameweek"""
        if not made_transfer:
            return min(current_ft + 1, 2)
        
        if num_transfers <= current_ft:
            return 1
        else:
            return 1
    
    def _calculate_transfer_value(
        self,
        transfers_in: List[Dict],
        transfers_out: List[Dict],
        players_df: pd.DataFrame
    ) -> float:
        """Calculate net transfer value"""
        value_in = sum(p.get('now_cost', 0) for p in transfers_in) / 10
        value_out = sum(p.get('now_cost', 0) for p in transfers_out) / 10
        return value_out - value_in
    
    def _generate_recommendation(
        self,
        should_transfer: bool,
        transfers_in: List[Dict],
        transfers_out: List[Dict],
        price_changes: Dict
    ) -> str:
        """Generate human-readable recommendation"""
        if not should_transfer:
            return "💰 Bank free transfer for next week"
        
        if not transfers_in:
            return "✅ Hold current team"
        
        rec = []
        for tin, tout in zip(transfers_in, transfers_out):
            rec.append(f"{tout.get('web_name')} → {tin.get('web_name')}")
        
        if price_changes['rise_soon']:
            rec.append("⚠️  Price rises imminent!")
        
        return " | ".join(rec)
    
    def _generate_summary(self, plan: Dict) -> str:
        """Generate summary of transfer plan"""
        total_transfers = sum(len(gw['transfers_in']) for gw in plan['gameweeks'] if gw['should_transfer'])
        total_hits = plan['total_transfer_cost'] // 4
        
        summary = f"📊 {plan['horizon']}-week plan: "
        summary += f"{total_transfers} transfers, "
        summary += f"{total_hits} hits (-{plan['total_transfer_cost']} pts), "
        summary += f"Expected: +{plan['total_expected_points']:.1f} pts"
        
        return summary
