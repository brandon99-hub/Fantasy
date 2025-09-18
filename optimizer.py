import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import logging
from typing import Dict, List, Optional, Tuple
from utils.fpl_rules import FPLRules

class FPLOptimizer:
    """Mathematical optimization for FPL team selection using OR-Tools"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = FPLRules()
        self.solver = None
        self.status = None
    
    def optimize_team(self, 
                     players_df: pd.DataFrame,
                     budget: float = 100.0,
                     current_team: List[int] = None,
                     free_transfers: int = 1,
                     use_wildcard: bool = False,
                     captain_id: int = None,
                     formation: str = "3-4-3") -> Dict:
        """
        Optimize FPL team selection
        
        Args:
            players_df: DataFrame with player data including predicted_points, now_cost
            budget: Available budget in millions
            current_team: List of current player IDs (for transfer optimization)
            free_transfers: Number of free transfers available
            use_wildcard: Whether wildcard is being used
            captain_id: Force specific captain (optional)
            formation: Preferred formation (3-4-3, 3-5-2, 4-3-3, etc.)
            
        Returns:
            Dictionary with optimized team, transfers, and objective value
        """
        try:
            self.logger.info("Starting team optimization...")
            
            # Validate inputs
            if players_df.empty:
                self.logger.error("No player data provided")
                return {}
            
            required_cols = ['id', 'web_name', 'position', 'team', 'now_cost', 'predicted_points']
            missing_cols = [col for col in required_cols if col not in players_df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return {}
            
            # Prepare data
            players = players_df.copy()
            players['price'] = players['now_cost'] / 10.0  # Convert to millions
            players['points'] = players['predicted_points'].fillna(0)
            
            # Filter out unavailable players
            if 'status' in players.columns:
                players = players[players['status'] == 'a']
            
            n_players = len(players)
            if n_players < 15:
                self.logger.error(f"Insufficient players available: {n_players}")
                return {}
            
            # Create solver
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                self.logger.error("Could not create solver")
                return {}
            
            # Decision variables
            x = {}  # x[i] = 1 if player i is in squad
            s = {}  # s[i] = 1 if player i is in starting XI
            c = {}  # c[i] = 1 if player i is captain
            
            for i in range(n_players):
                x[i] = solver.IntVar(0, 1, f'squad_{i}')
                s[i] = solver.IntVar(0, 1, f'start_{i}')
                c[i] = solver.IntVar(0, 1, f'captain_{i}')
            
            # Transfer variables (if optimizing transfers)
            transfer_in = {}
            transfer_out = {}
            if current_team is not None and not use_wildcard:
                for i in range(n_players):
                    transfer_in[i] = solver.IntVar(0, 1, f'transfer_in_{i}')
                    transfer_out[i] = solver.IntVar(0, 1, f'transfer_out_{i}')
            
            # Constraints
            
            # 1. Squad composition (15 players total)
            solver.Add(sum(x[i] for i in range(n_players)) == 15)
            
            # 2. Position constraints for squad
            gk_indices = [i for i, pos in enumerate(players['position']) if pos == 'GKP']
            def_indices = [i for i, pos in enumerate(players['position']) if pos == 'DEF']
            mid_indices = [i for i, pos in enumerate(players['position']) if pos == 'MID']
            fwd_indices = [i for i, pos in enumerate(players['position']) if pos == 'FWD']
            
            solver.Add(sum(x[i] for i in gk_indices) == 2)
            solver.Add(sum(x[i] for i in def_indices) == 5)
            solver.Add(sum(x[i] for i in mid_indices) == 5)
            solver.Add(sum(x[i] for i in fwd_indices) == 3)
            
            # 3. Starting XI constraints (11 players)
            solver.Add(sum(s[i] for i in range(n_players)) == 11)
            
            # 4. Formation constraints for starting XI
            formation_constraints = self._get_formation_constraints(formation)
            
            solver.Add(sum(s[i] for i in gk_indices) == 1)
            solver.Add(sum(s[i] for i in def_indices) >= formation_constraints['def_min'])
            solver.Add(sum(s[i] for i in def_indices) <= formation_constraints['def_max'])
            solver.Add(sum(s[i] for i in mid_indices) >= formation_constraints['mid_min'])
            solver.Add(sum(s[i] for i in mid_indices) <= formation_constraints['mid_max'])
            solver.Add(sum(s[i] for i in fwd_indices) >= formation_constraints['fwd_min'])
            solver.Add(sum(s[i] for i in fwd_indices) <= formation_constraints['fwd_max'])
            
            # 5. Starting players must be in squad
            for i in range(n_players):
                solver.Add(s[i] <= x[i])
            
            # 6. Budget constraint
            total_cost = sum(players.iloc[i]['price'] * x[i] for i in range(n_players))
            solver.Add(total_cost <= budget)
            
            # 7. Max 3 players per team
            teams = players['team'].unique()
            for team_id in teams:
                team_indices = [i for i, t in enumerate(players['team']) if t == team_id]
                if len(team_indices) > 0:
                    solver.Add(sum(x[i] for i in team_indices) <= 3)
            
            # 8. Captain constraints
            solver.Add(sum(c[i] for i in range(n_players)) == 1)
            for i in range(n_players):
                solver.Add(c[i] <= s[i])  # Captain must be in starting XI
            
            # 9. Force specific captain if provided
            if captain_id is not None:
                captain_idx = players[players['id'] == captain_id].index
                if not captain_idx.empty:
                    solver.Add(c[captain_idx[0]] == 1)
            
            # 10. Transfer constraints (if not using wildcard)
            if current_team is not None and not use_wildcard:
                current_indices = [i for i, pid in enumerate(players['id']) if pid in current_team]
                
                # Transfer balance
                for i in range(n_players):
                    if i in current_indices:
                        # Players currently in team
                        solver.Add(x[i] + transfer_out[i] <= 1)  # Can't have and transfer out
                        solver.Add(transfer_in[i] == 0)  # Can't transfer in current players
                    else:
                        # Players not in current team
                        solver.Add(x[i] == transfer_in[i])  # Must transfer in to have
                        solver.Add(transfer_out[i] == 0)  # Can't transfer out players not owned
                
                # Balance transfers
                total_transfers_out = sum(transfer_out[i] for i in range(n_players))
                total_transfers_in = sum(transfer_in[i] for i in range(n_players))
                solver.Add(total_transfers_out == total_transfers_in)
                
                # Limit transfers (hits)
                solver.Add(total_transfers_out <= free_transfers + 10)  # Max 10 hits
            
            # Objective: Maximize expected points (with captain double points minus transfer hits)
            objective_terms = []
            
            # Points from starting XI
            for i in range(n_players):
                objective_terms.append(players.iloc[i]['points'] * s[i])
            
            # Captain bonus points
            for i in range(n_players):
                objective_terms.append(players.iloc[i]['points'] * c[i])
            
            # Transfer hits penalty
            if current_team is not None and not use_wildcard:
                total_transfers_out = sum(transfer_out[i] for i in range(n_players))
                # Penalty for transfers beyond free transfers
                hit_transfers = solver.IntVar(0, 10, 'hit_transfers')
                solver.Add(hit_transfers >= total_transfers_out - free_transfers)
                objective_terms.append(-4 * hit_transfers)  # -4 points per hit
            
            solver.Maximize(sum(objective_terms))
            
            # Solve
            self.logger.info("Solving optimization problem...")
            status = solver.Solve()
            
            if status == pywraplp.Solver.OPTIMAL:
                self.logger.info("Optimal solution found")
                return self._extract_solution(solver, players, x, s, c, transfer_in, transfer_out)
            elif status == pywraplp.Solver.FEASIBLE:
                self.logger.info("Feasible solution found")
                return self._extract_solution(solver, players, x, s, c, transfer_in, transfer_out)
            else:
                self.logger.error("No feasible solution found")
                return {}
            
        except Exception as e:
            self.logger.error(f"Error in team optimization: {str(e)}")
            return {}
    
    def _get_formation_constraints(self, formation: str) -> Dict[str, int]:
        """Get formation constraints for starting XI"""
        formations = {
            "3-4-3": {"def_min": 3, "def_max": 3, "mid_min": 4, "mid_max": 4, "fwd_min": 3, "fwd_max": 3},
            "3-5-2": {"def_min": 3, "def_max": 3, "mid_min": 5, "mid_max": 5, "fwd_min": 2, "fwd_max": 2},
            "4-3-3": {"def_min": 4, "def_max": 4, "mid_min": 3, "mid_max": 3, "fwd_min": 3, "fwd_max": 3},
            "4-4-2": {"def_min": 4, "def_max": 4, "mid_min": 4, "mid_max": 4, "fwd_min": 2, "fwd_max": 2},
            "4-5-1": {"def_min": 4, "def_max": 4, "mid_min": 5, "mid_max": 5, "fwd_min": 1, "fwd_max": 1},
            "5-3-2": {"def_min": 5, "def_max": 5, "mid_min": 3, "mid_max": 3, "fwd_min": 2, "fwd_max": 2},
            "5-4-1": {"def_min": 5, "def_max": 5, "mid_min": 4, "mid_max": 4, "fwd_min": 1, "fwd_max": 1}
        }
        
        return formations.get(formation, formations["3-4-3"])
    
    def _extract_solution(self, solver, players, x, s, c, transfer_in=None, transfer_out=None) -> Dict:
        """Extract solution from solver"""
        try:
            solution = {
                'objective_value': solver.Objective().Value(),
                'squad': [],
                'starting_xi': [],
                'bench': [],
                'captain': None,
                'total_cost': 0,
                'predicted_points': 0,
                'transfers': {'in': [], 'out': [], 'cost': 0}
            }
            
            n_players = len(players)
            
            # Extract squad, starting XI, and bench
            for i in range(n_players):
                if x[i].solution_value() > 0.5:
                    player_info = {
                        'id': players.iloc[i]['id'],
                        'name': players.iloc[i]['web_name'],
                        'position': players.iloc[i]['position'],
                        'team': players.iloc[i].get('team_name', ''),
                        'cost': players.iloc[i]['price'],
                        'predicted_points': players.iloc[i]['points']
                    }
                    solution['squad'].append(player_info)
                    solution['total_cost'] += player_info['cost']
                    
                    if s[i].solution_value() > 0.5:
                        solution['starting_xi'].append(player_info)
                        solution['predicted_points'] += player_info['predicted_points']
                        
                        if c[i].solution_value() > 0.5:
                            solution['captain'] = player_info
                            solution['predicted_points'] += player_info['predicted_points']  # Captain bonus
                    else:
                        solution['bench'].append(player_info)
            
            # Extract transfers
            if transfer_in is not None and transfer_out is not None:
                for i in range(n_players):
                    if transfer_in[i].solution_value() > 0.5:
                        solution['transfers']['in'].append({
                            'id': players.iloc[i]['id'],
                            'name': players.iloc[i]['web_name'],
                            'cost': players.iloc[i]['price']
                        })
                    if transfer_out[i].solution_value() > 0.5:
                        solution['transfers']['out'].append({
                            'id': players.iloc[i]['id'],
                            'name': players.iloc[i]['web_name'],
                            'cost': players.iloc[i]['price']
                        })
                
                # Calculate transfer cost
                num_transfers = len(solution['transfers']['out'])
                free_transfers = 1  # This should be passed as parameter
                if num_transfers > free_transfers:
                    solution['transfers']['cost'] = (num_transfers - free_transfers) * 4
            
            # Sort by position
            position_order = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
            solution['starting_xi'].sort(key=lambda p: position_order.get(p['position'], 5))
            solution['bench'].sort(key=lambda p: position_order.get(p['position'], 5))
            
            self.logger.info(f"Solution extracted: {solution['predicted_points']:.1f} points, £{solution['total_cost']:.1f}M")
            return solution
            
        except Exception as e:
            self.logger.error(f"Error extracting solution: {str(e)}")
            return {}
    
    def analyze_team_value(self, team_solution: Dict) -> Dict:
        """Analyze the value and composition of optimized team"""
        try:
            if not team_solution or 'starting_xi' not in team_solution:
                return {}
            
            analysis = {
                'formation_analysis': {},
                'position_breakdown': {},
                'value_analysis': {},
                'risk_analysis': {}
            }
            
            # Formation analysis
            formation_count = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
            for player in team_solution['starting_xi']:
                formation_count[player['position']] += 1
            
            formation_str = f"{formation_count['DEF']}-{formation_count['MID']}-{formation_count['FWD']}"
            analysis['formation_analysis'] = {
                'formation': formation_str,
                'counts': formation_count
            }
            
            # Position breakdown
            for pos in ['GKP', 'DEF', 'MID', 'FWD']:
                pos_players = [p for p in team_solution['squad'] if p['position'] == pos]
                analysis['position_breakdown'][pos] = {
                    'count': len(pos_players),
                    'total_cost': sum(p['cost'] for p in pos_players),
                    'avg_cost': sum(p['cost'] for p in pos_players) / len(pos_players) if pos_players else 0,
                    'total_points': sum(p['predicted_points'] for p in pos_players)
                }
            
            # Value analysis
            analysis['value_analysis'] = {
                'total_cost': team_solution['total_cost'],
                'budget_remaining': 100.0 - team_solution['total_cost'],
                'points_per_million': team_solution['predicted_points'] / team_solution['total_cost'] if team_solution['total_cost'] > 0 else 0,
                'most_expensive': max(team_solution['squad'], key=lambda p: p['cost']),
                'cheapest': min(team_solution['squad'], key=lambda p: p['cost'])
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing team value: {str(e)}")
            return {}
    
    def optimize_captain_choice(self, team_df: pd.DataFrame) -> Dict:
        """Optimize captain choice for a given team"""
        try:
            if team_df.empty:
                return {}
            
            # Simple captain optimization: highest predicted points with risk consideration
            team_df = team_df.copy()
            
            # Add risk-adjusted points if start_probability is available
            if 'start_probability' in team_df.columns:
                team_df['risk_adjusted_points'] = (
                    team_df['predicted_points'] * team_df['start_probability']
                )
            else:
                team_df['risk_adjusted_points'] = team_df['predicted_points']
            
            # Find best captain choice
            best_captain = team_df.loc[team_df['risk_adjusted_points'].idxmax()]
            
            # Alternative choices
            alternatives = team_df.nlargest(3, 'risk_adjusted_points')[['web_name', 'predicted_points', 'risk_adjusted_points']]
            
            return {
                'recommended_captain': {
                    'id': best_captain.get('id'),
                    'name': best_captain.get('web_name'),
                    'predicted_points': best_captain.get('predicted_points'),
                    'risk_adjusted_points': best_captain.get('risk_adjusted_points'),
                    'position': best_captain.get('position')
                },
                'alternatives': alternatives.to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing captain choice: {str(e)}")
            return {}
