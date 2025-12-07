import math
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import logging
from typing import Dict, List, Optional, Tuple
from backend.src.utils.fpl_rules import FPLRules
from backend.src.utils.fixture_analyzer import FixtureAnalyzer
from backend.src.utils.price_predictor import PricePredictor
from backend.src.utils.strategic_planner import StrategicPlanner, ManagerPreferences
from backend.src.utils.news_analyzer import NewsAnalyzer
from backend.src.utils.effective_ownership import EffectiveOwnershipTracker
from backend.src.models.ensemble_predictor import EnsemblePredictor

class FPLOptimizer:
    """Mathematical optimization for FPL team selection using OR-Tools"""
    
    # Class-level model cache to avoid retraining on every request
    _ensemble_predictor_cache = None
    _fixture_analyzer_cache = None
    _price_predictor_cache = None
    _cache_lock = False  # Simple flag to prevent concurrent training
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = FPLRules()
        
        # Use cached instances if available, otherwise create and cache
        if FPLOptimizer._fixture_analyzer_cache is None:
            self.logger.info("Initializing FixtureAnalyzer (first time)")
            FPLOptimizer._fixture_analyzer_cache = FixtureAnalyzer()
        else:
            self.logger.info("Reusing cached FixtureAnalyzer")
            
        if FPLOptimizer._price_predictor_cache is None:
            self.logger.info("Initializing PricePredictor (first time)")
            FPLOptimizer._price_predictor_cache = PricePredictor()
        else:
            self.logger.info("Reusing cached PricePredictor")
            
        if FPLOptimizer._ensemble_predictor_cache is None:
            self.logger.info("Initializing EnsemblePredictor (first time)")
            FPLOptimizer._ensemble_predictor_cache = EnsemblePredictor()
        else:
            self.logger.info("Reusing cached EnsemblePredictor")
            
        self.fixture_analyzer = FPLOptimizer._fixture_analyzer_cache
        self.price_predictor = FPLOptimizer._price_predictor_cache
        self.ensemble_predictor = FPLOptimizer._ensemble_predictor_cache
        
        # These are lightweight, create new instances
        self.strategic_planner = StrategicPlanner()
        self.news_analyzer = NewsAnalyzer()
        self.eo_tracker = EffectiveOwnershipTracker()
        self.solver = None
        self.status = None
        self.position_map = {
            'Goalkeeper': 'GKP',
            'GKP': 'GKP',
            'GK': 'GKP',
            'Defender': 'DEF',
            'DEF': 'DEF',
            'Midfielder': 'MID',
            'MID': 'MID',
            'Forward': 'FWD',
            'FWD': 'FWD'
        }
        self.position_minimums = {
            'GKP': 1,
            'DEF': 3,
            'MID': 2,
            'FWD': 1
        }
        self._primary_attack_threshold = {
            'MID': {'ga_per90': 0.35, 'xgi_per90': 0.40},
            'FWD': {'ga_per90': 0.35, 'xgi_per90': 0.40},
        }

    @staticmethod
    def _safe_number(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, float) and math.isnan(value):
                return default
            return float(value)
        except Exception:
            return default

    def _get_value(self, player_row, key: str, default: float = 0.0) -> float:
        if isinstance(player_row, dict):
            raw = player_row.get(key, default)
        elif hasattr(player_row, "get"):
            raw = player_row.get(key, default)
        else:
            raw = getattr(player_row, key, default)
        return self._safe_number(raw, default)

    def _minutes_floor(self, current_gw: int) -> float:
        """Dynamic minutes floor so early-season players aren't unfairly filtered."""
        return 180.0 if current_gw <= 6 else 300.0

    def _player_usage_metrics(self, player_row, current_gw: int) -> Dict[str, float]:
        start_prob = self._get_value(player_row, 'start_probability')
        expected_minutes = self._get_value(player_row, 'expected_minutes')
        season_minutes = self._get_value(player_row, 'minutes')
        form = self._get_value(player_row, 'form')
        last_two_override = expected_minutes >= 75.0 and start_prob >= 0.75
        minutes_floor = self._minutes_floor(current_gw)
        return {
            'start_probability': start_prob,
            'expected_minutes': expected_minutes,
            'season_minutes': season_minutes,
            'form': form,
            'meets_minutes_floor': season_minutes >= minutes_floor or last_two_override
        }

    def _calculate_attacking_rates(self, player_row) -> Dict[str, float]:
        minutes_played = max(self._get_value(player_row, 'minutes'), 1.0)
        goals = self._get_value(player_row, 'goals_scored')
        assists = self._get_value(player_row, 'assists')
        xgi = self._get_value(player_row, 'expected_goal_involvements')
        ga_per90 = ((goals + assists) / minutes_played) * 90.0
        xgi_per90 = (xgi / minutes_played) * 90.0
        assists_per90 = (assists / minutes_played) * 90.0
        return {
            'ga_per90': ga_per90,
            'xgi_per90': xgi_per90,
            'assists_per90': assists_per90
        }

    def _calculate_defensive_metrics(self, player_row) -> Dict[str, float]:
        """Calculate defensive contribution metrics for CDMs and defenders."""
        minutes_played = max(self._get_value(player_row, 'minutes'), 1.0)
        tackles = self._get_value(player_row, 'tackles')
        defensive_contrib = self._get_value(player_row, 'defensive_contribution')
        clearances_blocks = self._get_value(player_row, 'clearances_blocks_interceptions')
        bps = self._get_value(player_row, 'bps')
        clean_sheets = self._get_value(player_row, 'clean_sheets')
        
        tackles_per90 = (tackles / minutes_played) * 90.0
        def_contrib_per90 = (defensive_contrib / minutes_played) * 90.0
        cbi_per90 = (clearances_blocks / minutes_played) * 90.0
        bps_per90 = (bps / minutes_played) * 90.0
        
        return {
            'tackles_per90': tackles_per90,
            'defensive_contrib_per90': def_contrib_per90,
            'cbi_per90': cbi_per90,
            'bps_per90': bps_per90,
            'clean_sheets': clean_sheets
        }

    def _passes_availability_filters(self, player_row, current_gw: int) -> bool:
        usage = self._player_usage_metrics(player_row, current_gw)
        if usage['start_probability'] < 0.60:
            return False
        if usage['expected_minutes'] < 60.0:
            return False
        if not usage['meets_minutes_floor']:
            return False
        return True

    def _passes_role_filters(
        self,
        player_row,
        ga_per90: float,
        xgi_per90: float,
        fixture_summary: Optional[Dict],
        points_gain: float
    ) -> bool:
        """Position-specific filters based on how each role actually scores FPL points."""
        if hasattr(player_row, 'get'):
            pos_value = player_row.get('position')
        else:
            pos_value = player_row['position']
        pos_code = self._normalize_position_code(pos_value)
        
        # Get additional metrics needed for position-specific checks
        attacking_rates = self._calculate_attacking_rates(player_row)
        defensive_metrics = self._calculate_defensive_metrics(player_row)
        assists_per90 = attacking_rates['assists_per90']
        start_prob = self._get_value(player_row, 'start_probability', 0.0)
        fixture_difficulty = (fixture_summary or {}).get('avg_difficulty', 3.0)
        implied_conceded = (fixture_summary or {}).get('implied_conceded', 1.6)
        implied_goals = (fixture_summary or {}).get('implied_goals', 1.4)
        
        # FORWARDS: Primary focus on goals/assists
        if pos_code == 'FWD':
            if ga_per90 >= 0.25 or xgi_per90 >= 0.30:
                return True
            # Override: exceptional predicted gain + excellent fixture/start odds
            if points_gain >= 3.0 and start_prob >= 0.70 and fixture_difficulty <= 2.5:
                return True
            return False
        
        # ATTACKING MIDFIELDERS: Goals, assists, or creative output
        if pos_code == 'MID':
            # Primary: G+A threshold
            if ga_per90 >= 0.25 or xgi_per90 >= 0.30:
                return True
            # Creative midfielder: high assist rate even if low goals
            if assists_per90 >= 0.20 and xgi_per90 >= 0.25:
                return True
            # Defensive midfielder path: check defensive contributions
            tackles_per90 = defensive_metrics['tackles_per90']
            def_contrib_per90 = defensive_metrics['defensive_contrib_per90']
            bps_per90 = defensive_metrics['bps_per90']
            # CDM/DM profile: strong defensive output + bonus potential
            if (tackles_per90 >= 2.5 or def_contrib_per90 >= 3.0) and bps_per90 >= 15.0:
                return True
            # Override: exceptional predicted gain + excellent context
            if points_gain >= 3.0 and start_prob >= 0.75 and fixture_difficulty <= 2.5:
                return True
            return False
        
        # DEFENDERS: Clean sheets, defensive contributions, or attacking threat (wing-backs)
        if pos_code == 'DEF':
            # Wing-back/full-back: attacking output acceptable
            if ga_per90 >= 0.10:
                return True
            # Clean sheet probability (low implied conceded = high CS chance)
            if implied_conceded <= 1.2:
                return True
            # Defensive contributions: tackles, clearances, blocks
            tackles_per90 = defensive_metrics['tackles_per90']
            cbi_per90 = defensive_metrics['cbi_per90']
            def_contrib_per90 = defensive_metrics['defensive_contrib_per90']
            if tackles_per90 >= 2.0 or cbi_per90 >= 4.0 or def_contrib_per90 >= 2.5:
                return True
            # Override: exceptional predicted gain + strong CS odds
            if points_gain >= 2.5 and implied_conceded <= 1.4:
                return True
            return False
        
        # GOALKEEPERS: Saves and clean sheet potential
        if pos_code == 'GKP':
            saves_per90 = self._get_value(player_row, 'saves_per_90', 0.0)
            # Strong CS odds (low implied conceded)
            if implied_conceded <= 1.2:
                return True
            # High save rate (busy keeper = save points)
            if saves_per90 >= 3.0:
                return True
            # Override: exceptional predicted gain + decent CS odds
            if points_gain >= 2.5 and implied_conceded <= 1.4:
                return True
            return False
        
        # Fallback for any unhandled positions
        return points_gain >= 1.0

    def _normalize_position_code(self, position: str) -> str:
        """Normalize position names to FPL codes"""
        return self.position_map.get(position, position)
    
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
            
            required_cols = ['id', 'web_name', 'position', 'team', 'now_cost']
            missing_cols = [col for col in required_cols if col not in players_df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return {}
            
            if 'predicted_points' not in players_df.columns:
                raise ValueError("Predicted points are required before optimization. Run the prediction pipeline first.")
            if players_df['predicted_points'].isna().all():
                raise ValueError("Predicted points column is empty. Please generate predictions before optimizing.")
            
            # Prepare data
            players = players_df.copy()
            players['price'] = players['now_cost'] / 10.0  # Convert to millions
            players['points'] = players['predicted_points'].fillna(0)
            
            # Filter out unavailable players, but always keep the manager's current squad
            if 'status' in players.columns:
                active_players = players[players['status'] == 'a']
                if current_team:
                    current_players = players[players['id'].isin(current_team)]
                    if not current_players.empty:
                        players = pd.concat([active_players, current_players]).drop_duplicates(subset='id')
                    else:
                        players = active_players
                else:
                    players = active_players
            
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
            # Map position names to short codes
            position_map = {
                'Goalkeeper': 'GKP',
                'Defender': 'DEF', 
                'Midfielder': 'MID',
                'Forward': 'FWD',
                'GKP': 'GKP',
                'DEF': 'DEF',
                'MID': 'MID', 
                'FWD': 'FWD'
            }
            
            # Normalize positions
            players['position_normalized'] = players['position'].map(position_map).fillna(players['position'])
            
            gk_indices = [i for i, pos in enumerate(players['position_normalized']) if pos == 'GKP']
            def_indices = [i for i, pos in enumerate(players['position_normalized']) if pos == 'DEF']
            mid_indices = [i for i, pos in enumerate(players['position_normalized']) if pos == 'MID']
            fwd_indices = [i for i, pos in enumerate(players['position_normalized']) if pos == 'FWD']
            
            # Check if we have enough players in each position
            self.logger.info(f"Available players - GKP: {len(gk_indices)}, DEF: {len(def_indices)}, MID: {len(mid_indices)}, FWD: {len(fwd_indices)}")
            
            if len(gk_indices) < 2:
                self.logger.error(f"Not enough goalkeepers available: {len(gk_indices)}")
                return {}
            if len(def_indices) < 5:
                self.logger.error(f"Not enough defenders available: {len(def_indices)}")
                return {}
            if len(mid_indices) < 5:
                self.logger.error(f"Not enough midfielders available: {len(mid_indices)}")
                return {}
            if len(fwd_indices) < 3:
                self.logger.error(f"Not enough forwards available: {len(fwd_indices)}")
                return {}
            
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
            
            # Debug budget info
            min_cost = players['price'].min()
            max_cost = players['price'].max()
            avg_cost = players['price'].mean()
            self.logger.info(f"Budget: {budget}M, Min player cost: {min_cost:.1f}M, Max: {max_cost:.1f}M, Avg: {avg_cost:.1f}M")
            
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
    
    def _create_team_analysis(self, team_ids: List[int], players_df: pd.DataFrame, 
                             team_optimization: Dict, free_transfers: int, use_wildcard: bool,
                             starting_ids: Optional[List[int]] = None,
                             chips_available: Optional[List[str]] = None,
                             bank_amount: float = 0.0) -> Dict:
        """Create comprehensive team analysis summary"""
        try:
            if not team_ids or len(team_ids) == 0:
                return {
                    'total_value': 0.0,
                    'predicted_points': 0.0,
                    'team_strength': 0.0,
                    'formation': '3-4-3',
                    'free_transfers': free_transfers,
                    'budget_remaining': round(bank_amount, 1),
                    'bank_remaining': round(bank_amount, 1)
                }
            
            # Get team players
            team_df = players_df[players_df['id'].isin(team_ids)]
            if team_df.empty:
                return {
                    'total_value': 0.0,
                    'predicted_points': 0.0,
                    'team_strength': 0.0,
                    'formation': '3-4-3',
                    'free_transfers': free_transfers,
                    'budget_remaining': round(bank_amount, 1),
                    'bank_remaining': round(bank_amount, 1)
                }
            
            chips_available = chips_available or ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit']
            all_chips = ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit']
            chips_used = [chip for chip in all_chips if chip not in chips_available]
            
            # Determine starting XI from provided IDs or predicted points
            starting_ids = starting_ids or []
            starting_df = team_df[team_df['id'].isin(starting_ids)]
            
            if starting_df.empty or len(starting_df) < 11:
                team_df_sorted = team_df.sort_values('predicted_points', ascending=False)
                starting_df = team_df_sorted.head(min(11, len(team_df_sorted)))
                starting_ids = starting_df['id'].tolist()
            
            bench_df = team_df[~team_df['id'].isin(starting_ids)]
            
            # Calculate position distribution based on starting XI
            position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
            for _, player in starting_df.iterrows():
                pos_code = self._normalize_position_code(player.get('position', ''))
                if pos_code in position_counts:
                    position_counts[pos_code] += 1
            
            formation = f"{position_counts.get('DEF', 0)}-{position_counts.get('MID', 0)}-{position_counts.get('FWD', 0)}"
            
            # Calculate metrics
            total_value = (team_df['now_cost'].sum() / 10.0) if 'now_cost' in team_df.columns else 0.0
            predicted_points = starting_df['predicted_points'].sum() if 'predicted_points' in starting_df.columns else 0.0
            
            # Team strength using starting XI data
            avg_form = starting_df['form'].mean() if 'form' in starting_df.columns else 0.0
            avg_points_per_game = starting_df['points_per_game'].mean() if 'points_per_game' in starting_df.columns else 0.0
            team_strength = min(10.0, (avg_form * 1.2 + avg_points_per_game * 0.4))
            
            starting_xi_value = (starting_df['now_cost'].sum() / 10.0) if 'now_cost' in starting_df.columns else 0.0
            bench_value = (bench_df['now_cost'].sum() / 10.0) if 'now_cost' in bench_df.columns else 0.0
            
            # Top and weakest performers within starting XI
            starting_sorted = starting_df.sort_values('predicted_points', ascending=False)
            bench_sorted = bench_df.sort_values('predicted_points', ascending=False)
            
            top_performer = None
            weakest_link = None
            if not starting_sorted.empty and 'predicted_points' in starting_sorted.columns:
                top = starting_sorted.iloc[0]
                top_performer = {
                    'name': top.get('web_name', 'Unknown'),
                    'predicted_points': round(float(top.get('predicted_points', 0)), 1),
                    'position': top.get('position', 'Unknown')
                }
                if len(starting_sorted) > 0:
                    weak = starting_sorted.iloc[-1]
                    weakest_link = {
                        'name': weak.get('web_name', 'Unknown'),
                        'predicted_points': round(float(weak.get('predicted_points', 0)), 1),
                        'position': weak.get('position', 'Unknown')
                    }
            
            return {
                'total_value': round(total_value, 1),
                'predicted_points': round(predicted_points, 1),
                'team_strength': round(team_strength, 1),
                'formation': formation,
                'free_transfers': free_transfers,
                'budget_remaining': round(bank_amount, 1),
                'bank_remaining': round(bank_amount, 1),
                'chips_available': chips_available,
                'chips_used': chips_used,
                'starting_xi_value': round(starting_xi_value, 1),
                'bench_value': round(bench_value, 1),
                'starting_xi_count': len(starting_df),
                'bench_count': len(bench_df),
                'top_performer': top_performer,
                'weakest_link': weakest_link,
                'position_distribution': position_counts,
                'starting_xi_ids': starting_ids,
                'bench_ids': bench_df['id'].tolist()
            }
            
            
            
        except Exception as e:
            self.logger.error(f"Error creating team analysis: {str(e)}")
            return {
                'total_value': 0.0,
                'predicted_points': 0.0,
                'team_strength': 0.0,
                'formation': '3-4-3',
                'free_transfers': free_transfers,
                'budget_remaining': round(bank_amount, 1),
                'bank_remaining': round(bank_amount, 1)
            }
    
    def _analyze_bench_strength(self, team_ids: List[int], players_df: pd.DataFrame,
                                include_fixture_analysis: bool, fixture_analysis: Dict,
                                starting_ids: Optional[List[int]] = None,
                                current_gw: int = 1) -> Dict:
        """Analyze bench strength and autosub potential"""
        try:
            team_df = players_df[players_df['id'].isin(team_ids)]
            
            if team_df.empty:
                return {
                    'bench_strength': 0.0,
                    'autosub_potential': [],
                    'bench_value': 0.0,
                    'recommendations': []
                }
            
            starting_ids = starting_ids or []
            starting_df = team_df[team_df['id'].isin(starting_ids)]
            
            if starting_df.empty or len(starting_df) < 11:
                team_df_sorted = team_df.sort_values('predicted_points', ascending=False)
                starting_df = team_df_sorted.head(min(11, len(team_df_sorted)))
                starting_ids = starting_df['id'].tolist()
            
            bench = team_df[~team_df['id'].isin(starting_ids)]
            if not bench.empty:
                bench = bench[
                    bench.apply(lambda row: self._passes_availability_filters(row, current_gw), axis=1)
                ]
            
            # Calculate bench strength (0-10 scale)
            bench_avg_points = bench['predicted_points'].mean() if not bench.empty else 0.0
            starting_avg_points = starting_df['predicted_points'].mean() if not starting_df.empty else 1.0
            bench_strength = min(10.0, (bench_avg_points / max(starting_avg_points, 1.0)) * 10)
            
            # Calculate bench value
            bench_value = (bench['now_cost'].sum() / 10.0) if 'now_cost' in bench.columns and not bench.empty else 0.0
            
            # Analyze autosub potential (bench players who might outscore starters)
            autosub_potential = []
            start_pos_counts = {code: 0 for code in self.position_minimums.keys()}
            for _, starter in starting_df.iterrows():
                pos_code = self._normalize_position_code(starter['position'])
                start_pos_counts[pos_code] = start_pos_counts.get(pos_code, 0) + 1
            
            for _, bench_player in bench.iterrows():
                # Find starters in same position with lower predicted points
                same_pos_starters = starting_df[starting_df['position'] == bench_player['position']]
                
                for _, starter in same_pos_starters.iterrows():
                    pos_code = self._normalize_position_code(starter['position'])
                    # Respect minimum requirements for each position
                    if start_pos_counts.get(pos_code, 0) <= self.position_minimums.get(pos_code, 0):
                        continue
                    
                    if bench_player['predicted_points'] > starter['predicted_points']:
                        points_diff = bench_player['predicted_points'] - starter['predicted_points']
                        
                        # Build context
                        context = f"{bench_player['web_name']} ({bench_player['predicted_points']:.1f} pts) could outscore {starter['web_name']} ({starter['predicted_points']:.1f} pts)"
                        
                        # Add fixture context if available
                        if include_fixture_analysis and 'team_summaries' in fixture_analysis:
                            bench_difficulty = fixture_analysis['team_summaries'].get(
                                bench_player['team'], {}
                            ).get('avg_difficulty', 3.0)
                            starter_difficulty = fixture_analysis['team_summaries'].get(
                                starter['team'], {}
                            ).get('avg_difficulty', 3.0)
                            
                            if bench_difficulty < starter_difficulty:
                                context += f" (Better fixture: {bench_difficulty:.1f} vs {starter_difficulty:.1f})"
                        
                        autosub_potential.append({
                            'bench_player': {
                                'id': int(bench_player['id']),
                                'web_name': bench_player['web_name'],
                                'team_name': bench_player.get('team_name', ''),
                                'position': bench_player['position'],
                                'predicted_points': round(bench_player['predicted_points'], 1)
                            },
                            'starter_to_replace': {
                                'id': int(starter['id']),
                                'web_name': starter['web_name'],
                                'team_name': starter.get('team_name', ''),
                                'position': starter['position'],
                                'predicted_points': round(starter['predicted_points'], 1)
                            },
                            'points_gain': round(points_diff, 1),
                            'recommendation': context
                        })
            
            # Sort by points gain
            autosub_potential.sort(key=lambda x: x['points_gain'], reverse=True)
            
            # Generate recommendations
            recommendations = []
            
            if bench_strength < 3:
                recommendations.append({
                    'type': 'warning',
                    'message': f'Weak bench (strength: {bench_strength:.1f}/10). Consider upgrading bench players.',
                    'priority': 'Medium'
                })
            elif bench_strength > 7:
                recommendations.append({
                    'type': 'info',
                    'message': f'Strong bench (strength: {bench_strength:.1f}/10). Good depth for rotation.',
                    'priority': 'Low'
                })
            
            if bench_value > 25:
                recommendations.append({
                    'type': 'info',
                    'message': f'High bench value (£{bench_value:.1f}m). Consider downgrading bench to strengthen starting XI.',
                    'priority': 'Medium'
                })
            elif bench_value < 15:
                recommendations.append({
                    'type': 'warning',
                    'message': f'Low bench value (£{bench_value:.1f}m). Bench may lack quality for emergencies.',
                    'priority': 'Low'
                })
            
            if len(autosub_potential) > 0:
                recommendations.append({
                    'type': 'action',
                    'message': f'{len(autosub_potential)} bench player(s) could outscore current starters. Review lineup.',
                    'priority': 'High'
                })
            
            # Bench boost chip potential
            bench_total_points = bench['predicted_points'].sum() if not bench.empty else 0.0
            bench_boost_value = round(bench_total_points, 1)
            
            if bench_boost_value > 20:
                recommendations.append({
                    'type': 'chip',
                    'message': f'Bench Boost could yield ~{bench_boost_value} points this week. Consider using if chip is available.',
                    'priority': 'High'
                })
            
            return {
                'bench_strength': round(bench_strength, 1),
                'autosub_potential': autosub_potential[:3],  # Top 3 recommendations
                'bench_value': round(bench_value, 1),
                'bench_total_predicted_points': bench_boost_value,
                'recommendations': recommendations,
                'bench_players': [
                    {
                        'id': int(p['id']),
                        'web_name': p['web_name'],
                        'team_name': p.get('team_name', ''),
                        'position': p['position'],
                        'predicted_points': round(p['predicted_points'], 1),
                        'cost': p['now_cost'] / 10.0
                    } for _, p in bench.iterrows()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing bench strength: {str(e)}")
            return {
                'bench_strength': 0.0,
                'autosub_potential': [],
                'bench_value': 0.0,
                'recommendations': []
            }
    
    def _apply_odds_boost(self, players_df: pd.DataFrame, fixture_analysis: Dict) -> pd.DataFrame:
        """Scale predicted points using bookmaker implied goals."""
        try:
            if players_df.empty or not fixture_analysis:
                return players_df
            summaries = fixture_analysis.get('team_summaries') or {}
            implied_values = [
                summary.get('implied_goals')
                for summary in summaries.values()
                if summary.get('implied_goals')
            ]
            if not implied_values:
                return players_df
            league_avg = sum(implied_values) / len(implied_values)
            if league_avg <= 0:
                return players_df
            multipliers = {}
            for team_id, summary in summaries.items():
                implied = summary.get('implied_goals')
                if implied:
                    multipliers[int(team_id)] = max(0.7, min(1.4, implied / league_avg))
            if not multipliers or 'team' not in players_df.columns:
                return players_df
            players_df = players_df.copy()
            players_df['odds_multiplier'] = players_df['team'].map(multipliers).fillna(1.0)
            players_df['predicted_points'] = players_df['predicted_points'] * players_df['odds_multiplier']
            return players_df
        except Exception as e:
            self.logger.error(f"Error applying odds boost: {str(e)}")
            return players_df
    
    def _generate_chip_recommendations(self, team_ids: List[int], players_df: pd.DataFrame,
                                      bench_analysis: Dict, fixture_analysis: Dict,
                                      captain_suggestions: List[Dict], free_transfers: int,
                                      wildcard_active: bool) -> List[Dict]:
        """Generate intelligent chip usage recommendations"""
        try:
            recommendations = []
            
            if not team_ids or len(team_ids) == 0:
                return recommendations
            
            team_df = players_df[players_df['id'].isin(team_ids)]
            
            if team_df.empty:
                return recommendations
            
            fixture_summaries = fixture_analysis.get('team_summaries') if fixture_analysis else {}
            if not fixture_summaries:
                recommendations.append({
                    'chip': 'Fixture Data',
                    'score': 1.0,
                    'reason': 'Fixture difficulty data unavailable. Refresh fixtures before making chip decisions.',
                    'recommended_gameweek': 'N/A',
                    'confidence': 'Low',
                    'priority': 5,
                    'conditions': [
                        'Upcoming fixture data missing',
                        'Run data refresh to enable accurate chip timing'
                    ]
                })
            
            # 1. Bench Boost Analysis
            bench_strength = bench_analysis.get('bench_strength', 0)
            bench_points = bench_analysis.get('bench_total_predicted_points', 0)
            
            if bench_points > 20:
                recommendations.append({
                    'chip': 'Bench Boost',
                    'score': min(10.0, bench_points / 3),
                    'reason': f'Strong bench performance expected (~{bench_points} points from bench)',
                    'recommended_gameweek': 'This week' if bench_points > 25 else 'Soon',
                    'confidence': 'High' if bench_points > 25 else 'Medium',
                    'priority': 1 if bench_points > 25 else 2,
                    'conditions': [
                        f'Bench predicted to score {bench_points} points',
                        f'Bench strength: {bench_strength}/10'
                    ]
                })
            elif bench_strength < 4:
                recommendations.append({
                    'chip': 'Bench Boost',
                    'score': 2.0,
                    'reason': f'Not recommended - weak bench (strength: {bench_strength}/10)',
                    'recommended_gameweek': 'Later (after bench upgrades)',
                    'confidence': 'Low',
                    'priority': 5,
                    'conditions': [
                        'Upgrade bench players first',
                        f'Current bench only expected to score {bench_points} points'
                    ]
                })
            
            # 2. Triple Captain Analysis
            if captain_suggestions and len(captain_suggestions) > 0:
                top_captain = captain_suggestions[0]
                captain_points = top_captain.get('expected_points', 0) / 2  # Divide by 2 since it's already doubled
                captain_score = top_captain.get('captain_score', 0)
                
                # Check for double gameweek from fixture analysis
                has_dgw = False
                if fixture_analysis and 'team_summaries' in fixture_analysis:
                    for team_summary in fixture_analysis['team_summaries'].values():
                        if team_summary.get('has_double_gameweek'):
                            has_dgw = True
                            break
                
                if captain_points > 10 or has_dgw:
                    tc_score = min(10.0, captain_score * 1.2 if has_dgw else captain_score * 0.8)
                    recommendations.append({
                        'chip': 'Triple Captain',
                        'score': tc_score,
                        'reason': f"{top_captain['player']['web_name']} expected {captain_points:.1f} points" + (" in DGW" if has_dgw else " with favorable fixture"),
                        'recommended_gameweek': 'This week' if has_dgw else 'Wait for DGW',
                        'confidence': 'High' if has_dgw else 'Medium',
                        'priority': 1 if has_dgw else 3,
                        'suggested_captain': top_captain['player']['web_name'],
                        'conditions': [
                            f"Captain predicted: {captain_points:.1f} points",
                            'Double Gameweek' if has_dgw else 'Good single fixture',
                            f"Captain confidence: {top_captain.get('confidence', 'Medium')}"
                        ]
                    })
                else:
                    recommendations.append({
                        'chip': 'Triple Captain',
                        'score': 4.0,
                        'reason': 'Wait for better opportunity (DGW or premium fixture)',
                        'recommended_gameweek': 'Wait for DGW',
                        'confidence': 'Low',
                        'priority': 4,
                        'conditions': [
                            'No DGW this week',
                            f'Best captain only expected {captain_points:.1f} points',
                            'Wait for better opportunity'
                        ]
                    })
            
            # 3. Wildcard Analysis
            if not wildcard_active:
                # Count underperforming players
                team_avg_points = team_df['predicted_points'].mean() if not team_df.empty else 0
                underperformers = team_df[team_df['predicted_points'] < team_avg_points * 0.7]
                num_underperformers = len(underperformers)
                
                # Check team value efficiency
                team_value = (team_df['now_cost'].sum() / 10.0) if 'now_cost' in team_df.columns else 100
                
                # Check if many transfers needed
                transfers_needed = num_underperformers
                
                if transfers_needed > free_transfers + 2:
                    wildcard_score = min(10.0, transfers_needed * 1.5)
                    recommendations.append({
                        'chip': 'Wildcard',
                        'score': wildcard_score,
                        'reason': f'{transfers_needed} players need replacing (more than free transfers allow)',
                        'recommended_gameweek': 'This week or next',
                        'confidence': 'High' if transfers_needed > 5 else 'Medium',
                        'priority': 1 if transfers_needed > 5 else 2,
                        'conditions': [
                            f'{num_underperformers} underperforming players',
                            f'Only {free_transfers} free transfer(s) available',
                            f'Would cost {(transfers_needed - free_transfers) * 4} points otherwise'
                        ]
                    })
                elif num_underperformers > 3:
                    recommendations.append({
                        'chip': 'Wildcard',
                        'score': 5.0,
                        'reason': f'{num_underperformers} players underperforming but manageable with transfers',
                        'recommended_gameweek': 'Consider for later',
                        'confidence': 'Medium',
                        'priority': 3,
                        'conditions': [
                            f'{num_underperformers} players could be upgraded',
                            'Not urgent - can use regular transfers',
                            'Save for crisis or major fixture swing'
                        ]
                    })
                else:
                    recommendations.append({
                        'chip': 'Wildcard',
                        'score': 2.0,
                        'reason': 'Team looks good - no urgent need',
                        'recommended_gameweek': 'Save for later',
                        'confidence': 'Low',
                        'priority': 5,
                        'conditions': [
                            'Team performing well',
                            f'Only {num_underperformers} minor issues',
                            'Save for crisis or injury wave'
                        ]
                    })
            
            # 4. Free Hit Analysis
            # Check for blank gameweeks or injury crisis
            if fixture_analysis and 'team_summaries' in fixture_analysis:
                teams_with_fixtures = len([t for t in fixture_analysis['team_summaries'].values() if t.get('avg_difficulty', 0) > 0])
                
                if teams_with_fixtures < 15:  # Likely blank gameweek
                    recommendations.append({
                        'chip': 'Free Hit',
                        'score': 8.0,
                        'reason': 'Possible blank gameweek or limited fixtures',
                        'recommended_gameweek': 'This week',
                        'confidence': 'High',
                        'priority': 1,
                        'conditions': [
                            f'Only {teams_with_fixtures} teams have fixtures',
                            'Free Hit allows optimal team for one week',
                            'Team reverts next week'
                        ]
                    })
                else:
                    recommendations.append({
                        'chip': 'Free Hit',
                        'score': 3.0,
                        'reason': 'Save for blank gameweek or double gameweek',
                        'recommended_gameweek': 'Wait for BGW/DGW',
                        'confidence': 'Low',
                        'priority': 4,
                        'conditions': [
                            'Normal fixture schedule',
                            'Save for blank/double gameweeks',
                            'Or use for injury crisis'
                        ]
                    })
            
            # Sort by priority (lower number = higher priority)
            recommendations.sort(key=lambda x: (x['priority'], -x['score']))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating chip recommendations: {str(e)}")
            return []
    
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
    
    def advanced_optimize_team(self, 
                             players_df: pd.DataFrame,
                             budget: float = 100.0,
                             current_team: List[int] = None,
                             free_transfers: int = 1,
                             use_wildcard: bool = False,
                             captain_id: int = None,
                             formation: str = "3-4-3",
                             preferences: ManagerPreferences = None,
                             include_fixture_analysis: bool = True,
                             include_price_analysis: bool = True,
                             include_strategic_planning: bool = True,
                             starting_xi: Optional[List[int]] = None,
                             chips_available: Optional[List[str]] = None,
                             bank_amount: float = 0.0) -> Dict:
        """
        Advanced team optimization with comprehensive analysis
        
        Args:
            players_df: DataFrame with player data
            budget: Available budget in millions
            current_team: List of current player IDs
            free_transfers: Number of free transfers available
            use_wildcard: Whether wildcard is being used
            captain_id: Force specific captain
            formation: Preferred formation
            preferences: Manager preferences
            include_fixture_analysis: Include fixture difficulty analysis
            include_price_analysis: Include price change predictions
            include_strategic_planning: Include strategic planning features
            starting_xi: Preferred starting XI player IDs selected by manager
            chips_available: Chips still available to the manager
            
        Returns:
            Dictionary with optimized team and comprehensive analysis
        """
        try:
            self.logger.info("Starting advanced team optimization...")
            
            # Initialize results
            result = {
                'optimization_success': False,
                'team_optimization': {},
                'fixture_analysis': {},
                'price_analysis': {},
                'strategic_planning': {},
                'transfer_suggestions': [],
                'captain_suggestions': [],
                'chip_opportunities': [],
                'long_term_strategy': {},
                'confidence_score': 0.0,
                'warnings': [],
                'errors': []
            }
            
            chips_available = chips_available or ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit']
            chips_available = list(dict.fromkeys(chips_available))
            
            current_team = current_team or []
            starting_ids_for_analysis = starting_xi or []
            
            available_player_ids = set(players_df['id'].tolist())
            starting_ids_for_analysis = [
                pid for pid in starting_ids_for_analysis if pid in available_player_ids
            ]
            
            # Generate predictions FIRST before any code that needs them
            if 'predicted_points' not in players_df.columns:
                self.logger.info("Generating predictions before optimization...")
                try:
                    # Train ensemble predictor if not already trained
                    if not self.ensemble_predictor.is_fitted:
                        self.logger.info("Training ensemble predictor...")
                        self.ensemble_predictor.train()
                    
                    # Use ensemble predictor if available
                    if self.ensemble_predictor.is_fitted:
                        # Get minutes predictions
                        from backend.src.models.minutes_model import MinutesPredictor
                        minutes_model = MinutesPredictor()
                        if not minutes_model.is_trained():
                            minutes_model.train()
                        
                        minutes_predictions = minutes_model.predict_minutes(players_df)
                        
                        # Get ensemble predictions (without fixture analysis for now)
                        enhanced_predictions = self.ensemble_predictor.predict_points(
                            players_df, minutes_predictions, None
                        )
                        
                        if not enhanced_predictions.empty and 'predicted_points' in enhanced_predictions.columns:
                            # Merge predictions into players_df
                            merge_cols = ['id', 'predicted_points']
                            if 'id' in players_df.columns:
                                players_df = players_df.merge(
                                    enhanced_predictions[merge_cols],
                                    on='id', how='left', suffixes=('', '_enhanced')
                                )
                                if 'predicted_points_enhanced' in players_df.columns:
                                    players_df['predicted_points'] = players_df['predicted_points_enhanced'].fillna(players_df['predicted_points'])
                                    players_df.drop('predicted_points_enhanced', axis=1, inplace=True)
                            else:
                                raise ValueError("players_df missing 'id' column required for predictions merge")
                        else:
                            raise ValueError("Failed to generate predictions")
                    else:
                        raise ValueError("Ensemble predictor not fitted and could not be trained")
                except Exception as e:
                    self.logger.error(f"Error generating initial predictions: {str(e)}")
                    raise ValueError(f"Failed to generate predictions: {str(e)}")
            
            # Ensure predicted_points column exists after generation
            if 'predicted_points' not in players_df.columns:
                raise ValueError("predicted_points missing from players_df after generation attempt.")
            
            # Ensure we have at least 11 players in the lineup when possible
            if current_team:
                remaining_ids = [pid for pid in current_team if pid not in starting_ids_for_analysis]
                if remaining_ids:
                    remaining_df = players_df[players_df['id'].isin(remaining_ids)].sort_values('predicted_points', ascending=False)
                    for pid in remaining_df['id'].tolist():
                        if pid not in starting_ids_for_analysis:
                            starting_ids_for_analysis.append(int(pid))
                        if len(starting_ids_for_analysis) >= min(11, len(current_team)):
                            break
                
                if len(starting_ids_for_analysis) < min(11, len(current_team)):
                    for pid in remaining_ids:
                        if pid not in starting_ids_for_analysis:
                            starting_ids_for_analysis.append(pid)
                        if len(starting_ids_for_analysis) >= min(11, len(current_team)):
                            break
            
            starting_ids_for_analysis = starting_ids_for_analysis[:11]
            
            # Get additional data for analysis
            from backend.src.core.database import FPLDatabase
            db = FPLDatabase()
            
            fixtures_df = db.get_fixtures()
            teams_df = db.get_teams()
            current_gw = db.get_current_gameweek() or 1
            
            # 1. Fixture Analysis
            if include_fixture_analysis and not fixtures_df.empty and not teams_df.empty:
                try:
                    self.logger.info("Performing fixture analysis...")
                    fixture_analysis = self.fixture_analyzer.calculate_fixture_difficulty_score(
                        fixtures_df, teams_df, gameweeks=5
                    )
                    
                    # Get team fixture summaries
                    team_fixture_summaries = {}
                    for team_id in teams_df['id'].unique():
                        summary = self.fixture_analyzer.get_team_fixture_summary(
                            team_id, fixtures_df, teams_df, gameweeks=5
                        )
                        if summary:
                            team_fixture_summaries[team_id] = summary
                    
                    result['fixture_analysis'] = {
                        'team_summaries': team_fixture_summaries,
                        'double_gameweeks': self.fixture_analyzer.detect_double_gameweeks(fixtures_df),
                        'blank_gameweeks': self.fixture_analyzer.detect_blank_gameweeks(fixtures_df)
                    }
                    result['fixture_analysis']['status'] = 'ok'
                    
                except Exception as e:
                    self.logger.error(f"Error in fixture analysis: {str(e)}")
                    result['errors'].append(f"Fixture analysis failed: {str(e)}")
                    result['fixture_analysis'] = {'team_summaries': {}, 'status': 'unavailable'}
            else:
                result['fixture_analysis'] = {'team_summaries': {}, 'status': 'unavailable'}
            
            fixture_odds_map = db.get_fixture_odds_map()
            team_summaries = result.get('fixture_analysis', {}).get('team_summaries', {})
            if team_summaries and fixture_odds_map:
                for team_id, summary in team_summaries.items():
                    next_fixture = summary.get('next_fixture') or {}
                    fixture_id = next_fixture.get('fixture_id')
                    if not fixture_id:
                        continue
                    odds_row = fixture_odds_map.get(int(fixture_id))
                    if not odds_row:
                        continue
                    venue = next_fixture.get('venue', 'home')
                    if venue == 'home':
                        implied_for = odds_row.get('home_implied_goals')
                        implied_against = odds_row.get('away_implied_goals')
                    else:
                        implied_for = odds_row.get('away_implied_goals')
                        implied_against = odds_row.get('home_implied_goals')
                    summary['implied_goals'] = implied_for
                    summary['implied_conceded'] = implied_against
                    next_fixture['implied_goals'] = implied_for
                    next_fixture['implied_conceded'] = implied_against
                    if odds_row.get('bookmaker'):
                        summary['bookmaker'] = odds_row['bookmaker']
            
            # 2. Price Analysis
            if include_price_analysis:
                try:
                    self.logger.info("Performing price analysis...")
                    if not self.price_predictor.is_trained:
                        self.price_predictor.train()
                    
                    price_predictions = self.price_predictor.predict_price_changes(players_df)
                    result['price_analysis'] = {
                        'price_predictions': price_predictions.to_dict('records') if not price_predictions.empty else [],
                        'rising_players': price_predictions[price_predictions['price_rise_probability'] > 0.7].to_dict('records') if not price_predictions.empty else [],
                        'falling_players': price_predictions[price_predictions['price_fall_probability'] > 0.7].to_dict('records') if not price_predictions.empty else []
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error in price analysis: {str(e)}")
                    result['errors'].append(f"Price analysis failed: {str(e)}")
            
            # 3. Strategic Planning
            if include_strategic_planning:
                try:
                    self.logger.info("Performing strategic planning...")
                    
                    # Chip opportunities
                    chip_opportunities = self.strategic_planner.analyze_chip_usage_opportunities(
                        fixtures_df, players_df, current_gw
                    )
                    
                    # Long-term strategy
                    long_term_strategy = self.strategic_planner.create_long_term_strategy(
                        players_df, fixtures_df, current_gw, preferences
                    )
                    
                    # Captain strategy
                    captain_strategy = self.strategic_planner.optimize_captain_strategy(
                        players_df, fixtures_df, current_gw, preferences
                    )
                    
                    result['strategic_planning'] = {
                        'chip_opportunities': [chip.__dict__ for chip in chip_opportunities],
                        'long_term_strategy': long_term_strategy,
                        'captain_strategy': captain_strategy
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error in strategic planning: {str(e)}")
                    result['errors'].append(f"Strategic planning failed: {str(e)}")
            
            # 4. Enhanced Points Prediction (with fixture analysis)
            try:
                self.logger.info("Enhancing predictions with fixture analysis...")
                
                # Only enhance if we have fixture analysis and predictions already exist
                if include_fixture_analysis and 'team_summaries' in result.get('fixture_analysis', {}) and 'predicted_points' in players_df.columns:
                    # Get fixture analysis for players
                    team_summaries = result['fixture_analysis']['team_summaries']
                    fixture_data = []
                    for team_id, summary in team_summaries.items():
                        fixture_data.append({
                            'team_id': team_id,
                            'avg_difficulty': summary.get('avg_difficulty', 3.0),
                            'has_double_gameweek': summary.get('has_double_gameweek', False),
                            'rotation_risk': summary.get('rotation_risk', 0.1)
                        })
                    fixture_analysis_df = pd.DataFrame(fixture_data)
                    
                    if not fixture_analysis_df.empty and self.ensemble_predictor.is_fitted:
                        # Get minutes predictions
                        from backend.src.models.minutes_model import MinutesPredictor
                        minutes_model = MinutesPredictor()
                        if not minutes_model.is_trained():
                            minutes_model.train()
                        
                        minutes_predictions = minutes_model.predict_minutes(players_df)
                        
                        # Get enhanced ensemble predictions with fixture analysis
                        enhanced_predictions = self.ensemble_predictor.predict_points(
                            players_df, minutes_predictions, fixture_analysis_df
                        )
                        
                        self.logger.info(f"Enhanced predictions shape: {enhanced_predictions.shape if not enhanced_predictions.empty else 'Empty'}")
                        if not enhanced_predictions.empty:
                            self.logger.info(f"Enhanced predictions columns: {list(enhanced_predictions.columns)}")
                            # Check if required columns exist
                            required_cols = ['id', 'predicted_points', 'uncertainty', 'risk_category']
                            available_cols = [col for col in required_cols if col in enhanced_predictions.columns]
                            
                            if 'predicted_points' in available_cols:
                                # Use enhanced predictions for optimization
                                merge_cols = ['id'] + [col for col in available_cols if col != 'id']
                                self.logger.info(f"Merging columns: {merge_cols}")
                                
                                # Check if players_df has 'id' column
                                if 'id' not in players_df.columns:
                                    raise ValueError("players_df missing 'id' column required for predictions merge")
                                try:
                                    players_df = players_df.merge(
                                        enhanced_predictions[merge_cols],
                                        on='id', how='left', suffixes=('', '_enhanced')
                                    )
                                    if 'predicted_points_enhanced' in players_df.columns:
                                        # Update with enhanced predictions where available
                                        players_df['predicted_points'] = players_df['predicted_points_enhanced'].fillna(players_df['predicted_points'])
                                        players_df.drop('predicted_points_enhanced', axis=1, inplace=True)
                                    elif 'predicted_points' not in players_df.columns:
                                        raise ValueError("predicted_points column missing after merge")
                                except Exception as merge_error:
                                    self.logger.warning(f"Failed to merge enhanced predictions: {merge_error}, using existing predictions")
                            else:
                                self.logger.warning("Enhanced predictions missing required columns, using existing predictions")
                        else:
                            self.logger.warning("Enhanced predictions empty, using existing predictions")
                else:
                    self.logger.info("Skipping prediction enhancement (no fixture analysis or predictions already exist)")
                
            except Exception as e:
                self.logger.warning(f"Error enhancing predictions with fixture analysis: {str(e)}")
                # Don't raise - we already have predictions from earlier, enhancement is optional
                if 'predicted_points' not in players_df.columns:
                    raise ValueError(f"Failed to generate predictions: {str(e)}")
            
            players_df = self._apply_odds_boost(players_df, result.get('fixture_analysis', {}))
            
            # 5. Core Team Optimization
            try:
                self.logger.info("Performing core team optimization...")
                team_optimization = self.optimize_team(
                    players_df, budget, current_team, free_transfers, 
                    use_wildcard, captain_id, formation
                )
                
                if team_optimization:
                    result['team_optimization'] = team_optimization
                    result['optimization_success'] = True
                else:
                    result['errors'].append("Core team optimization failed")
                    
            except Exception as e:
                self.logger.error(f"Error in core optimization: {str(e)}")
                result['errors'].append(f"Core optimization failed: {str(e)}")
            
            team_for_captain = current_team if (current_team and len(current_team) > 0) else None
            
            # 6. Enhanced Transfer Recommendations & Captain Suggestions
            if result['optimization_success']:
                try:
                    # 6a. Generate Captain Suggestions
                    self.logger.info("Generating captain suggestions...")
                    self.logger.info(f"Current team size: {len(current_team) if current_team else 0}")
                    captain_suggestions = []
                    
                    # If no current team, extract from optimization result
                    if not team_for_captain and 'squad' in result.get('team_optimization', {}):
                        team_for_captain = [p['id'] for p in result['team_optimization']['squad']]
                        self.logger.info(f"Using optimized team for captain suggestions: {len(team_for_captain)} players")
                    
                    if team_for_captain and len(team_for_captain) > 0:
                        # Get captain suggestions for current team
                        current_team_df = players_df[players_df['id'].isin(team_for_captain)]
                        
                        implied_values = []
                        if include_fixture_analysis and 'team_summaries' in result['fixture_analysis']:
                            implied_values = [
                                summary.get('implied_goals')
                                for summary in result['fixture_analysis']['team_summaries'].values()
                                if summary.get('implied_goals')
                            ]
                        league_implied_avg = sum(implied_values) / len(implied_values) if implied_values else 1.4
                        
                        # Calculate captain scores for each player
                        if 'predicted_points' in current_team_df.columns:
                            captain_scores = []
                            
                            for _, player in current_team_df.iterrows():
                                # Base score from predicted points (40% weight)
                                base_score = player.get('predicted_points', 0) * 0.4
                                
                                # Fixture difficulty multiplier (25% weight)
                                fixture_score = 0
                                fixture_text = ""
                                if include_fixture_analysis and 'team_summaries' in result['fixture_analysis']:
                                    team_summary = result['fixture_analysis']['team_summaries'].get(player['team'], {})
                                    difficulty = team_summary.get('avg_difficulty', 3.0)
                                    implied_text = ""
                                    implied_goals = team_summary.get('implied_goals')
                                    
                                    # Convert difficulty to score (easier = higher score)
                                    # Difficulty 1-2: Excellent (1.5x)
                                    # Difficulty 2-3: Good (1.2x)
                                    # Difficulty 3-4: Average (1.0x)
                                    # Difficulty 4-5: Hard (0.7x)
                                    if difficulty <= 2:
                                        fixture_multiplier = 1.5
                                        fixture_text = f"Easy ({difficulty:.1f}/5)"
                                    elif difficulty <= 3:
                                        fixture_multiplier = 1.2
                                        fixture_text = f"Favorable ({difficulty:.1f}/5)"
                                    elif difficulty <= 4:
                                        fixture_multiplier = 1.0
                                        fixture_text = f"Average ({difficulty:.1f}/5)"
                                    else:
                                        fixture_multiplier = 0.7
                                        fixture_text = f"Tough ({difficulty:.1f}/5)"
                                    
                                    fixture_score = player.get('predicted_points', 0) * 0.25 * fixture_multiplier
                                    
                                    # Bonus for double gameweek
                                    if team_summary.get('has_double_gameweek'):
                                        fixture_score *= 1.8
                                        fixture_text += ", DGW"
                                else:
                                    fixture_score = player.get('predicted_points', 0) * 0.25
                                    implied_text = ""
                                    implied_goals = None
                                
                                if implied_goals:
                                    fixture_score += implied_goals * 0.3
                                    implied_text = f"Implied goals {implied_goals:.1f}"
                                
                                # Form trend (15% weight)
                                form = player.get('form', 0)
                                form_score = 0
                                form_text = ""
                                if form > 7:
                                    form_score = form * 0.15 * 1.3  # Bonus for excellent form
                                    form_text = f"Excellent form ({form:.1f})"
                                elif form > 5:
                                    form_score = form * 0.15
                                    form_text = f"Good form ({form:.1f})"
                                else:
                                    form_score = form * 0.15 * 0.8  # Penalty for poor form
                                    form_text = f"Form ({form:.1f})"
                                
                                # Home/Away consideration (10% weight) - placeholder for now
                                home_away_score = player.get('predicted_points', 0) * 0.10
                                
                                # Ownership consideration for differentials (10% weight)
                                ownership = player.get('selected_by_percent', 50)
                                ownership_score = 0
                                ownership_text = ""
                                if ownership < 5:
                                    ownership_score = player.get('predicted_points', 0) * 0.10 * 1.5  # Differential bonus
                                    ownership_text = f"Differential ({ownership:.1f}%)"
                                elif ownership > 50:
                                    ownership_score = player.get('predicted_points', 0) * 0.10 * 1.1  # Template bonus
                                    ownership_text = f"Template ({ownership:.1f}%)"
                                else:
                                    ownership_score = player.get('predicted_points', 0) * 0.10
                                
                                # Total captain score
                                total_score = base_score + fixture_score + form_score + home_away_score + ownership_score
                                
                                # Build reason
                                reason_parts = []
                                reason_parts.append(f"Predicted {player['predicted_points']:.1f} points")
                                if fixture_text:
                                    reason_parts.append(fixture_text)
                                if implied_text:
                                    reason_parts.append(implied_text)
                                if form_text:
                                    reason_parts.append(form_text)
                                if ownership_text and ownership < 10:
                                    reason_parts.append(ownership_text)
                                
                                captain_scores.append({
                                    'player': player,
                                    'score': total_score,
                                    'reason': ', '.join(reason_parts),
                                    'confidence': 'High' if player.get('uncertainty', 2.0) < 1.5 and fixture_score > 3 else 'Medium'
                                })
                            
                            # Sort by score and take top 3
                            captain_scores.sort(key=lambda x: x['score'], reverse=True)
                            
                            for item in captain_scores[:3]:
                                player = item['player']
                                captain_suggestions.append({
                                    'player': {
                                        'id': int(player['id']),
                                        'web_name': player['web_name'],
                                        'team_name': player.get('team_name', ''),
                                        'position': player['position'],
                                        'expected_points': round(player['predicted_points'], 1)
                                    },
                                    'expected_points': round(
                                        player['predicted_points'] * 2 * (
                                            max(0.9, min(1.5, (team_summary.get('implied_goals', league_implied_avg) or league_implied_avg) / league_implied_avg))
                                        ), 1
                                    ),
                                    'reason': item['reason'],
                                    'confidence': item['confidence'],
                                    'captain_score': round(item['score'], 1)
                                })
                    
                    result['captain_suggestions'] = captain_suggestions[:3]
                    
                    # 6b. Generate Transfer Recommendations
                    # Use current team if provided, otherwise use optimized team
                    team_for_transfers = current_team if (current_team and len(current_team) > 0) else None
                    
                    # If no current team, extract from optimization result  
                    if not team_for_transfers and 'squad' in result.get('team_optimization', {}):
                        team_for_transfers = [p['id'] for p in result['team_optimization']['squad']]
                        self.logger.info(f"Using optimized team for transfer suggestions: {len(team_for_transfers)} players")
                    
                    if team_for_transfers and len(team_for_transfers) > 0:
                        self.logger.info("Generating enhanced transfer recommendations...")
                        
                        # Get current team predictions
                        current_team_df = players_df[players_df['id'].isin(team_for_transfers)]
                        current_starting_ids = [pid for pid in starting_ids_for_analysis if pid in current_team_df['id'].values]
                        required_starters = min(11, len(current_team_df))
                        if len(current_starting_ids) < required_starters:
                            additional_ids = current_team_df[~current_team_df['id'].isin(current_starting_ids)].sort_values('predicted_points', ascending=False)['id'].tolist()
                            for pid in additional_ids:
                                current_starting_ids.append(int(pid))
                                if len(current_starting_ids) >= required_starters:
                                    break
                        current_starting_df = current_team_df[current_team_df['id'].isin(current_starting_ids)]
                        start_pos_counts = {}
                        for _, starter in current_starting_df.iterrows():
                            code = self._normalize_position_code(starter['position'])
                            start_pos_counts[code] = start_pos_counts.get(code, 0) + 1
                        all_predictions = players_df[~players_df['id'].isin(team_for_transfers)]
                        team_summaries = result.get('fixture_analysis', {}).get('team_summaries', {})
                        
                        # Enrich with manager ownership data (top manager insights)
                        try:
                            self.logger.info("Enriching transfer candidates with manager ownership data...")
                            eo_data = self.eo_tracker.calculate_effective_ownership(all_predictions)
                            if not eo_data.empty:
                                all_predictions = all_predictions.merge(
                                    eo_data[['id', 'effective_ownership', 'template_status', 'differential_value']],
                                    on='id',
                                    how='left'
                                )
                                # Also enrich current team for comparison
                                current_team_df = current_team_df.merge(
                                    eo_data[['id', 'effective_ownership', 'template_status', 'differential_value']],
                                    on='id',
                                    how='left'
                                )
                                self.logger.info(f"Enriched {len(all_predictions)} players with manager ownership data")
                            else:
                                self.logger.warning("Manager ownership data unavailable, using basic ownership")
                                all_predictions['effective_ownership'] = all_predictions.get('selected_by_percent', 0)
                                all_predictions['template_status'] = 'unknown'
                                all_predictions['differential_value'] = 100 - all_predictions.get('selected_by_percent', 0)
                        except Exception as e:
                            self.logger.warning(f"Could not enrich with manager data: {e}")
                            all_predictions['effective_ownership'] = all_predictions.get('selected_by_percent', 0)
                            all_predictions['template_status'] = 'unknown'
                            all_predictions['differential_value'] = 100 - all_predictions.get('selected_by_percent', 0)
                        
                        # Convert available bank to 0.1M units (£0.1m = 1 unit)
                        bank_units = max(int(round((bank_amount or 0.0) * 10)), 0)
                        
                        transfer_suggestions = []
                        recommended_in_ids = set()
                        premium_price_threshold = 100  # £10.0m
                        premium_points_threshold = 200
                        premium_ids = set(current_team_df[current_team_df['now_cost'] >= premium_price_threshold]['id'].tolist())
                        premium_ids.update(current_team_df[current_team_df.get('total_points', 0) >= premium_points_threshold]['id'].tolist())
                        captain_candidate_ids = set(starting_ids_for_analysis[:3])
                        
                        fallback_candidates = []
                        
                        for _, current_player in current_team_df.iterrows():
                            # Calculate available funds = bank + player's selling price
                            available_funds = bank_units + current_player['now_cost']
                            
                            self.logger.info(f"Transfer out {current_player['web_name']}: Available funds = £{available_funds/10:.1f}m (£{bank_units/10:.1f}m bank + £{current_player['now_cost']/10:.1f}m from sale)")
                            
                            # Find better alternatives within budget
                            position_alternatives = all_predictions[
                                (all_predictions['position'] == current_player['position']) &
                                (all_predictions['now_cost'] <= available_funds) &  # Must be affordable!
                                (all_predictions['predicted_points'] > current_player['predicted_points'])
                            ].nlargest(5, 'predicted_points')  # Get top 5 to find best option
                            
                            # Only take the BEST alternative that passes availability filters
                            if not position_alternatives.empty:
                                alternative = None
                                alternative_usage = {}
                                for _, candidate in position_alternatives.iterrows():
                                    if not self._passes_availability_filters(candidate, current_gw):
                                        continue
                                    usage = self._player_usage_metrics(candidate, current_gw)
                                    alternative = candidate
                                    alternative_usage = usage
                                    break
                                
                                if alternative is None:
                                    continue
                                
                                if int(alternative['id']) in recommended_in_ids:
                                    continue
                                
                                current_usage = self._player_usage_metrics(current_player, current_gw)
                                current_start_prob = current_usage['start_probability'] or 1.0
                                current_effective = current_player['predicted_points'] * current_start_prob
                                alternative_start_prob = alternative_usage.get('start_probability', 0.0)
                                alternative_minutes = alternative_usage.get('expected_minutes', 0.0)
                                alternative_rates = self._calculate_attacking_rates(alternative)
                                alternative_defensive = self._calculate_defensive_metrics(alternative)
                                
                                alt_effective = alternative['predicted_points'] * alternative_start_prob
                                
                                is_premium = int(current_player['id']) in premium_ids
                                is_captain_candidate = int(current_player['id']) in captain_candidate_ids
                                
                                # Configurable strictness for transfer suggestions
                                # Get strictness from preferences or use default
                                transfer_strictness = preferences.transfer_strictness if (preferences and hasattr(preferences, 'transfer_strictness')) else 'balanced'
                                
                                # Set threshold based on strictness mode
                                if transfer_strictness == 'strict':
                                    min_gain_required = 0.2
                                    if is_premium or is_captain_candidate:
                                        min_gain_required = 1.5  # Original strict threshold
                                elif transfer_strictness == 'relaxed':
                                    min_gain_required = 0.1
                                    if is_premium or is_captain_candidate:
                                        min_gain_required = 0.2  # Very relaxed
                                else:  # 'balanced' (default)
                                    min_gain_required = 0.2
                                    if is_premium or is_captain_candidate:
                                        min_gain_required = 0.5  # Balanced threshold
                                
                                # Factor in venue advantage for the incoming player
                                venue_multiplier = 1.0
                                venue_context = ""
                                alt_team_summary = team_summaries.get(alternative['team'], {}) if team_summaries else {}
                                alt_next = alt_team_summary.get('next_fixture', {}) if alt_team_summary else {}
                                venue_multiplier = alt_next.get('venue_multiplier', 1.0)
                                venue = alt_next.get('venue')
                                head_to_head = alt_next.get('head_to_head', {})
                                if venue:
                                    h2h_desc = ""
                                    if head_to_head and head_to_head.get('samples'):
                                        home_wins = head_to_head.get('home_wins', 0)
                                        away_wins = head_to_head.get('away_wins', 0)
                                        draws = head_to_head.get('draws', 0)
                                        h2h_desc = f" | H2H last {head_to_head['samples']}: {home_wins}-{draws}-{away_wins}"
                                    venue_context = f"{alternative.get('team_name', '')} play {venue} next (difficulty {alt_next.get('difficulty', 'N/A')}){h2h_desc}"
                                alt_effective *= venue_multiplier

                                # Risk-adjusted scores
                                risk_penalty = 0.6
                                alt_uncertainty = float(alternative.get('uncertainty', 2.0) or 2.0)
                                current_uncertainty = float(current_player.get('uncertainty', 2.0) or 2.0)
                                risk_adjusted_alt = alt_effective - risk_penalty * alt_uncertainty
                                risk_adjusted_current = current_effective - risk_penalty * current_uncertainty

                                # Fixture & head-to-head bonuses
                                fixture_bonus = 0.0
                                h2h_bonus = 0.0
                                current_summary = team_summaries.get(current_player['team'], {}) if team_summaries else {}
                                current_difficulty = current_summary.get('avg_difficulty', 3.0)
                                alt_difficulty = alt_team_summary.get('avg_difficulty', 3.0) if alt_team_summary else 3.0
                                fixture_bonus = (current_difficulty - alt_difficulty) * 0.4
                                trend = (head_to_head.get('trend') or "").lower()
                                if trend == 'home edge':
                                    h2h_bonus = 0.3
                                elif trend == 'away edge':
                                    h2h_bonus = 0.15

                                points_gain = (risk_adjusted_alt - risk_adjusted_current) + fixture_bonus + h2h_bonus
                                
                                if not self._passes_role_filters(
                                    alternative,
                                    alternative_rates['ga_per90'],
                                    alternative_rates['xgi_per90'],
                                    alt_team_summary,
                                    points_gain
                                ):
                                    continue
                                
                                if points_gain > min_gain_required:  # Meaningful improvement
                                    cost_change = (alternative['now_cost'] - current_player['now_cost']) / 10
                                    
                                    # Check if transfer requires using bank funds
                                    uses_bank_funds = alternative['now_cost'] > current_player['now_cost']
                                    required_units = max(0, alternative['now_cost'] - current_player['now_cost'])
                                    if uses_bank_funds and required_units > bank_units:
                                        continue
                                    bank_required = required_units / 10
                                    if uses_bank_funds:
                                        new_bank_balance = (bank_units - required_units) / 10
                                    else:
                                        refund_units = max(0, current_player['now_cost'] - alternative['now_cost'])
                                        new_bank_balance = (bank_units + refund_units) / 10
                                    
                                    # Add fixture analysis
                                    fixture_advantage = ""
                                    if include_fixture_analysis and 'team_summaries' in result['fixture_analysis']:
                                        current_team_summary = result['fixture_analysis']['team_summaries'].get(current_player['team'], {})
                                        alternative_team_summary = result['fixture_analysis']['team_summaries'].get(alternative['team'], {})
                                        
                                        current_difficulty = current_team_summary.get('avg_difficulty', 3.0)
                                        alternative_difficulty = alternative_team_summary.get('avg_difficulty', 3.0)
                                        
                                        if alternative_difficulty < current_difficulty - 0.5:
                                            fixture_advantage = f"Easier fixtures: {alternative_difficulty:.1f} vs {current_difficulty:.1f}"
                                    
                                    # Add price analysis
                                    price_advantage = ""
                                    if include_price_analysis and 'price_predictions' in result['price_analysis']:
                                        price_predictions = result['price_analysis']['price_predictions']
                                        current_price_pred = next((p for p in price_predictions if p['id'] == current_player['id']), {})
                                        alternative_price_pred = next((p for p in price_predictions if p['id'] == alternative['id']), {})
                                        
                                        if alternative_price_pred.get('price_rise_probability', 0) > 0.7:
                                            price_advantage = "Price likely to rise"
                                        elif current_price_pred.get('price_fall_probability', 0) > 0.7:
                                            price_advantage = "Current player price likely to fall"
                                    
                                    # Budget context
                                    budget_context = ""
                                    if uses_bank_funds:
                                        budget_context = f"Uses £{bank_required:.1f}m from bank (£{new_bank_balance:.1f}m remaining)"
                                    elif cost_change < 0:
                                        budget_context = f"Frees £{abs(cost_change):.1f}m (£{new_bank_balance:.1f}m in bank)"
                                    
                                    # Determine if player should be in starting XI or bench
                                    is_current_starter = current_player['id'] in current_starting_ids
                                    bench_guidance = "Bench player - Squad depth"
                                    would_be_starter = False
                                    
                                    same_pos_starters = current_starting_df[current_starting_df['position'] == alternative['position']]
                                    
                                    if is_current_starter:
                                        would_be_starter = True
                                        bench_guidance = "Starting XI - Direct replacement"
                                    elif not same_pos_starters.empty:
                                        weakest_same_pos = same_pos_starters.sort_values('predicted_points').iloc[0]
                                        pos_code = self._normalize_position_code(weakest_same_pos['position'])
                                        min_required = self.position_minimums.get(pos_code, 0)
                                        if start_pos_counts.get(pos_code, 0) <= min_required:
                                            bench_guidance = f"Bench player - Keep {weakest_same_pos['web_name']} to meet formation"
                                            would_be_starter = False
                                        elif alternative['predicted_points'] > weakest_same_pos['predicted_points']:
                                            would_be_starter = True
                                            bench_guidance = f"Starting XI - Bench {weakest_same_pos['web_name']}"
                                        else:
                                            bench_guidance = "Bench player - Squad depth"
                                    else:
                                        bench_guidance = "Bench player - Squad depth"
                                    
                                    pos_code = self._normalize_position_code(alternative.get('position', ''))
                                    reason_parts = [
                                        f"{alternative['web_name']} projected {alternative['predicted_points']:.1f} pts with {alternative_start_prob*100:.0f}% start odds (vs {current_player['web_name']} {current_player['predicted_points']:.1f} pts / {current_start_prob*100:.0f}%)",
                                        f"Risk-adjusted delta +{(risk_adjusted_alt - risk_adjusted_current):.1f}"
                                    ]
                                    # Position-specific metric highlights
                                    if pos_code == 'FWD':
                                        reason_parts.append(
                                            f"Attacking output: G+A {alternative_rates['ga_per90']:.2f}/90, xGI {alternative_rates['xgi_per90']:.2f}/90"
                                        )
                                    elif pos_code == 'MID':
                                        if alternative_rates['ga_per90'] >= 0.35:
                                            reason_parts.append(
                                                f"Attacking: G+A {alternative_rates['ga_per90']:.2f}/90, Assists {alternative_rates['assists_per90']:.2f}/90"
                                            )
                                        elif alternative_defensive['tackles_per90'] >= 2.5:
                                            reason_parts.append(
                                                f"Defensive: {alternative_defensive['tackles_per90']:.1f} tackles/90, {alternative_defensive['defensive_contrib_per90']:.1f} def contrib/90, BPS {alternative_defensive['bps_per90']:.1f}/90"
                                            )
                                        else:
                                            reason_parts.append(
                                                f"Creative: Assists {alternative_rates['assists_per90']:.2f}/90, xGI {alternative_rates['xgi_per90']:.2f}/90"
                                            )
                                    elif pos_code == 'DEF':
                                        if alternative_rates['ga_per90'] >= 0.15:
                                            reason_parts.append(
                                                f"Attacking defender: G+A {alternative_rates['ga_per90']:.2f}/90"
                                            )
                                        else:
                                            reason_parts.append(
                                                f"Defensive: {alternative_defensive['tackles_per90']:.1f} tackles/90, {alternative_defensive['cbi_per90']:.1f} CBI/90"
                                            )
                                    elif pos_code == 'GKP':
                                        saves_per90 = self._get_value(alternative, 'saves_per_90', 0.0)
                                        reason_parts.append(
                                            f"Goalkeeper: {saves_per90:.1f} saves/90, CS odds from fixture analysis"
                                        )
                                    else:
                                        reason_parts.append(
                                            f"Per-90 impact: G+A {alternative_rates['ga_per90']:.2f}, xGI {alternative_rates['xgi_per90']:.2f}"
                                        )
                                    if alternative_minutes:
                                        reason_parts.append(f"Expected minutes ≈{int(alternative_minutes)}")
                                    if fixture_advantage:
                                        reason_parts.append(fixture_advantage)
                                    if venue_context:
                                        reason_parts.append(venue_context)
                                    implied_goals = alternative_team_summary.get('implied_goals')
                                    if implied_goals:
                                        reason_parts.append(f"Bookmakers expect {implied_goals:.1f} {alternative.get('team_name', '')} goals")
                                    fixture_delta_value = None
                                    if current_summary and alt_team_summary:
                                        fixture_delta_value = current_difficulty - alt_difficulty

                                    if price_advantage:
                                        reason_parts.append(price_advantage)
                                    if budget_context:
                                        reason_parts.append(budget_context)
                                    
                                    transfer_dict = {
                                        'player_out': {
                                            'id': int(current_player['id']),
                                            'name': current_player['web_name'],
                                            'web_name': current_player['web_name'],
                                            'team': current_player.get('team_name', ''),
                                            'team_name': current_player.get('team_name', ''),
                                            'position': current_player['position'],
                                            'cost': current_player['now_cost'] / 10.0,
                                            'now_cost': current_player['now_cost'] / 10.0,
                                            'predicted_points': round(current_player['predicted_points'], 1),
                                            'expected_points': round(current_player['predicted_points'], 1),
                                            'total_points': int(current_player.get('total_points', 0)),
                                            'uncertainty': round(current_player.get('uncertainty', 2.0), 1),
                                            'risk_category': current_player.get('risk_category', 'Medium Risk')
                                        },
                                        'player_in': {
                                            'id': int(alternative['id']),
                                            'name': alternative['web_name'],
                                            'web_name': alternative['web_name'],
                                            'team': alternative.get('team_name', ''),
                                            'team_name': alternative.get('team_name', ''),
                                            'position': alternative['position'],
                                            'cost': alternative['now_cost'] / 10.0,
                                            'now_cost': alternative['now_cost'] / 10.0,
                                            'predicted_points': round(alternative['predicted_points'], 1),
                                            'expected_points': round(alternative['predicted_points'], 1),
                                            'total_points': int(alternative.get('total_points', 0)),
                                            'uncertainty': round(alternative.get('uncertainty', 2.0), 1),
                                            'risk_category': alternative.get('risk_category', 'Medium Risk')
                                        },
                                        'points_gain': round(points_gain, 1),
                                        'cost_change': round(cost_change, 1),
                                        'reason': " ".join(reason_parts),
                                        'priority': 'High' if points_gain > 2 else 'Medium',
                                        'confidence': 'High' if alternative.get('uncertainty', 2.0) < 1.5 else 'Medium',
                                        'bank_remaining': round(new_bank_balance, 1),
                                        'uses_bank_funds': uses_bank_funds,
                                        'bench_guidance': bench_guidance,
                                        'should_start': would_be_starter,
                                        'metrics': {
                                            'start_probability': round(alternative_start_prob, 2),
                                            'expected_minutes': round(alternative_minutes, 1),
                                            'ga_per90': round(alternative_rates['ga_per90'], 2),
                                            'xgi_per90': round(alternative_rates['xgi_per90'], 2),
                                            'assists_per90': round(alternative_rates['assists_per90'], 2),
                                            'tackles_per90': round(alternative_defensive['tackles_per90'], 2),
                                            'defensive_contrib_per90': round(alternative_defensive['defensive_contrib_per90'], 2),
                                            'cbi_per90': round(alternative_defensive['cbi_per90'], 2),
                                            'bps_per90': round(alternative_defensive['bps_per90'], 2),
                                            'fixture_difficulty': alt_next.get('difficulty') if alt_next else None,
                                            'fixture_delta': round(fixture_delta_value, 2) if fixture_delta_value is not None else None,
                                            'implied_goals': round(alt_team_summary.get('implied_goals', 0.0), 2) if alt_team_summary else None,
                                            'implied_conceded': round(alt_team_summary.get('implied_conceded', 1.6), 2) if alt_team_summary else None,
                                            'ownership_delta': round(
                                                self._get_value(alternative, 'selected_by_percent') - self._get_value(current_player, 'selected_by_percent'),
                                                1
                                            ),
                                            'position': pos_code,
                                            # Manager ML insights
                                            'effective_ownership': round(alternative.get('effective_ownership', 0), 1),
                                            'template_status': alternative.get('template_status', 'unknown'),
                                            'differential_value': round(alternative.get('differential_value', 0), 1),
                                            'current_template_status': current_player.get('template_status', 'unknown')
                                        }
                                    }
                                    
                                    # Apply ownership-based scoring adjustments (Manager ML integration)
                                    ownership_boost = 0.0
                                    ownership_notes = []
                                    
                                    # Boost if top managers are buying this player (effective ownership trending up)
                                    alt_eo = alternative.get('effective_ownership', alternative.get('selected_by_percent', 0))
                                    curr_eo = current_player.get('effective_ownership', current_player.get('selected_by_percent', 0))
                                    eo_delta = alt_eo - curr_eo
                                    
                                    if eo_delta > 5:  # Significant ownership increase
                                        ownership_boost += 0.3
                                        ownership_notes.append(f"Top managers favor this (+{eo_delta:.1f}% EO)")
                                    elif eo_delta > 2:
                                        ownership_boost += 0.15
                                        ownership_notes.append(f"Rising ownership among top managers")
                                    
                                    # Penalize selling template players (risky move)
                                    if current_player.get('template_status') == 'template':
                                        ownership_boost -= 0.2
                                        ownership_notes.append("Selling template player (risky)")
                                    
                                    # Boost differential picks if they have good predicted points
                                    if alternative.get('template_status') == 'differential' and points_gain >= 1.5:
                                        ownership_boost += 0.2
                                        ownership_notes.append("Smart differential pick")
                                    
                                    # Apply the ownership boost
                                    if ownership_boost != 0:
                                        transfer_dict['points_gain'] = round(points_gain + ownership_boost, 1)
                                        transfer_dict['ownership_boost'] = round(ownership_boost, 2)
                                        if ownership_notes:
                                            transfer_dict['reason'] += " | " + "; ".join(ownership_notes)
                                    
                                    transfer_suggestions.append(transfer_dict)
                                    fallback_candidates.append(transfer_dict)
                                    recommended_in_ids.add(int(alternative['id']))
                        
                        # Sort by points gain
                        transfer_suggestions.sort(key=lambda x: x['points_gain'], reverse=True)
                        
                        # Separate free transfers from hit transfers
                        free_transfer_suggestions = []
                        worthwhile_hit_suggestions = []
                        
                        for i, transfer in enumerate(transfer_suggestions):
                            transfer_num = i + 1
                            
                            if transfer_num <= free_transfers:
                                # Free transfers - always include
                                transfer['cost_warning'] = None
                                transfer['is_free'] = True
                                transfer['transfer_cost'] = 0
                                free_transfer_suggestions.append(transfer)
                            elif use_wildcard:
                                # Wildcard active - all transfers are free
                                transfer['cost_warning'] = None
                                transfer['is_free'] = True
                                transfer['transfer_cost'] = 0
                                transfer['reason'] += " (Wildcard)"
                                free_transfer_suggestions.append(transfer)
                            else:
                                # Calculate hit cost
                                points_hit = (len(free_transfer_suggestions) - free_transfers + len(worthwhile_hit_suggestions) + 1) * 4
                                net_gain = transfer['points_gain'] - points_hit
                                
                                # Only include if net gain is positive and significant
                                if net_gain >= 2.0:  # Must gain at least 2 points after hit
                                    transfer['cost_warning'] = f"-{points_hit} points hit"
                                    transfer['is_free'] = False
                                    transfer['transfer_cost'] = points_hit
                                    transfer['net_gain'] = net_gain
                                    transfer['priority'] = 'Medium' if net_gain > 4 else 'Low'
                                    transfer['reason'] += f" (Net: +{net_gain:.1f} after -{points_hit}pt hit - WORTH THE HIT)"
                                    worthwhile_hit_suggestions.append(transfer)
                                # Otherwise exclude - not worth the hit
                        
                        # Respect user's free transfers limit
                        if use_wildcard:
                            # On wildcard, still limit to five high-confidence moves
                            result['transfer_suggestions'] = free_transfer_suggestions[:5]
                        else:
                            # Prioritize free transfers, then add worthwhile hits (max 2 extra)
                            final_suggestions = free_transfer_suggestions[:free_transfers]
                            
                            # Only add hit suggestions if they're significantly beneficial
                            if worthwhile_hit_suggestions:
                                # Limit to 2 additional hits maximum, and only if net gain > 2
                                extra_hits = [t for t in worthwhile_hit_suggestions if t['net_gain'] >= 2.0][:2]
                                final_suggestions.extend(extra_hits)
                            
                            result['transfer_suggestions'] = final_suggestions[:5]
                        
                        # Fallback: if still empty, surface best raw improvements with low confidence tags
                        if not result['transfer_suggestions']:
                            self.logger.info("No transfer suggestions passed thresholds; using fallback recommendations.")
                            
                            # Less aggressive filtering - show transfers even if below ideal threshold
                            filtered_candidates = []
                            for cand in fallback_candidates:
                                premium_cost = cand['player_out'].get('cost', 0)
                                premium_points = cand['player_out'].get('predicted_points', 0)
                                is_premium_out = premium_cost >= 9 or premium_points >= 4
                                
                                # Relaxed threshold: only filter if gain is very small for premium
                                if is_premium_out and cand['points_gain'] < 0.3:  # Lowered from 2.5
                                    continue
                                
                                # Tag with low confidence if below optimal threshold
                                if cand['points_gain'] < 1.0:
                                    cand['confidence'] = 'Low'
                                    cand['priority'] = 'Low'
                                    cand['reason'] += " | Low confidence - marginal gain"
                                
                                filtered_candidates.append(cand)
                            
                            fallback_candidates = filtered_candidates
                            fallback_candidates.sort(key=lambda x: x['points_gain'], reverse=True)
                            
                            if not fallback_candidates:
                                # Build basic fallback with even more relaxed filters
                                basic_suggestions = []
                                for _, current_player in current_team_df.iterrows():
                                    alternatives = all_predictions[
                                        (all_predictions['position'] == current_player['position']) &
                                        (all_predictions['predicted_points'] > current_player['predicted_points'])
                                    ].nlargest(3, 'predicted_points')  # Get top 3 instead of 1
                                    
                                    if alternatives.empty:
                                        continue
                                    
                                    # Take the best alternative
                                    alternative = alternatives.iloc[0]
                                    gain = alternative['predicted_points'] - current_player['predicted_points']
                                    
                                    if gain <= 0:
                                        continue
                                    
                                    # Determine confidence based on gain
                                    if gain >= 1.0:
                                        confidence = 'Medium'
                                        priority = 'Medium'
                                    else:
                                        confidence = 'Low'
                                        priority = 'Low'
                                    
                                    basic_suggestions.append({
                                        'player_out': {
                                            'id': int(current_player['id']),
                                            'name': current_player['web_name'],
                                            'web_name': current_player['web_name'],
                                            'team_name': current_player.get('team_name', ''),
                                            'position': current_player['position'],
                                            'cost': current_player['now_cost'] / 10.0,
                                            'predicted_points': round(current_player['predicted_points'], 1)
                                        },
                                        'player_in': {
                                            'id': int(alternative['id']),
                                            'name': alternative['web_name'],
                                            'web_name': alternative['web_name'],
                                            'team_name': alternative.get('team_name', ''),
                                            'position': alternative['position'],
                                            'cost': alternative['now_cost'] / 10.0,
                                            'predicted_points': round(alternative['predicted_points'], 1)
                                        },
                                        'points_gain': round(gain, 1),
                                        'priority': priority,
                                        'confidence': confidence,
                                        'reason': f"{alternative['web_name']} offers +{gain:.1f} pts over {current_player['web_name']}" + 
                                                 (" | Low confidence - small gain" if confidence == 'Low' else ""),
                                        'is_free': True,
                                        'transfer_cost': 0,
                                        'bench_guidance': 'Starting XI - direct upgrade',
                                        'should_start': True
                                    })
                                
                                # Only filter out extremely poor suggestions
                                filtered_basic = []
                                for cand in basic_suggestions:
                                    premium_cost = cand['player_out'].get('cost', 0)
                                    premium_points = cand['player_out'].get('predicted_points', 0)
                                    
                                    # Very relaxed: only filter if premium AND gain < 0.2
                                    if (premium_cost >= 9 or premium_points >= 4) and cand['points_gain'] < 0.2:  # Lowered from 2.5
                                        continue
                                    
                                    filtered_basic.append(cand)
                                
                                fallback_candidates = filtered_basic
                            
                            # Show top 3-5 transfers even if low confidence
                            num_suggestions = max(3, min(5, free_transfers))
                            result['transfer_suggestions'] = fallback_candidates[:num_suggestions]
                    
                except Exception as e:
                    self.logger.error(f"Error generating transfer recommendations: {str(e)}")
                    result['warnings'].append(f"Transfer recommendations limited: {str(e)}")
            
            # 7. Calculate Overall Confidence Score
            confidence_factors = []
            
            if result['optimization_success']:
                confidence_factors.append(0.4)  # Core optimization success
            
            if include_fixture_analysis and 'team_summaries' in result['fixture_analysis']:
                confidence_factors.append(0.2)  # Fixture analysis available
            
            if include_price_analysis and 'price_predictions' in result['price_analysis']:
                confidence_factors.append(0.2)  # Price analysis available
            
            if include_strategic_planning and 'chip_opportunities' in result['strategic_planning']:
                confidence_factors.append(0.2)  # Strategic planning available
            
            result['confidence_score'] = sum(confidence_factors) if confidence_factors else 0.0
            
            # 8. News Analysis
            if include_strategic_planning:
                try:
                    self.logger.info("Performing news analysis...")
                    
                    # Analyze player news and injuries
                    news_analysis = self.news_analyzer.analyze_player_news(players_df)
                    high_risk_players = self.news_analyzer.identify_high_risk_players(players_df)
                    news_summary = self.news_analyzer.get_news_summary(players_df)
                    
                    result['news_analysis'] = {
                        'high_risk_players': high_risk_players,
                        'news_summary': news_summary,
                        'total_analyzed': len(news_analysis)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error in news analysis: {str(e)}")
                    result['warnings'].append(f"News analysis limited: {str(e)}")
            
            # 9. Effective Ownership Analysis
            if include_strategic_planning:
                try:
                    self.logger.info("Performing effective ownership analysis...")
                    
                    # Calculate effective ownership
                    eo_data = self.eo_tracker.calculate_effective_ownership(players_df)
                    
                    # Identify template team
                    template_team = self.eo_tracker.identify_template_team(players_df)
                    
                    # Find differential opportunities
                    differentials = self.eo_tracker.find_differential_opportunities(players_df)
                    
                    # Calculate team risk profile
                    team_risk = self.eo_tracker.calculate_team_risk_profile(current_team, players_df) if current_team else {}
                    
                    # Get ownership insights
                    ownership_insights = self.eo_tracker.get_ownership_insights(players_df)
                    
                    # Recommend ownership strategy
                    ownership_strategy = self.eo_tracker.recommend_ownership_strategy(
                        current_team, players_df, 
                        preferences.risk_tolerance if preferences else 0.5
                    )
                    
                    result['effective_ownership'] = {
                        'template_team': template_team,
                        'differential_opportunities': differentials,
                        'team_risk_profile': team_risk,
                        'ownership_insights': ownership_insights,
                        'ownership_strategy': ownership_strategy
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error in effective ownership analysis: {str(e)}")
                    result['warnings'].append(f"EO analysis limited: {str(e)}")
            
            # 10. Final Confidence Score Calculation
            confidence_factors = []
            
            if result['optimization_success']:
                confidence_factors.append(0.3)  # Core optimization success
            
            if include_fixture_analysis and 'team_summaries' in result['fixture_analysis']:
                confidence_factors.append(0.15)  # Fixture analysis available
            
            if include_price_analysis and 'price_predictions' in result['price_analysis']:
                confidence_factors.append(0.15)  # Price analysis available
            
            if include_strategic_planning and 'chip_opportunities' in result['strategic_planning']:
                confidence_factors.append(0.15)  # Strategic planning available
            
            if 'news_analysis' in result and result['news_analysis'].get('total_analyzed', 0) > 0:
                confidence_factors.append(0.1)  # News analysis available
            
            if 'effective_ownership' in result and result['effective_ownership'].get('template_team'):
                confidence_factors.append(0.1)  # EO analysis available
            
            # Add model ensemble confidence
            if self.ensemble_predictor.is_fitted:
                confidence_factors.append(0.05)  # Advanced ML models
            
            result['confidence_score'] = sum(confidence_factors) if confidence_factors else 0.0
            
            # Add comprehensive feature summary
            result['features_used'] = {
                'fixture_analysis': include_fixture_analysis,
                'price_analysis': include_price_analysis,
                'strategic_planning': include_strategic_planning,
                'news_analysis': 'news_analysis' in result,
                'effective_ownership': 'effective_ownership' in result,
                'ensemble_models': self.ensemble_predictor.is_fitted,
                'total_features': len(confidence_factors)
            }
            
            # 11. Create Team Analysis Summary
            self.logger.info("Creating team analysis summary...")
            team_analysis = self._create_team_analysis(
                team_for_captain if team_for_captain else [],
                players_df,
                result.get('team_optimization', {}),
                free_transfers,
                use_wildcard,
                starting_ids=starting_ids_for_analysis,
                chips_available=chips_available,
                bank_amount=bank_amount
            )
            result['team_analysis'] = team_analysis
            
            # 12. Bench Strength Analysis
            if team_for_captain:
                self.logger.info("Performing bench strength analysis...")
                bench_analysis = self._analyze_bench_strength(
                    team_for_captain,
                    players_df,
                    include_fixture_analysis,
                    result.get('fixture_analysis', {}),
                    starting_ids=starting_ids_for_analysis,
                    current_gw=current_gw
                )
                result['bench_analysis'] = bench_analysis
            else:
                result['bench_analysis'] = {}
            
            # 13. Enhanced Chip Recommendations
            self.logger.info("Generating chip recommendations...")
            chip_recommendations = self._generate_chip_recommendations(
                team_for_captain if team_for_captain else [],
                players_df,
                result.get('bench_analysis', {}),
                result.get('fixture_analysis', {}),
                result.get('captain_suggestions', []),
                free_transfers,
                use_wildcard
            )
            result['chip_opportunities'] = chip_recommendations
            
            self.logger.info(f"Advanced optimization completed. Confidence: {result['confidence_score']:.2f}")
            self.logger.info(f"Features used: {result['features_used']['total_features']}/6")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in advanced optimization: {str(e)}")
            return {
                'optimization_success': False,
                'errors': [f"Advanced optimization failed: {str(e)}"],
                'confidence_score': 0.0
            }
