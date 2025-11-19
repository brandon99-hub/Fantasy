import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import logging
from typing import Dict, List, Optional, Tuple
from utils.fpl_rules import FPLRules
from utils.fixture_analyzer import FixtureAnalyzer
from utils.price_predictor import PricePredictor
from utils.strategic_planner import StrategicPlanner, ManagerPreferences
from utils.news_analyzer import NewsAnalyzer
from utils.effective_ownership import EffectiveOwnershipTracker
from models.ensemble_predictor import EnsemblePredictor

class FPLOptimizer:
    """Mathematical optimization for FPL team selection using OR-Tools"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = FPLRules()
        self.fixture_analyzer = FixtureAnalyzer()
        self.price_predictor = PricePredictor()
        self.strategic_planner = StrategicPlanner()
        self.news_analyzer = NewsAnalyzer()
        self.eo_tracker = EffectiveOwnershipTracker()
        self.ensemble_predictor = EnsemblePredictor()
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
            
            required_cols = ['id', 'web_name', 'position', 'team', 'now_cost']
            missing_cols = [col for col in required_cols if col not in players_df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return {}
            
            # Add predicted_points if missing (use form as fallback)
            if 'predicted_points' not in players_df.columns:
                if 'form' in players_df.columns:
                    players_df['predicted_points'] = players_df['form'].fillna(0)
                    self.logger.info("Using 'form' as predicted_points fallback")
                else:
                    players_df['predicted_points'] = 0
                    self.logger.warning("No predicted_points or form data available, using 0")
            
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
                             include_strategic_planning: bool = True) -> Dict:
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
                'transfer_recommendations': [],
                'captain_recommendations': [],
                'chip_opportunities': [],
                'long_term_strategy': {},
                'confidence_score': 0.0,
                'warnings': [],
                'errors': []
            }
            
            # Get additional data for analysis
            from database import FPLDatabase
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
                    
                except Exception as e:
                    self.logger.error(f"Error in fixture analysis: {str(e)}")
                    result['errors'].append(f"Fixture analysis failed: {str(e)}")
            
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
            
            # 4. Enhanced Points Prediction
            try:
                self.logger.info("Generating enhanced points predictions...")
                
                # Train ensemble predictor if not already trained
                if not self.ensemble_predictor.is_fitted:
                    self.logger.info("Training ensemble predictor...")
                    self.ensemble_predictor.train()
                
                # Use ensemble predictor if available
                if self.ensemble_predictor.is_fitted:
                    # Get minutes predictions
                    from models.minutes_model import MinutesPredictor
                    minutes_model = MinutesPredictor()
                    if not minutes_model.is_trained():
                        minutes_model.train()
                    
                    minutes_predictions = minutes_model.predict_minutes(players_df)
                    
                    # Get fixture analysis for players
                    fixture_analysis_df = None
                    if include_fixture_analysis and 'team_summaries' in result['fixture_analysis']:
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
                    
                    # Get ensemble predictions
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
                                self.logger.error("players_df missing 'id' column")
                                players_df['predicted_points'] = players_df.get('form', 0)
                                players_df['uncertainty'] = 2.0
                                players_df['risk_category'] = 'Medium Risk'
                            else:
                                try:
                                    players_df = players_df.merge(
                                        enhanced_predictions[merge_cols],
                                        on='id', how='left', suffixes=('', '_enhanced')
                                    )
                                    # Check if predicted_points column exists after merge
                                    if 'predicted_points' in players_df.columns:
                                        players_df['predicted_points'] = players_df['predicted_points'].fillna(players_df.get('form', 0))
                                    elif 'predicted_points_enhanced' in players_df.columns:
                                        players_df['predicted_points'] = players_df['predicted_points_enhanced'].fillna(players_df.get('form', 0))
                                        players_df.drop('predicted_points_enhanced', axis=1, inplace=True)
                                    else:
                                        self.logger.error("predicted_points column missing after merge")
                                        players_df['predicted_points'] = players_df.get('form', 0)
                                except Exception as merge_error:
                                    self.logger.error(f"Merge error: {str(merge_error)}")
                                    players_df['predicted_points'] = players_df.get('form', 0)
                                    players_df['uncertainty'] = 2.0
                                    players_df['risk_category'] = 'Medium Risk'
                        else:
                            self.logger.warning("Enhanced predictions missing required columns, using fallback")
                            players_df['predicted_points'] = players_df.get('form', 0)
                            players_df['uncertainty'] = 2.0
                            players_df['risk_category'] = 'Medium Risk'
                
                else:
                    # Fallback to basic form-based predictions
                    players_df['predicted_points'] = players_df.get('form', 0)
                    players_df['uncertainty'] = 2.0
                    players_df['risk_category'] = 'Medium Risk'
                
            except Exception as e:
                self.logger.error(f"Error in enhanced predictions: {str(e)}")
                result['warnings'].append(f"Using fallback predictions: {str(e)}")
                players_df['predicted_points'] = players_df.get('form', 0)
            
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
            
            # 6. Enhanced Transfer Recommendations
            if result['optimization_success'] and current_team:
                try:
                    self.logger.info("Generating enhanced transfer recommendations...")
                    
                    # Get current team predictions
                    current_team_df = players_df[players_df['id'].isin(current_team)]
                    all_predictions = players_df[~players_df['id'].isin(current_team)]
                    
                    transfer_recommendations = []
                    
                    for _, current_player in current_team_df.iterrows():
                        # Find better alternatives
                        position_alternatives = all_predictions[
                            (all_predictions['position'] == current_player['position']) &
                            (all_predictions['now_cost'] >= current_player['now_cost'] - 20) &
                            (all_predictions['now_cost'] <= current_player['now_cost'] + 20) &
                            (all_predictions['predicted_points'] > current_player['predicted_points'])
                        ].nlargest(3, 'predicted_points')
                        
                        for _, alternative in position_alternatives.iterrows():
                            points_gain = alternative['predicted_points'] - current_player['predicted_points']
                            
                            if points_gain > 0.5:  # Meaningful improvement
                                cost_change = (alternative['now_cost'] - current_player['now_cost']) / 10
                                
                                # Add fixture analysis
                                fixture_advantage = ""
                                if include_fixture_analysis and 'team_summaries' in result['fixture_analysis']:
                                    current_team_summary = result['fixture_analysis']['team_summaries'].get(current_player['team'], {})
                                    alternative_team_summary = result['fixture_analysis']['team_summaries'].get(alternative['team'], {})
                                    
                                    current_difficulty = current_team_summary.get('avg_difficulty', 3.0)
                                    alternative_difficulty = alternative_team_summary.get('avg_difficulty', 3.0)
                                    
                                    if alternative_difficulty < current_difficulty - 0.5:
                                        fixture_advantage = f" (Easier fixtures: {alternative_difficulty:.1f} vs {current_difficulty:.1f})"
                                
                                # Add price analysis
                                price_advantage = ""
                                if include_price_analysis and 'price_predictions' in result['price_analysis']:
                                    price_predictions = result['price_analysis']['price_predictions']
                                    current_price_pred = next((p for p in price_predictions if p['id'] == current_player['id']), {})
                                    alternative_price_pred = next((p for p in price_predictions if p['id'] == alternative['id']), {})
                                    
                                    if alternative_price_pred.get('price_rise_probability', 0) > 0.7:
                                        price_advantage = " (Price likely to rise)"
                                    elif current_price_pred.get('price_fall_probability', 0) > 0.7:
                                        price_advantage = " (Current player price likely to fall)"
                                
                                transfer_recommendations.append({
                                    'player_out': {
                                        'id': int(current_player['id']),
                                        'name': current_player['web_name'],
                                        'team': current_player.get('team_name', ''),
                                        'position': current_player['position'],
                                        'cost': current_player['now_cost'] / 10.0,
                                        'predicted_points': round(current_player['predicted_points'], 1),
                                        'uncertainty': round(current_player.get('uncertainty', 2.0), 1),
                                        'risk_category': current_player.get('risk_category', 'Medium Risk')
                                    },
                                    'player_in': {
                                        'id': int(alternative['id']),
                                        'name': alternative['web_name'],
                                        'team': alternative.get('team_name', ''),
                                        'position': alternative['position'],
                                        'cost': alternative['now_cost'] / 10.0,
                                        'predicted_points': round(alternative['predicted_points'], 1),
                                        'uncertainty': round(alternative.get('uncertainty', 2.0), 1),
                                        'risk_category': alternative.get('risk_category', 'Medium Risk')
                                    },
                                    'points_gain': round(points_gain, 1),
                                    'cost_change': round(cost_change, 1),
                                    'reason': f"AI predicts {alternative['predicted_points']:.1f} vs {current_player['predicted_points']:.1f} points{fixture_advantage}{price_advantage}",
                                    'priority': 'High' if points_gain > 2 else 'Medium',
                                    'confidence': 'High' if alternative.get('uncertainty', 2.0) < 1.5 else 'Medium'
                                })
                    
                    # Sort by points gain and limit results
                    transfer_recommendations.sort(key=lambda x: x['points_gain'], reverse=True)
                    result['transfer_recommendations'] = transfer_recommendations[:10]
                    
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
