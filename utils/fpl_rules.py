from typing import Dict, List, Tuple
import logging

class FPLRules:
    """Encapsulates Fantasy Premier League rules and constraints"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Squad composition rules
        self.SQUAD_SIZE = 15
        self.STARTING_XI_SIZE = 11
        self.BENCH_SIZE = 4
        
        # Position requirements for squad
        self.SQUAD_POSITIONS = {
            'GKP': 2,
            'DEF': 5,
            'MID': 5,
            'FWD': 3
        }
        
        # Formation constraints for starting XI
        self.FORMATION_CONSTRAINTS = {
            'GKP': {'min': 1, 'max': 1},
            'DEF': {'min': 3, 'max': 5},
            'MID': {'min': 2, 'max': 5},
            'FWD': {'min': 1, 'max': 3}
        }
        
        # Financial rules
        self.INITIAL_BUDGET = 100.0  # £100m
        self.MAX_PLAYER_COST = 15.0  # £15m (theoretical max)
        self.MIN_PLAYER_COST = 4.0   # £4m (typical minimum)
        
        # Team selection rules
        self.MAX_PLAYERS_PER_TEAM = 3
        
        # Transfer rules
        self.FREE_TRANSFERS_PER_GW = 1
        self.MAX_FREE_TRANSFERS_SAVED = 2
        self.TRANSFER_HIT_COST = 4  # points
        
        # Captain rules
        self.CAPTAIN_MULTIPLIER = 2
        self.TRIPLE_CAPTAIN_MULTIPLIER = 3
        
        # Chip rules
        self.CHIPS = {
            'wildcard': {'uses_per_season': 2, 'phases': ['first_half', 'second_half']},
            'free_hit': {'uses_per_season': 1},
            'bench_boost': {'uses_per_season': 1},
            'triple_captain': {'uses_per_season': 1}
        }
        
        # Scoring system
        self.SCORING_SYSTEM = {
            'GKP': {
                'minutes_0': 0,
                'minutes_1_59': 1,
                'minutes_60_plus': 2,
                'goals': 6,
                'assists': 3,
                'clean_sheet': 4,
                'penalties_saved': 5,
                'saves_3': 1,  # Every 3 saves = 1 point
                'yellow_card': -1,
                'red_card': -3,
                'own_goal': -2,
                'goals_conceded_2': -1  # Every 2 goals conceded = -1 point
            },
            'DEF': {
                'minutes_0': 0,
                'minutes_1_59': 1,
                'minutes_60_plus': 2,
                'goals': 6,
                'assists': 3,
                'clean_sheet': 4,
                'yellow_card': -1,
                'red_card': -3,
                'own_goal': -2,
                'goals_conceded_2': -1
            },
            'MID': {
                'minutes_0': 0,
                'minutes_1_59': 1,
                'minutes_60_plus': 2,
                'goals': 5,
                'assists': 3,
                'clean_sheet': 1,
                'yellow_card': -1,
                'red_card': -3,
                'own_goal': -2
            },
            'FWD': {
                'minutes_0': 0,
                'minutes_1_59': 1,
                'minutes_60_plus': 2,
                'goals': 4,
                'assists': 3,
                'yellow_card': -1,
                'red_card': -3,
                'own_goal': -2
            }
        }
        
        # Valid formations
        self.VALID_FORMATIONS = [
            '3-4-3', '3-5-2', '4-3-3', '4-4-2', 
            '4-5-1', '5-3-2', '5-4-1'
        ]
    
    def validate_squad_composition(self, players: List[Dict]) -> Dict:
        """Validate that squad meets composition requirements"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            if len(players) != self.SQUAD_SIZE:
                validation['is_valid'] = False
                validation['errors'].append(f"Squad must have exactly {self.SQUAD_SIZE} players, found {len(players)}")
            
            # Check position counts
            position_counts = {}
            for player in players:
                pos = player.get('position', 'UNKNOWN')
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            for pos, required_count in self.SQUAD_POSITIONS.items():
                actual_count = position_counts.get(pos, 0)
                if actual_count != required_count:
                    validation['is_valid'] = False
                    validation['errors'].append(f"{pos}: need {required_count}, found {actual_count}")
            
            # Check team limits
            team_counts = {}
            for player in players:
                team = player.get('team', 'UNKNOWN')
                team_counts[team] = team_counts.get(team, 0) + 1
            
            for team, count in team_counts.items():
                if count > self.MAX_PLAYERS_PER_TEAM:
                    validation['is_valid'] = False
                    validation['errors'].append(f"Too many players from {team}: {count} > {self.MAX_PLAYERS_PER_TEAM}")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating squad composition: {str(e)}")
            validation['is_valid'] = False
            validation['errors'].append(f"Validation error: {str(e)}")
            return validation
    
    def validate_starting_eleven(self, starting_xi: List[Dict]) -> Dict:
        """Validate starting XI formation and composition"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'formation': None
        }
        
        try:
            if len(starting_xi) != self.STARTING_XI_SIZE:
                validation['is_valid'] = False
                validation['errors'].append(f"Starting XI must have exactly {self.STARTING_XI_SIZE} players")
                return validation
            
            # Count positions
            position_counts = {}
            for player in starting_xi:
                pos = player.get('position', 'UNKNOWN')
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # Validate position constraints
            for pos, constraints in self.FORMATION_CONSTRAINTS.items():
                count = position_counts.get(pos, 0)
                if count < constraints['min']:
                    validation['is_valid'] = False
                    validation['errors'].append(f"{pos}: minimum {constraints['min']}, found {count}")
                elif count > constraints['max']:
                    validation['is_valid'] = False
                    validation['errors'].append(f"{pos}: maximum {constraints['max']}, found {count}")
            
            # Determine formation
            if validation['is_valid']:
                def_count = position_counts.get('DEF', 0)
                mid_count = position_counts.get('MID', 0)
                fwd_count = position_counts.get('FWD', 0)
                validation['formation'] = f"{def_count}-{mid_count}-{fwd_count}"
                
                if validation['formation'] not in self.VALID_FORMATIONS:
                    validation['warnings'].append(f"Formation {validation['formation']} is unusual")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating starting XI: {str(e)}")
            validation['is_valid'] = False
            validation['errors'].append(f"Validation error: {str(e)}")
            return validation
    
    def validate_budget(self, players: List[Dict], budget: float = None) -> Dict:
        """Validate squad cost against budget"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'total_cost': 0,
            'budget_remaining': 0
        }
        
        try:
            if budget is None:
                budget = self.INITIAL_BUDGET
            
            total_cost = sum(player.get('cost', 0) for player in players)
            validation['total_cost'] = total_cost
            validation['budget_remaining'] = budget - total_cost
            
            if total_cost > budget:
                validation['is_valid'] = False
                validation['errors'].append(f"Over budget: £{total_cost:.1f}M > £{budget:.1f}M")
            
            if validation['budget_remaining'] < 0.5:
                validation['warnings'].append(f"Very little budget remaining: £{validation['budget_remaining']:.1f}M")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating budget: {str(e)}")
            validation['is_valid'] = False
            validation['errors'].append(f"Budget validation error: {str(e)}")
            return validation
    
    def validate_transfers(self, 
                          transfers_out: List[int], 
                          transfers_in: List[int], 
                          free_transfers: int = 1,
                          use_wildcard: bool = False) -> Dict:
        """Validate transfer rules"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'transfer_cost': 0,
            'num_transfers': 0
        }
        
        try:
            if use_wildcard:
                validation['transfer_cost'] = 0
                validation['num_transfers'] = len(transfers_out)
                return validation
            
            num_transfers = len(transfers_out)
            validation['num_transfers'] = num_transfers
            
            if num_transfers != len(transfers_in):
                validation['is_valid'] = False
                validation['errors'].append("Number of transfers in must equal transfers out")
            
            if num_transfers > free_transfers:
                hits = num_transfers - free_transfers
                validation['transfer_cost'] = hits * self.TRANSFER_HIT_COST
                validation['warnings'].append(f"Taking {hits} hits (-{validation['transfer_cost']} points)")
            
            if num_transfers > 15:  # Theoretical maximum
                validation['is_valid'] = False
                validation['errors'].append("Too many transfers in one gameweek")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating transfers: {str(e)}")
            validation['is_valid'] = False
            validation['errors'].append(f"Transfer validation error: {str(e)}")
            return validation
    
    def calculate_points(self, player_stats: Dict, position: str) -> int:
        """Calculate FPL points for a player based on their stats"""
        try:
            if position not in self.SCORING_SYSTEM:
                return 0
            
            scoring = self.SCORING_SYSTEM[position]
            points = 0
            
            # Minutes points
            minutes = player_stats.get('minutes', 0)
            if minutes == 0:
                points += scoring['minutes_0']
            elif minutes < 60:
                points += scoring['minutes_1_59']
            else:
                points += scoring['minutes_60_plus']
            
            # Goals and assists
            points += player_stats.get('goals_scored', 0) * scoring.get('goals', 0)
            points += player_stats.get('assists', 0) * scoring.get('assists', 0)
            
            # Clean sheets
            if player_stats.get('clean_sheets', 0) > 0:
                points += scoring.get('clean_sheet', 0)
            
            # Cards
            points += player_stats.get('yellow_cards', 0) * scoring.get('yellow_card', 0)
            points += player_stats.get('red_cards', 0) * scoring.get('red_card', 0)
            
            # Own goals
            points += player_stats.get('own_goals', 0) * scoring.get('own_goal', 0)
            
            # Position-specific scoring
            if position == 'GKP':
                # Penalty saves
                points += player_stats.get('penalties_saved', 0) * scoring.get('penalties_saved', 0)
                
                # Save points (every 3 saves)
                saves = player_stats.get('saves', 0)
                points += (saves // 3) * scoring.get('saves_3', 0)
                
                # Goals conceded (every 2)
                goals_conceded = player_stats.get('goals_conceded', 0)
                points += (goals_conceded // 2) * scoring.get('goals_conceded_2', 0)
            
            elif position == 'DEF':
                # Goals conceded (every 2)
                goals_conceded = player_stats.get('goals_conceded', 0)
                points += (goals_conceded // 2) * scoring.get('goals_conceded_2', 0)
            
            # Bonus points
            points += player_stats.get('bonus', 0)
            
            return max(0, points)  # Minimum 0 points
            
        except Exception as e:
            self.logger.error(f"Error calculating points: {str(e)}")
            return 0
    
    def get_formation_constraints(self, formation: str) -> Dict:
        """Get specific constraints for a formation"""
        formation_map = {
            '3-4-3': {'DEF': 3, 'MID': 4, 'FWD': 3},
            '3-5-2': {'DEF': 3, 'MID': 5, 'FWD': 2},
            '4-3-3': {'DEF': 4, 'MID': 3, 'FWD': 3},
            '4-4-2': {'DEF': 4, 'MID': 4, 'FWD': 2},
            '4-5-1': {'DEF': 4, 'MID': 5, 'FWD': 1},
            '5-3-2': {'DEF': 5, 'MID': 3, 'FWD': 2},
            '5-4-1': {'DEF': 5, 'MID': 4, 'FWD': 1}
        }
        
        constraints = formation_map.get(formation, formation_map['3-4-3'])
        constraints['GKP'] = 1  # Always 1 goalkeeper
        
        return constraints
    
    def is_valid_captain_choice(self, player: Dict, starting_xi: List[Dict]) -> bool:
        """Check if player can be captain"""
        try:
            player_id = player.get('id')
            starting_ids = [p.get('id') for p in starting_xi]
            
            return player_id in starting_ids
            
        except Exception as e:
            self.logger.error(f"Error checking captain validity: {str(e)}")
            return False
