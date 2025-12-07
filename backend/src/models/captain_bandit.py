"""
Contextual Bandit for Captain Selection

Uses multi-armed bandit approach with contextual information to learn
optimal captain choices through exploration and exploitation.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pickle
import os
from datetime import datetime

from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.core.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()


class CaptainBandit:
    """
    Contextual bandit for captain selection
    
    Uses epsilon-greedy strategy with linear regression to predict
    captain points based on context (fixture, form, ownership, etc.)
    """
    
    def __init__(self, epsilon: float = 0.1, learning_rate: float = 0.01):
        """
        Initialize captain bandit
        
        Args:
            epsilon: Exploration rate (0-1). Higher = more exploration
            learning_rate: Learning rate for weight updates
        """
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.weights = {}  # player_id -> weight vector
        self.postgres_db = PostgresManagerDB()
        self.logger = logging.getLogger(__name__)
        self.models_dir = os.path.join(settings.BASE_DIR, "data", "models")
    
    def select_captain(
        self,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
        gameweek: int
    ) -> Tuple[int, float, str]:
        """
        Select captain using epsilon-greedy strategy
        
        Args:
            candidates: List of candidate players with their features
            context: Contextual information (gameweek, fixtures, etc.)
            gameweek: Current gameweek
        
        Returns:
            Tuple of (player_id, expected_points, strategy_used)
        """
        if not candidates:
            raise ValueError("No captain candidates provided")
        
        # Epsilon-greedy: explore or exploit
        if np.random.random() < self.epsilon:
            # Explore: random selection
            selected = np.random.choice(candidates)
            strategy = "explore"
            expected_points = self._predict_captain_points(selected, context)
        else:
            # Exploit: select best based on learned weights
            best_player = None
            best_score = -np.inf
            
            for candidate in candidates:
                score = self._predict_captain_points(candidate, context)
                if score > best_score:
                    best_score = score
                    best_player = candidate
            
            selected = best_player
            expected_points = best_score
            strategy = "exploit"
        
        self.logger.info(
            f"Selected captain {selected['web_name']} (ID: {selected['id']}) "
            f"with expected {expected_points:.2f} points using {strategy} strategy"
        )
        
        return selected['id'], expected_points, strategy
    
    def _predict_captain_points(
        self,
        player: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """
        Predict captain points using learned weights
        
        Args:
            player: Player data
            context: Contextual information
        
        Returns:
            Predicted captain points
        """
        player_id = player['id']
        
        # Get or initialize weights for this player
        if player_id not in self.weights:
            self.weights[player_id] = np.random.randn(10) * 0.01
        
        # Extract features
        features = self._extract_features(player, context)
        
        # Linear prediction
        prediction = np.dot(self.weights[player_id], features)
        
        # Captain gets 2x points, so multiply by 2
        return prediction * 2
    
    def _extract_features(
        self,
        player: Dict[str, Any],
        context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract feature vector from player and context
        
        Features:
        0. Form (recent points per game)
        1. Fixture difficulty (0-5, lower is easier)
        2. Home/Away (1 for home, 0 for away)
        3. Expected goals (xG)
        4. Expected assists (xA)
        5. Ownership percentage
        6. Price (in millions)
        7. Minutes played (recent average)
        8. Opponent defensive strength
        9. Bias term (always 1)
        """
        features = np.zeros(10)
        
        features[0] = float(player.get('form', 0))
        features[1] = 5.0 - float(context.get('fixture_difficulty', 3))  # Invert so easier = higher
        features[2] = 1.0 if context.get('is_home', False) else 0.0
        features[3] = float(player.get('expected_goals', 0))
        features[4] = float(player.get('expected_assists', 0))
        features[5] = float(player.get('selected_by_percent', 0)) / 100.0
        features[6] = float(player.get('now_cost', 0)) / 10.0
        features[7] = float(player.get('minutes', 0)) / 90.0
        features[8] = 5.0 - float(context.get('opponent_strength', 3))
        features[9] = 1.0  # Bias
        
        return features
    
    def update(
        self,
        player_id: int,
        context: Dict[str, Any],
        actual_points: float,
        player_data: Dict[str, Any]
    ):
        """
        Update weights based on actual captain performance
        
        Args:
            player_id: Player who was captained
            context: Context when decision was made
            actual_points: Actual points scored (already 2x for captain)
            player_data: Player data at time of decision
        """
        if player_id not in self.weights:
            self.weights[player_id] = np.random.randn(10) * 0.01
        
        # Extract features
        features = self._extract_features(player_data, context)
        
        # Predicted points
        predicted = np.dot(self.weights[player_id], features)
        
        # Error
        error = actual_points - predicted
        
        # Gradient descent update
        gradient = error * features
        self.weights[player_id] += self.learning_rate * gradient
        
        self.logger.info(
            f"Updated weights for player {player_id}: "
            f"predicted={predicted:.2f}, actual={actual_points:.2f}, error={error:.2f}"
        )
    
    def batch_update_from_history(self, num_gameweeks: int = 10):
        """
        Batch update weights from historical captain data
        
        Args:
            num_gameweeks: Number of recent gameweeks to learn from
        """
        self.logger.info(f"Batch updating from last {num_gameweeks} gameweeks")
        
        try:
            # Get captain data from manager_picks
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            mp.element_id as player_id,
                            mp.event as gameweek,
                            ph.total_points * 2 as captain_points,
                            p.form,
                            p.expected_goals,
                            p.expected_assists,
                            p.selected_by_percent,
                            p.now_cost,
                            ph.minutes,
                            ph.was_home,
                            f.team_h_difficulty,
                            f.team_a_difficulty
                        FROM manager_picks mp
                        JOIN player_history ph ON mp.element_id = ph.player_id AND mp.event = ph.event
                        JOIN players p ON mp.element_id = p.id
                        LEFT JOIN fixtures f ON ph.fixture = f.id
                        WHERE mp.is_captain = TRUE
                          AND mp.event >= (SELECT MAX(id) FROM gameweeks) - %s
                        LIMIT 1000;
                    """, (num_gameweeks,))
                    
                    captain_data = cur.fetchall()
            
            # Update weights for each captain choice
            updates = 0
            for row in captain_data:
                player_data = {
                    'id': row['player_id'],
                    'form': row['form'],
                    'expected_goals': row['expected_goals'],
                    'expected_assists': row['expected_assists'],
                    'selected_by_percent': row['selected_by_percent'],
                    'now_cost': row['now_cost'],
                    'minutes': row['minutes']
                }
                
                context = {
                    'is_home': row['was_home'],
                    'fixture_difficulty': row['team_h_difficulty'] if row['was_home'] else row['team_a_difficulty'],
                    'opponent_strength': 3  # Default
                }
                
                self.update(
                    row['player_id'],
                    context,
                    row['captain_points'],
                    player_data
                )
                updates += 1
            
            self.logger.info(f"Batch update complete: {updates} captain choices processed")
            
        except Exception as e:
            self.logger.error(f"Error in batch update: {str(e)}")
    
    def get_top_captains(
        self,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top N captain recommendations with expected points
        
        Args:
            candidates: List of candidate players
            context: Contextual information
            top_n: Number of recommendations to return
        
        Returns:
            List of top captain recommendations
        """
        recommendations = []
        
        for candidate in candidates:
            expected_points = self._predict_captain_points(candidate, context)
            recommendations.append({
                'player_id': candidate['id'],
                'web_name': candidate.get('web_name', 'Unknown'),
                'team': candidate.get('team_name', 'Unknown'),
                'expected_captain_points': expected_points,
                'ownership': candidate.get('selected_by_percent', 0),
                'form': candidate.get('form', 0)
            })
        
        # Sort by expected points
        recommendations.sort(key=lambda x: x['expected_captain_points'], reverse=True)
        
        return recommendations[:top_n]
    
    def save_model(self, filepath: str = None):
        """Save bandit model to disk"""
        if filepath is None:
            filepath = os.path.join(self.models_dir, "captain_bandit.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'weights': self.weights,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Captain bandit model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load bandit model from disk"""
        if filepath is None:
            filepath = os.path.join(self.models_dir, "captain_bandit.pkl")
        
        if not os.path.exists(filepath):
            self.logger.warning(f"No saved model found at {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.weights = model_data['weights']
            self.epsilon = model_data['epsilon']
            self.learning_rate = model_data['learning_rate']
            
            self.logger.info(f"Captain bandit model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
