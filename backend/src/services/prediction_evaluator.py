"""
Prediction Evaluator Service

Evaluates prediction accuracy after each gameweek by comparing predictions
from the predictions table against actual points from player_history.
Stores results in prediction_feedback table for continuous learning.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.core.database import FPLDatabase


logger = logging.getLogger(__name__)


class PredictionEvaluator:
    """Evaluate and track prediction accuracy for continuous learning"""
    
    def __init__(self):
        self.db = FPLDatabase()
        self.postgres_db = PostgresManagerDB()
        self.logger = logging.getLogger(__name__)
    
    def evaluate_gameweek(self, gameweek: int, model_version: str = "ensemble_v1") -> Dict[str, Any]:
        """
        Evaluate prediction accuracy for a specific gameweek
        
        Args:
            gameweek: Gameweek number to evaluate
            model_version: Model version identifier
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating predictions for GW {gameweek}")
        
        try:
            # Get predictions for this gameweek
            predictions = self._get_predictions(gameweek)
            if predictions.empty:
                self.logger.warning(f"No predictions found for GW {gameweek}")
                return {"error": "No predictions found"}
            
            # Get actual results from player_history
            actuals = self._get_actual_results(gameweek)
            if actuals.empty:
                self.logger.warning(f"No actual results found for GW {gameweek}")
                return {"error": "No actual results found"}
            
            # Merge predictions with actuals
            merged = predictions.merge(
                actuals,
                left_on='player_id',
                right_on='player_id',
                how='inner'
            )
            
            if merged.empty:
                self.logger.warning(f"No matching data for GW {gameweek}")
                return {"error": "No matching data"}
            
            # Calculate errors
            feedback_records = []
            for _, row in merged.iterrows():
                predicted = float(row['predicted_points'])
                actual = int(row['total_points'])
                error = predicted - actual
                
                feedback_records.append({
                    'player_id': int(row['player_id']),
                    'gameweek': gameweek,
                    'predicted_points': predicted,
                    'actual_points': actual,
                    'prediction_error': error,
                    'absolute_error': abs(error),
                    'squared_error': error ** 2,
                    'model_version': model_version,
                    'player_position': int(row.get('element_type', 0)),
                    'player_price': int(row.get('now_cost', 0)),
                    'player_team': int(row.get('team_id', 0)),
                    'was_home': bool(row.get('was_home', False))
                })
            
            # Store feedback in database
            success = self.postgres_db.store_prediction_feedback(feedback_records)
            
            if not success:
                self.logger.error(f"Failed to store prediction feedback for GW {gameweek}")
                return {"error": "Failed to store feedback"}
            
            # Calculate aggregate metrics
            metrics = self._calculate_metrics(feedback_records)
            metrics['gameweek'] = gameweek
            metrics['model_version'] = model_version
            metrics['total_predictions'] = len(feedback_records)
            
            self.logger.info(f"GW {gameweek} evaluation complete: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating GW {gameweek}: {str(e)}")
            return {"error": str(e)}
    
    def _get_predictions(self, gameweek: int):
        """Get predictions from database for a gameweek"""
        import pandas as pd
        
        try:
            with self.postgres_db.get_connection() as conn:
                query = """
                    SELECT 
                        p.player_id,
                        p.predicted_points,
                        pl.element_type,
                        pl.now_cost,
                        pl.team_id
                    FROM predictions p
                    JOIN players pl ON p.player_id = pl.id
                    WHERE p.gameweek = %s;
                """
                return pd.read_sql(query, conn, params=(gameweek,))
        except Exception as e:
            self.logger.error(f"Error getting predictions: {str(e)}")
            return pd.DataFrame()
    
    def _get_actual_results(self, gameweek: int):
        """Get actual results from player_history for a gameweek"""
        import pandas as pd
        
        try:
            with self.postgres_db.get_connection() as conn:
                query = """
                    SELECT 
                        player_id,
                        total_points,
                        was_home,
                        minutes
                    FROM player_history
                    WHERE event = %s;
                """
                return pd.read_sql(query, conn, params=(gameweek,))
        except Exception as e:
            self.logger.error(f"Error getting actual results: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_metrics(self, feedback_records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate prediction metrics"""
        if not feedback_records:
            return {}
        
        errors = [r['prediction_error'] for r in feedback_records]
        abs_errors = [r['absolute_error'] for r in feedback_records]
        sq_errors = [r['squared_error'] for r in feedback_records]
        
        return {
            'mae': np.mean(abs_errors),
            'rmse': np.sqrt(np.mean(sq_errors)),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': max(abs_errors),
            'min_error': min(abs_errors),
            'median_error': np.median(abs_errors)
        }
    
    def evaluate_recent_gameweeks(self, num_gameweeks: int = 5) -> List[Dict[str, Any]]:
        """Evaluate multiple recent gameweeks"""
        results = []
        
        try:
            # Get current gameweek
            current_gw = self._get_current_gameweek()
            if not current_gw:
                return results
            
            # Evaluate last N finished gameweeks
            for gw in range(max(1, current_gw - num_gameweeks), current_gw):
                if self._is_gameweek_finished(gw):
                    result = self.evaluate_gameweek(gw)
                    if 'error' not in result:
                        results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating recent gameweeks: {str(e)}")
            return results
    
    def _get_current_gameweek(self) -> Optional[int]:
        """Get current gameweek number"""
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id FROM gameweeks 
                        WHERE is_current = TRUE 
                        LIMIT 1;
                    """)
                    result = cur.fetchone()
                    return result['id'] if result else None
        except Exception as e:
            self.logger.error(f"Error getting current gameweek: {str(e)}")
            return None
    
    def _is_gameweek_finished(self, gameweek: int) -> bool:
        """Check if a gameweek is finished"""
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT finished FROM gameweeks 
                        WHERE id = %s;
                    """, (gameweek,))
                    result = cur.fetchone()
                    return result['finished'] if result else False
        except Exception as e:
            self.logger.error(f"Error checking gameweek status: {str(e)}")
            return False
    
    def get_accuracy_by_position(self, gameweek: int = None) -> List[Dict[str, Any]]:
        """Get prediction accuracy broken down by position"""
        return self.postgres_db.get_prediction_accuracy_by_position(gameweek)
    
    def get_overall_accuracy(self, gameweek: int = None) -> Dict[str, Any]:
        """Get overall prediction accuracy metrics"""
        return self.postgres_db.get_prediction_accuracy(gameweek)
    
    def should_retrain_model(self, threshold_mae: float = 2.5) -> bool:
        """
        Determine if model should be retrained based on recent accuracy
        
        Args:
            threshold_mae: MAE threshold above which retraining is triggered
        
        Returns:
            True if retraining recommended
        """
        try:
            # Get accuracy for last 3 gameweeks
            accuracy = self.postgres_db.get_prediction_accuracy()
            
            if not accuracy or 'mae' not in accuracy:
                self.logger.warning("No accuracy data available")
                return False
            
            mae = float(accuracy['mae'])
            
            if mae > threshold_mae:
                self.logger.info(f"Retraining recommended: MAE {mae:.2f} > threshold {threshold_mae}")
                return True
            
            self.logger.info(f"Retraining not needed: MAE {mae:.2f} <= threshold {threshold_mae}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking retrain status: {str(e)}")
            return False
    
    def identify_problem_areas(self, gameweek: int = None) -> Dict[str, Any]:
        """
        Identify areas where predictions are struggling
        
        Returns:
            Dictionary with problem areas and recommendations
        """
        try:
            # Get accuracy by position
            position_accuracy = self.get_accuracy_by_position(gameweek)
            
            problems = {
                'high_error_positions': [],
                'recommendations': []
            }
            
            for pos_data in position_accuracy:
                position = pos_data['player_position']
                mae = float(pos_data['mae'])
                
                # Flag positions with high error
                if mae > 3.0:
                    position_names = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                    problems['high_error_positions'].append({
                        'position': position_names.get(position, 'Unknown'),
                        'mae': mae,
                        'count': pos_data['count']
                    })
            
            # Generate recommendations
            if problems['high_error_positions']:
                problems['recommendations'].append(
                    "Consider adding position-specific features or adjusting feature weights"
                )
            
            return problems
            
        except Exception as e:
            self.logger.error(f"Error identifying problem areas: {str(e)}")
            return {}
