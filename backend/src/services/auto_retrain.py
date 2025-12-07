"""
Auto-Retraining Service

Automatically retrains ML models when new gameweek data is available
or when prediction accuracy falls below threshold.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os

from backend.src.models.ensemble_predictor import EnsemblePredictor
from backend.src.models.minutes_model import MinutesPredictor
from backend.src.services.prediction_evaluator import PredictionEvaluator
from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.core.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()


class AutoRetrainer:
    """Automatically retrain models based on triggers"""
    
    def __init__(self):
        self.evaluator = PredictionEvaluator()
        self.postgres_db = PostgresManagerDB()
        self.logger = logging.getLogger(__name__)
        self.models_dir = os.path.join(settings.BASE_DIR, "data", "models")
    
    def should_retrain(self) -> Dict[str, Any]:
        """
        Determine if models should be retrained
        
        Returns:
            Dictionary with decision and reasoning
        """
        reasons = []
        should_retrain = False
        
        try:
            # Check 1: New gameweek data available
            if self._has_new_gameweek_data():
                reasons.append("New gameweek data available")
                should_retrain = True
            
            # Check 2: Prediction accuracy below threshold
            if self.evaluator.should_retrain_model(threshold_mae=2.5):
                reasons.append("Prediction accuracy below threshold")
                should_retrain = True
            
            # Check 3: Model is old (>7 days since last training)
            if self._is_model_stale(days=7):
                reasons.append("Model is stale (>7 days old)")
                should_retrain = True
            
            return {
                'should_retrain': should_retrain,
                'reasons': reasons,
                'checked_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error checking retrain status: {str(e)}")
            return {
                'should_retrain': False,
                'reasons': [f"Error: {str(e)}"],
                'checked_at': datetime.now().isoformat()
            }
    
    def _has_new_gameweek_data(self) -> bool:
        """Check if there's new gameweek data since last training"""
        try:
            # Get latest gameweek with data
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT MAX(event) as latest_gw
                        FROM player_history;
                    """)
                    result = cur.fetchone()
                    latest_gw = result['latest_gw'] if result else None
            
            if not latest_gw:
                return False
            
            # Get last training gameweek from model_metrics
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT MAX(train_end_gw) as last_trained_gw
                        FROM model_metrics
                        WHERE model_name = 'ensemble';
                    """)
                    result = cur.fetchone()
                    last_trained_gw = result['last_trained_gw'] if result else 0
            
            return latest_gw > last_trained_gw
            
        except Exception as e:
            self.logger.error(f"Error checking new gameweek data: {str(e)}")
            return False
    
    def _is_model_stale(self, days: int = 7) -> bool:
        """Check if model hasn't been trained in N days"""
        try:
            with self.postgres_db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT MAX(created_at) as last_training
                        FROM model_metrics
                        WHERE model_name = 'ensemble';
                    """)
                    result = cur.fetchone()
                    
                    if not result or not result['last_training']:
                        return True
                    
                    last_training = result['last_training']
                    days_since = (datetime.now() - last_training).days
                    
                    return days_since > days
                    
        except Exception as e:
            self.logger.error(f"Error checking model staleness: {str(e)}")
            return False
    
    def retrain_models(self, force: bool = False) -> Dict[str, Any]:
        """
        Retrain all models
        
        Args:
            force: Force retraining even if not needed
        
        Returns:
            Dictionary with retraining results
        """
        self.logger.info("Starting model retraining process")
        
        try:
            # Check if retraining needed
            if not force:
                check = self.should_retrain()
                if not check['should_retrain']:
                    self.logger.info("Retraining not needed")
                    return {
                        'success': False,
                        'message': 'Retraining not needed',
                        'reasons': check['reasons']
                    }
            
            results = {
                'success': True,
                'models_trained': [],
                'errors': [],
                'started_at': datetime.now().isoformat()
            }
            
            # Get current gameweek for versioning
            current_gw = self._get_current_gameweek()
            version = f"v{datetime.now().strftime('%Y%m%d')}_{current_gw}"
            
            # Retrain minutes model
            self.logger.info("Retraining minutes model...")
            try:
                minutes_model = MinutesPredictor()
                minutes_metrics = minutes_model.train(retrain=True)
                
                # Save model
                minutes_path = os.path.join(self.models_dir, f"minutes_{version}.pkl")
                minutes_model.save_model(minutes_path)
                
                results['models_trained'].append({
                    'model': 'minutes',
                    'version': version,
                    'metrics': minutes_metrics,
                    'path': minutes_path
                })
                
                self.logger.info(f"Minutes model trained: {minutes_metrics}")
                
            except Exception as e:
                error_msg = f"Minutes model training failed: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
            
            # Retrain ensemble model
            self.logger.info("Retraining ensemble model...")
            try:
                ensemble_model = EnsemblePredictor()
                ensemble_metrics = ensemble_model.train(retrain=True)
                
                # Save model
                ensemble_path = os.path.join(self.models_dir, f"ensemble_{version}.pkl")
                ensemble_model.save_model(ensemble_path)
                
                results['models_trained'].append({
                    'model': 'ensemble',
                    'version': version,
                    'metrics': ensemble_metrics,
                    'path': ensemble_path
                })
                
                self.logger.info(f"Ensemble model trained: {ensemble_metrics}")
                
                # Store feature importance
                self._store_feature_importance(ensemble_model, version, current_gw)
                
            except Exception as e:
                error_msg = f"Ensemble model training failed: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
            
            results['completed_at'] = datetime.now().isoformat()
            results['success'] = len(results['models_trained']) > 0
            
            if results['success']:
                self.logger.info(f"Retraining completed successfully: {len(results['models_trained'])} models trained")
            else:
                self.logger.error("Retraining failed for all models")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during retraining: {str(e)}")
            return {
                'success': False,
                'message': f'Retraining error: {str(e)}',
                'errors': [str(e)]
            }
    
    def _get_current_gameweek(self) -> int:
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
                    return result['id'] if result else 1
        except Exception as e:
            self.logger.error(f"Error getting current gameweek: {str(e)}")
            return 1
    
    def _store_feature_importance(self, model: EnsemblePredictor, version: str, gameweek: int):
        """Store feature importance from trained model"""
        try:
            importance_data = model.get_model_importance()
            
            if not importance_data:
                self.logger.warning("No feature importance data available")
                return
            
            # Convert to database format
            records = []
            for model_name, features in importance_data.items():
                for rank, (feature_name, importance_score) in enumerate(features.items(), 1):
                    records.append({
                        'model_name': model_name,
                        'model_version': version,
                        'gameweek': gameweek,
                        'feature_name': feature_name,
                        'importance_score': float(importance_score),
                        'rank': rank,
                        'feature_type': self._classify_feature_type(feature_name)
                    })
            
            # Store in database
            self.postgres_db.store_feature_importance(records)
            self.logger.info(f"Stored feature importance for {len(records)} features")
            
        except Exception as e:
            self.logger.error(f"Error storing feature importance: {str(e)}")
    
    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify feature into type category"""
        feature_name_lower = feature_name.lower()
        
        if 'rolling' in feature_name_lower or 'ewm' in feature_name_lower:
            return 'rolling_average'
        elif 'xg' in feature_name_lower or 'xa' in feature_name_lower:
            return 'expected_stats'
        elif 'form' in feature_name_lower or 'recent' in feature_name_lower:
            return 'form'
        elif 'price' in feature_name_lower or 'cost' in feature_name_lower:
            return 'price'
        elif 'fixture' in feature_name_lower or 'opponent' in feature_name_lower:
            return 'fixture'
        else:
            return 'other'
    
    def evaluate_and_retrain(self) -> Dict[str, Any]:
        """
        Complete workflow: evaluate recent predictions and retrain if needed
        
        Returns:
            Dictionary with evaluation and retraining results
        """
        self.logger.info("Starting evaluate and retrain workflow")
        
        results = {
            'evaluation': None,
            'retraining': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Evaluate recent predictions
            self.logger.info("Evaluating recent predictions...")
            evaluation_results = self.evaluator.evaluate_recent_gameweeks(num_gameweeks=3)
            results['evaluation'] = {
                'gameweeks_evaluated': len(evaluation_results),
                'results': evaluation_results
            }
            
            # Step 2: Check if retraining needed
            retrain_check = self.should_retrain()
            results['retrain_check'] = retrain_check
            
            # Step 3: Retrain if needed
            if retrain_check['should_retrain']:
                self.logger.info("Retraining triggered")
                retraining_results = self.retrain_models()
                results['retraining'] = retraining_results
            else:
                self.logger.info("Retraining not needed")
                results['retraining'] = {'skipped': True, 'reason': 'Not needed'}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in evaluate and retrain workflow: {str(e)}")
            results['error'] = str(e)
            return results
