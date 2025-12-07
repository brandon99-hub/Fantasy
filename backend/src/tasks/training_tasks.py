"""Background tasks for model training"""

from celery import Task
import logging
from datetime import datetime
from typing import Dict, Any

from backend.src.core.celery_app import celery_app
from backend.src.core.database import FPLDatabase
from backend.src.core.data_collector import FPLDataCollector
from backend.src.models.ensemble_predictor import EnsemblePredictor
from backend.src.models.minutes_model import MinutesPredictor
from backend.src.models.points_model import PointsPredictor
from backend.src.core.cache import get_cache

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task with callbacks"""
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"✅ Task {self.name} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"❌ Task {self.name} failed: {exc}")


@celery_app.task(base=CallbackTask, bind=True)
def train_all_models(self) -> Dict[str, Any]:
    """
    Train all ML models (ensemble, minutes, points)
    Runs weekly via Celery Beat
    """
    logger.info("🚀 Starting automated model training...")
    
    results = {
        'started_at': datetime.now().isoformat(),
        'models_trained': [],
        'errors': []
    }
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Training ensemble model'})
        
        # Train ensemble predictor
        try:
            ensemble = EnsemblePredictor()
            ensemble.train(retrain=True)
            results['models_trained'].append('ensemble')
            logger.info("✅ Ensemble model trained")
        except Exception as e:
            logger.error(f"❌ Ensemble training failed: {e}")
            results['errors'].append(f"Ensemble: {str(e)}")
        
        # Train minutes predictor
        self.update_state(state='PROGRESS', meta={'status': 'Training minutes model'})
        try:
            minutes = MinutesPredictor()
            minutes.train(retrain=True)
            results['models_trained'].append('minutes')
            logger.info("✅ Minutes model trained")
        except Exception as e:
            logger.error(f"❌ Minutes training failed: {e}")
            results['errors'].append(f"Minutes: {str(e)}")
        
        # Train points predictor
        self.update_state(state='PROGRESS', meta={'status': 'Training points model'})
        try:
            points = PointsPredictor()
            points.train(retrain=True)
            results['models_trained'].append('points')
            logger.info("✅ Points model trained")
        except Exception as e:
            logger.error(f"❌ Points training failed: {e}")
            results['errors'].append(f"Points: {str(e)}")
        
        # Clear prediction caches
        self.update_state(state='PROGRESS', meta={'status': 'Clearing caches'})
        cache = get_cache()
        cache.clear_pattern('predictions:*')
        cache.clear_pattern('analysis:*')
        logger.info("🗑️  Cleared prediction caches")
        
        results['completed_at'] = datetime.now().isoformat()
        results['success'] = len(results['errors']) == 0
        
        logger.info(f"✅ Model training completed. Trained: {results['models_trained']}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        results['errors'].append(str(e))
        results['success'] = False
        return results


@celery_app.task(base=CallbackTask)
def train_single_model(model_name: str) -> Dict[str, Any]:
    """
    Train a single model on demand
    
    Args:
        model_name: 'ensemble', 'minutes', or 'points'
    """
    logger.info(f"🚀 Training {model_name} model...")
    
    try:
        if model_name == 'ensemble':
            model = EnsemblePredictor()
        elif model_name == 'minutes':
            model = MinutesPredictor()
        elif model_name == 'points':
            model = PointsPredictor()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.train(retrain=True)
        
        # Clear related caches
        cache = get_cache()
        cache.clear_pattern(f'{model_name}:*')
        cache.clear_pattern('predictions:*')
        
        logger.info(f"✅ {model_name} model trained successfully")
        return {'success': True, 'model': model_name}
        
    except Exception as e:
        logger.error(f"❌ {model_name} training failed: {e}")
        return {'success': False, 'model': model_name, 'error': str(e)}
