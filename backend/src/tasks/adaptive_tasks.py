"""
Celery tasks for adaptive learning system
"""

import logging
from celery import shared_task

from backend.src.services.prediction_evaluator import PredictionEvaluator
from backend.src.services.auto_retrain import AutoRetrainer


logger = logging.getLogger(__name__)


@shared_task(name='backend.src.tasks.adaptive_tasks.evaluate_predictions_task')
def evaluate_predictions_task():
    """
    Celery task to evaluate predictions after gameweek completion
    Runs on Tuesdays at 3 AM (after gameweek finishes)
    """
    logger.info("Starting prediction evaluation task")
    
    try:
        evaluator = PredictionEvaluator()
        
        # Evaluate last 3 gameweeks
        results = evaluator.evaluate_recent_gameweeks(num_gameweeks=3)
        
        logger.info(f"Prediction evaluation complete: {len(results)} gameweeks evaluated")
        
        return {
            'success': True,
            'gameweeks_evaluated': len(results),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Prediction evaluation task failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@shared_task(name='backend.src.tasks.adaptive_tasks.auto_retrain_task')
def auto_retrain_task():
    """
    Celery task to check if retraining is needed and retrain models
    Runs on Tuesdays at 4 AM (after prediction evaluation)
    """
    logger.info("Starting auto-retrain task")
    
    try:
        retrainer = AutoRetrainer()
        
        # Check if retraining needed and execute
        results = retrainer.evaluate_and_retrain()
        
        if results.get('retraining', {}).get('success'):
            logger.info("Auto-retrain task completed successfully")
        else:
            logger.info("Auto-retrain task completed (no retraining needed)")
        
        return results
        
    except Exception as e:
        logger.error(f"Auto-retrain task failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@shared_task(name='backend.src.tasks.adaptive_tasks.force_retrain_task')
def force_retrain_task():
    """
    Celery task to force model retraining (manual trigger)
    """
    logger.info("Starting forced retrain task")
    
    try:
        retrainer = AutoRetrainer()
        results = retrainer.retrain_models(force=True)
        
        logger.info(f"Forced retrain complete: {results.get('success', False)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Forced retrain task failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
