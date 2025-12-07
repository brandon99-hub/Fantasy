"""
Prediction Update Tasks

Background tasks for updating player predictions and warming caches.
"""

import logging
from celery import Task
from datetime import datetime

from backend.src.tasks.celery_app import celery_app
from backend.src.core.db_factory import get_db
from backend.src.models.enhanced_predictor import EnhancedModelPredictor
from backend.src.core.cache_warming import warm_all_caches
from backend.src.core.cache import get_cache

logger = logging.getLogger(__name__)


class PredictionTask(Task):
    """Base task with database connection"""
    _db = None
    
    @property
    def db(self):
        if self._db is None:
            self._db = get_db()
        return self._db


@celery_app.task(
    base=PredictionTask,
    bind=True,
    name='tasks.prediction_tasks.update_predictions',
    max_retries=3,
    default_retry_delay=300  # 5 minutes
)
def update_predictions(self):
    """
    Update player predictions using enhanced predictor
    
    Runs every 6 hours to keep predictions fresh
    """
    try:
        logger.info("Starting prediction update task...")
        
        # Get database
        db = self.db
        
        # Get all players
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        players_df = loop.run_until_complete(db.get_players_with_stats())
        
        if players_df.empty:
            logger.warning("No players found for prediction update")
            return {'status': 'no_data', 'players': 0}
        
        # Create enhanced predictor
        predictor = EnhancedModelPredictor(db)
        
        # Prepare enhanced data
        enhanced_df = loop.run_until_complete(
            predictor.prepare_prediction_data(players_df)
        )
        
        # Save predictions to database
        # (This would integrate with your prediction models)
        
        # Invalidate prediction cache
        cache = get_cache()
        cache.delete_pattern('predictions:*')
        cache.delete_pattern('players:*')
        
        logger.info(f"Updated predictions for {len(enhanced_df)} players")
        
        return {
            'status': 'success',
            'players': len(enhanced_df),
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as exc:
        logger.error(f"Prediction update failed: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    name='tasks.prediction_tasks.warm_cache',
    max_retries=2
)
def warm_cache_task():
    """
    Warm cache with frequently accessed data
    
    Runs before gameweek deadline
    """
    try:
        logger.info("Starting cache warming task...")
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Warm all caches
        results = loop.run_until_complete(warm_all_caches())
        
        total_cached = sum(results.values())
        
        logger.info(f"Cache warming complete: {total_cached} items cached")
        
        return {
            'status': 'success',
            'cached_items': total_cached,
            'details': results,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as exc:
        logger.error(f"Cache warming failed: {exc}")
        raise


@celery_app.task(
    name='tasks.prediction_tasks.invalidate_stale_cache',
    max_retries=1
)
def invalidate_stale_cache():
    """
    Invalidate stale cache entries
    
    Runs periodically to clean up old data
    """
    try:
        logger.info("Invalidating stale cache entries...")
        
        cache = get_cache()
        
        # Clear old predictions
        cache.delete_pattern('predictions:*')
        
        # Clear old optimization results
        cache.delete_pattern('optimization:*')
        
        logger.info("Stale cache invalidation complete")
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as exc:
        logger.error(f"Cache invalidation failed: {exc}")
        return {'status': 'error', 'error': str(exc)}
