"""
Data Refresh Tasks

Background tasks for refreshing FPL data and xG data.
"""

import logging
from celery import Task
from datetime import datetime

from backend.src.tasks.celery_app import celery_app
from backend.src.core.data_collector import FPLDataCollector
from backend.src.core.db_factory import get_db
from backend.src.integrations.xg_integrator import XGIntegrator
from backend.src.core.cache import get_cache

logger = logging.getLogger(__name__)


@celery_app.task(
    name='tasks.data_tasks.refresh_fpl_data',
    bind=True,
    max_retries=3,
    default_retry_delay=600  # 10 minutes
)
def refresh_fpl_data(self):
    """
    Refresh FPL data from official API
    
    Runs every hour to keep data current
    """
    try:
        logger.info("Starting FPL data refresh...")
        
        collector = FPLDataCollector()
        success = collector.update_all_data()
        
        if not success:
            raise Exception("Data collection failed")
        
        # Invalidate relevant caches
        cache = get_cache()
        cache.clear_pattern('players:*')
        cache.clear_pattern('teams:*')
        cache.clear_pattern('fixtures:*')
        cache.clear_pattern('gameweeks:*')
        
        logger.info("FPL data refresh complete")
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as exc:
        logger.error(f"FPL data refresh failed: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    name='tasks.data_tasks.update_xg_data',
    bind=True,
    max_retries=2,
    default_retry_delay=1800  # 30 minutes
)
def update_xg_data(self):
    """
    Update xG/xA data from FBRef
    
    Runs daily to update expected goals data
    """
    try:
        logger.info("Starting xG data update...")
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get database
        db = get_db()
        
        # Get players
        players_df = loop.run_until_complete(db.get_players_with_stats())
        
        if players_df.empty:
            logger.warning("No players found for xG update")
            return {'status': 'no_data'}
        
        # Update xG data
        integrator = XGIntegrator(db)
        enriched_df = loop.run_until_complete(
            integrator.enrich_player_data(players_df, force_refresh=True)
        )
        
        # Count xG sources
        source_counts = enriched_df['xg_source'].value_counts().to_dict()
        
        logger.info(f"xG data updated: {source_counts}")
        
        # Invalidate player cache
        cache = get_cache()
        cache.delete_pattern('players:*')
        
        return {
            'status': 'success',
            'players': len(enriched_df),
            'sources': source_counts,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as exc:
        logger.error(f"xG data update failed: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    name='tasks.data_tasks.cleanup_old_data',
    max_retries=1
)
def cleanup_old_data():
    """
    Clean up old historical data
    
    Runs weekly to remove stale data
    """
    try:
        logger.info("Starting data cleanup...")
        
        # This would clean up old gameweek data, predictions, etc.
        # Implementation depends on your retention policy
        
        logger.info("Data cleanup complete")
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as exc:
        logger.error(f"Data cleanup failed: {exc}")
        return {'status': 'error', 'error': str(exc)}
