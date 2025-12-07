"""
Task Scheduling Configuration

Celery Beat schedule for periodic tasks.
"""

from celery.schedules import crontab
from backend.src.tasks.celery_app import celery_app


# Celery Beat schedule
celery_app.conf.beat_schedule = {
    # High priority - frequent updates
    'refresh-fpl-data-hourly': {
        'task': 'tasks.data_tasks.refresh_fpl_data',
        'schedule': crontab(minute=0),  # Every hour on the hour
        'options': {'queue': 'high_priority'}
    },
    
    # Medium priority - regular maintenance
    'update-predictions-6h': {
        'task': 'tasks.prediction_tasks.update_predictions',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
        'options': {'queue': 'high_priority'}
    },
    
    'update-xg-data-daily': {
        'task': 'tasks.data_tasks.update_xg_data',
        'schedule': crontab(minute=0, hour=2),  # 2 AM daily
        'options': {'queue': 'medium_priority'}
    },
    
    'warm-cache-before-deadline': {
        'task': 'tasks.prediction_tasks.warm_cache',
        'schedule': crontab(minute=0, hour=17, day_of_week=5),  # Friday 5 PM
        'options': {'queue': 'medium_priority'}
    },
    
    # Low priority - heavy operations
    'train-models-weekly': {
        'task': 'tasks.training_tasks.train_models',
        'schedule': crontab(minute=0, hour=3, day_of_week=1),  # Monday 3 AM
        'options': {'queue': 'low_priority'}
    },
    
    # Cleanup tasks
    'invalidate-stale-cache-daily': {
        'task': 'tasks.prediction_tasks.invalidate_stale_cache',
        'schedule': crontab(minute=30, hour=1),  # 1:30 AM daily
        'options': {'queue': 'default'}
    },
    
    'cleanup-old-data-weekly': {
        'task': 'tasks.data_tasks.cleanup_old_data',
        'schedule': crontab(minute=0, hour=4, day_of_week=0),  # Sunday 4 AM
        'options': {'queue': 'low_priority'}
    },
}
