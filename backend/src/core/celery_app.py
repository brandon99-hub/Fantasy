"""
Celery application configuration for background tasks
"""

from celery import Celery
from celery.schedules import crontab
import logging

from backend.src.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    'fpl_optimizer',
    broker=getattr(settings, 'CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=getattr(settings, 'CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Celery Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # Train models weekly (Sunday at 2 AM)
    'train-models-weekly': {
        'task': 'backend.src.tasks.training_tasks.train_all_models',
        'schedule': crontab(day_of_week=0, hour=2, minute=0),
    },
    # Refresh FPL data daily (every day at 6 AM)
    'refresh-data-daily': {
        'task': 'backend.src.tasks.data_tasks.refresh_fpl_data',
        'schedule': crontab(hour=6, minute=0),
    },
    # Check price changes hourly
    'check-price-changes': {
        'task': 'backend.src.tasks.data_tasks.check_price_changes',
        'schedule': crontab(minute=0),  # Every hour
    },
    # Update fixture odds every 6 hours
    'update-fixture-odds': {
        'task': 'backend.src.tasks.data_tasks.update_fixture_odds',
        'schedule': crontab(minute=0, hour='*/6'),
    },
    # Adaptive Learning: Evaluate predictions after gameweek (Tuesdays at 3 AM)
    'evaluate-predictions': {
        'task': 'backend.src.tasks.adaptive_tasks.evaluate_predictions_task',
        'schedule': crontab(day_of_week=2, hour=3, minute=0),
    },
    # Adaptive Learning: Check and auto-retrain models (Tuesdays at 4 AM)
    'auto-retrain-models': {
        'task': 'backend.src.tasks.adaptive_tasks.auto_retrain_task',
        'schedule': crontab(day_of_week=2, hour=4, minute=0),
    },
}

logger.info("Celery app configured successfully")
