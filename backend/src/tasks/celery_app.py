"""
Celery Application Configuration

Configures Celery for background task processing with:
- Redis broker and result backend
- Task routing and priorities
- Retry logic and time limits
- JSON serialization
"""

import os
from celery import Celery
from kombu import Queue

from backend.src.core.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    'fpl_optimizer',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,
    
    # Time limits
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minutes soft limit
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_disable_rate_limits=False,
    
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    result_persistent=True,
    
    # Task result settings
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
    
    # Retry settings
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,
)

# Task routing - different queues for different priorities
celery_app.conf.task_routes = {
    # High priority - user-facing operations
    'tasks.prediction_tasks.update_predictions': {'queue': 'high_priority'},
    'tasks.data_tasks.refresh_fpl_data': {'queue': 'high_priority'},
    
    # Medium priority - scheduled maintenance
    'tasks.data_tasks.update_xg_data': {'queue': 'medium_priority'},
    'tasks.prediction_tasks.warm_cache': {'queue': 'medium_priority'},
    
    # Low priority - heavy operations
    'tasks.training_tasks.train_models': {'queue': 'low_priority'},
    'tasks.training_tasks.retrain_ensemble': {'queue': 'low_priority'},
}

# Define queues
celery_app.conf.task_queues = (
    Queue('high_priority', routing_key='high_priority'),
    Queue('medium_priority', routing_key='medium_priority'),
    Queue('low_priority', routing_key='low_priority'),
    Queue('default', routing_key='default'),
)

# Default queue
celery_app.conf.task_default_queue = 'default'
celery_app.conf.task_default_exchange = 'tasks'
celery_app.conf.task_default_routing_key = 'default'

# Import tasks to register them
celery_app.autodiscover_tasks([
    'backend.src.tasks.prediction_tasks',
    'backend.src.tasks.training_tasks',
    'backend.src.tasks.data_tasks',
])

# Logging
celery_app.conf.worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
celery_app.conf.worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'


if __name__ == '__main__':
    celery_app.start()
