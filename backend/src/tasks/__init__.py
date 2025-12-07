"""Task package initialization"""

from backend.src.tasks.training_tasks import train_all_models, train_single_model
from backend.src.tasks.data_tasks import (
    refresh_fpl_data,
    update_xg_data,
    cleanup_old_data
)

__all__ = [
    'train_all_models',
    'train_single_model',
    'refresh_fpl_data',
    'update_xg_data',
    'cleanup_old_data'
]
