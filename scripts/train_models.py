"""
Script to train all ML models
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.src.models.minutes_model import MinutesPredictor
from backend.src.models.points_model import PointsPredictor
from backend.src.models.ensemble_predictor import EnsemblePredictor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train all models"""
    logger.info("🤖 Starting model training...")
    logger.info("=" * 50)
    
    # Train minutes model
    logger.info("Training minutes prediction model...")
    minutes_model = MinutesPredictor()
    if minutes_model.train():
        logger.info("✅ Minutes model trained successfully")
    else:
        logger.error("❌ Minutes model training failed")
        return False
    
    # Train points model
    logger.info("Training points prediction model...")
    points_model = PointsPredictor()
    if points_model.train():
        logger.info("✅ Points model trained successfully")
    else:
        logger.error("❌ Points model training failed")
        return False
    
    # Train ensemble model
    logger.info("Training ensemble model (this may take a few minutes)...")
    ensemble_model = EnsemblePredictor()
    if ensemble_model.train():
        logger.info("✅ Ensemble model trained successfully")
    else:
        logger.error("❌ Ensemble model training failed")
        return False
    
    logger.info("=" * 50)
    logger.info("🎉 All models trained successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

