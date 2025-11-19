"""
Startup script for FPL AI Optimizer
Runs backend and optionally trains models
"""

import sys
import subprocess
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.src.core.database import FPLDatabase
from backend.src.core.data_collector import FPLDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_and_update_data():
    """Check if data exists and update if needed"""
    try:
        db = FPLDatabase()
        data_collector = FPLDataCollector()
        
        players_df = db.get_players_with_stats()
        if players_df.empty:
            logger.info("📊 No data found. Updating from FPL API...")
            success = data_collector.update_all_data()
            if success:
                logger.info("✅ Data updated successfully!")
                return True
            else:
                logger.error("❌ Data update failed")
                return False
        else:
            logger.info(f"✅ Found {len(players_df)} players in database")
            return True
    except Exception as e:
        logger.error(f"Error checking data: {e}")
        return False


def start_backend():
    """Start the backend server"""
    try:
        logger.info("🚀 Starting backend server...")
        subprocess.run([
            "uvicorn",
            "backend.src.api.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        logger.info("👋 Backend server stopped")
    except Exception as e:
        logger.error(f"Error starting backend: {e}")


if __name__ == "__main__":
    logger.info("🎯 FPL AI Optimizer Startup")
    logger.info("=" * 50)
    
    # Check and update data
    if not check_and_update_data():
        logger.warning("⚠️  Starting without data - update manually")
    
    # Start backend
    start_backend()

