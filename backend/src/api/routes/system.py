"""System management API endpoints"""

from fastapi import APIRouter, HTTPException
import logging

from backend.src.core.database import FPLDatabase
from backend.src.core.data_collector import FPLDataCollector
from backend.src.models.minutes_model import MinutesPredictor
from backend.src.models.points_model import PointsPredictor
from backend.src.models.ensemble_predictor import EnsemblePredictor
from backend.src.core.optimizer import FPLOptimizer
from backend.src.schemas.models import (
    SystemStatusResponse,
    DataStatusResponse,
    RefreshDataResponse,
    TrainModelsResponse,
    ModelStatusResponse,
    HealthCheckResponse,
)

router = APIRouter(prefix="/api/system", tags=["system"])
logger = logging.getLogger(__name__)

# Initialize components
db = FPLDatabase()
data_collector = FPLDataCollector()
minutes_model = MinutesPredictor()
points_model = PointsPredictor()
ensemble_model = EnsemblePredictor()
optimizer = FPLOptimizer()


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status information"""
    try:
        return {
            "database_connected": db.test_connection(),
            "models_trained": minutes_model.is_trained() and points_model.is_trained(),
            "last_data_update": db.get_last_update(),
            "current_gameweek": db.get_current_gameweek(),
            "player_count": db.get_player_count(),
            "upcoming_fixtures": db.get_upcoming_fixtures_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")


@router.get("/data-status", response_model=DataStatusResponse)
async def get_data_status():
    """Check if we have player data"""
    try:
        players_df = db.get_players_with_stats()
        has_data = not players_df.empty
        return {
            "has_data": has_data,
            "player_count": len(players_df) if has_data else 0,
            "message": "Data available" if has_data else "No data - please update first"
        }
    except Exception as e:
        return {
            "has_data": False,
            "player_count": 0,
            "message": f"Error checking data: {str(e)}"
        }


@router.post("/refresh-data", response_model=RefreshDataResponse)
async def refresh_data():
    """Refresh FPL data from API"""
    try:
        success = data_collector.update_all_data()
        return {
            "success": success,
            "message": "Data updated successfully" if success else "Data update failed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")


@router.post("/train-models", response_model=TrainModelsResponse)
async def train_models():
    """Train AI models"""
    try:
        minutes_success = minutes_model.train()
        points_success = points_model.train()
        ensemble_success = ensemble_model.train()
        
        success = minutes_success and points_success and ensemble_success
        return {
            "success": success,
            "message": "Models trained successfully" if success else "Model training failed",
            "details": {
                "minutes_model": minutes_success,
                "points_model": points_success,
                "ensemble_model": ensemble_success
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")


@router.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get status of all AI models"""
    try:
        return {
            "minutes_model": {
                "trained": minutes_model.is_trained(),
                "type": "Gradient Boosting"
            },
            "points_model": {
                "trained": points_model.is_trained(),
                "type": "Gradient Boosting"
            },
            "ensemble_model": {
                "trained": ensemble_model.is_fitted,
                "type": "Ensemble (Multiple Algorithms)",
                "models": list(ensemble_model.model_configs.keys()) if ensemble_model.is_fitted else []
            },
            "price_predictor": {
                "trained": optimizer.price_predictor.is_trained,
                "type": "Gradient Boosting"
            },
            "fixture_analyzer": {
                "available": True,
                "type": "Rule-based Analysis"
            },
            "strategic_planner": {
                "available": True,
                "type": "Strategic Planning Engine"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model status: {str(e)}")


@router.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "FPL AI Optimizer API is running"}

