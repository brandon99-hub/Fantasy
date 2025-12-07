"""System management API endpoints"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import logging

from backend.src.core.async_db import AsyncDatabaseWrapper, get_async_db
from backend.src.core.data_collector import FPLDataCollector
from backend.src.models.minutes_model import MinutesPredictor
from backend.src.models.points_model import PointsPredictor
from backend.src.models.ensemble_predictor import EnsemblePredictor
from backend.src.core.optimizer import FPLOptimizer
from backend.src.services.prediction_evaluator import PredictionEvaluator
from backend.src.services.auto_retrain import AutoRetrainer
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

# Initialize components (these are stateful, keep as singletons)
data_collector = FPLDataCollector()
minutes_model = MinutesPredictor()
points_model = PointsPredictor()
ensemble_model = EnsemblePredictor()
optimizer = FPLOptimizer()
prediction_evaluator = PredictionEvaluator()
auto_retrainer = AutoRetrainer()


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(db: AsyncDatabaseWrapper = Depends(get_async_db)):
    """Get system status information"""
    try:
        return {
            "database_connected": await db.test_connection(),
            "models_trained": minutes_model.is_trained() and points_model.is_trained(),
            "last_data_update": await db.get_last_update(),
            "current_gameweek": await db.get_current_gameweek(),
            "player_count": await db.get_player_count(),
            "upcoming_fixtures": await db.get_upcoming_fixtures_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")


@router.get("/data-status", response_model=DataStatusResponse)
async def get_data_status(db: AsyncDatabaseWrapper = Depends(get_async_db)):
    """Check if we have player data"""
    try:
        players_data = await db.get_players_with_stats()
        # Handle both list (PostgreSQL) and DataFrame (SQLite) return types
        has_data = len(players_data) > 0 if players_data is not None else False
        return {
            "has_data": has_data,
            "player_count": len(players_data) if has_data else 0,
            "message": "Data available" if has_data else "No data - please update first"
        }
    except Exception as e:
        return {
            "has_data": False,
            "player_count": 0,
            "message": f"Error checking data: {str(e)}"
        }


@router.post("/refresh-data", response_model=RefreshDataResponse)
async def refresh_data(background_tasks: BackgroundTasks):
    """Refresh FPL data from API (runs in background)"""
    try:
        # Run data update in background to avoid blocking
        def update_data():
            try:
                success = data_collector.update_all_data()
                logger.info(f"Background data update: {'success' if success else 'failed'}")
            except Exception as e:
                logger.error(f"Background data update error: {e}")
        
        background_tasks.add_task(update_data)
        
        return {
            "success": True,
            "message": "Data update started in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")


@router.post("/train-models", response_model=TrainModelsResponse)
async def train_models(background_tasks: BackgroundTasks):
    """Train AI models (runs in background)"""
    try:
        # Run model training in background to avoid blocking
        def train_all_models():
            try:
                minutes_success = minutes_model.train()
                points_success = points_model.train()
                ensemble_success = ensemble_model.train()
                logger.info(f"Model training complete: minutes={minutes_success}, points={points_success}, ensemble={ensemble_success}")
            except Exception as e:
                logger.error(f"Model training error: {e}")
        
        background_tasks.add_task(train_all_models)
        
        return {
            "success": True,
            "message": "Model training started in background",
            "details": {
                "minutes_model": "training",
                "points_model": "training",
                "ensemble_model": "training"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")


@router.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status(db: AsyncDatabaseWrapper = Depends(get_async_db)):
    """Get status of all AI models"""
    try:
        metrics = await db.get_latest_model_metrics()
        return {
            "minutes_model": {
                "trained": minutes_model.is_trained(),
                "type": "Gradient Boosting",
                "metrics": metrics.get("minutes_model")
            },
            "points_model": {
                "trained": points_model.is_trained(),
                "type": "Gradient Boosting",
                "metrics": metrics.get("points_model")
            },
            "ensemble_model": {
                "trained": ensemble_model.is_fitted,
                "type": "Ensemble (Multiple Algorithms)",
                "models": list(ensemble_model.model_configs.keys()) if ensemble_model.is_fitted else [],
                "metrics": metrics.get("ensemble_model")
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


# ==========================================
# ADAPTIVE LEARNING ENDPOINTS
# ==========================================

@router.post("/evaluate-predictions")
async def evaluate_predictions(gameweek: int = None):
    """Evaluate prediction accuracy for a specific gameweek or recent gameweeks"""
    try:
        if gameweek:
            result = prediction_evaluator.evaluate_gameweek(gameweek)
        else:
            results = prediction_evaluator.evaluate_recent_gameweeks(num_gameweeks=3)
            result = {
                "gameweeks_evaluated": len(results),
                "results": results
            }
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"Error evaluating predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating predictions: {str(e)}")


@router.get("/prediction-accuracy")
async def get_prediction_accuracy(gameweek: int = None):
    """Get prediction accuracy metrics"""
    try:
        overall_accuracy = prediction_evaluator.get_overall_accuracy(gameweek)
        position_accuracy = prediction_evaluator.get_accuracy_by_position(gameweek)
        problem_areas = prediction_evaluator.identify_problem_areas(gameweek)
        
        return {
            "success": True,
            "overall": overall_accuracy,
            "by_position": position_accuracy,
            "problem_areas": problem_areas
        }
    except Exception as e:
        logger.error(f"Error getting prediction accuracy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting prediction accuracy: {str(e)}")


@router.get("/learning-status")
async def get_learning_status():
    """Get adaptive learning system status"""
    try:
        # Get prediction accuracy
        accuracy = prediction_evaluator.get_overall_accuracy()
        
        # Get retrain status
        retrain_check = auto_retrainer.should_retrain()
        
        # Get latest model info
        from backend.src.core.postgres_db import PostgresManagerDB
        postgres_db = PostgresManagerDB()
        
        with postgres_db.get_connection() as conn:
            with conn.cursor() as cur:
                # Get latest training
                cur.execute("""
                    SELECT model_name, version, created_at, mae, rmse, r2_score
                    FROM model_metrics
                    ORDER BY created_at DESC
                    LIMIT 5;
                """)
                recent_trainings = cur.fetchall()
                
                # Get feedback count
                cur.execute("""
                    SELECT COUNT(*) as total_feedback,
                           MAX(gameweek) as latest_gw
                    FROM prediction_feedback;
                """)
                feedback_stats = cur.fetchone()
        
        return {
            "success": True,
            "prediction_accuracy": accuracy,
            "retrain_status": retrain_check,
            "recent_trainings": recent_trainings,
            "feedback_stats": feedback_stats,
            "adaptive_learning_active": True
        }
    except Exception as e:
        logger.error(f"Error getting learning status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting learning status: {str(e)}")


@router.get("/should-retrain")
async def check_should_retrain():
    """Check if models should be retrained"""
    try:
        result = auto_retrainer.should_retrain()
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"Error checking retrain status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking retrain status: {str(e)}")


@router.post("/force-retrain")
async def force_retrain(background_tasks: BackgroundTasks):
    """Force model retraining (runs in background)"""
    try:
        def retrain():
            try:
                result = auto_retrainer.retrain_models(force=True)
                logger.info(f"Forced retrain complete: {result.get('success', False)}")
            except Exception as e:
                logger.error(f"Forced retrain error: {e}")
        
        background_tasks.add_task(retrain)
        
        return {
            "success": True,
            "message": "Model retraining started in background"
        }
    except Exception as e:
        logger.error(f"Error starting forced retrain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting forced retrain: {str(e)}")


@router.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "FPL AI Optimizer API is running"}

