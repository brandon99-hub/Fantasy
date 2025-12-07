"""Analysis and optimization API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
import logging

from backend.src.core.async_db import AsyncDatabaseWrapper, get_async_db
from backend.src.core.optimizer import FPLOptimizer
from backend.src.models.points_model import PointsPredictor
from backend.src.models.minutes_model import MinutesPredictor
from backend.src.models.ensemble_predictor import EnsemblePredictor
from backend.src.utils.strategic_planner import ManagerPreferences
from backend.src.utils.serialization import convert_numpy_types
from backend.src.services.elite_analyzer import EliteAnalyzer
from backend.src.models.captain_bandit import CaptainBandit
from backend.src.services.feature_importance_tracker import FeatureImportanceTracker
from backend.src.services.ab_testing import ABTestingFramework
from backend.src.schemas.models import (
    TeamAnalysisRequest,
    TeamAnalysisResponse,
    AdvancedOptimizeRequest,
    AdvancedOptimizeResponse,
)

router = APIRouter(prefix="/api", tags=["analysis"])
logger = logging.getLogger(__name__)

# Initialize stateful components (keep as singletons)
optimizer = FPLOptimizer()
points_model = PointsPredictor()
minutes_model = MinutesPredictor()
ensemble_model = EnsemblePredictor()
elite_analyzer = EliteAnalyzer()
captain_bandit = CaptainBandit()
feature_tracker = FeatureImportanceTracker()
ab_framework = ABTestingFramework()


@router.post("/analyze-team", response_model=TeamAnalysisResponse)
async def analyze_team(
    request: TeamAnalysisRequest,
    db: AsyncDatabaseWrapper = Depends(get_async_db)
):
    """Analyze team using AI models and provide transfer and captain suggestions"""
    try:
        player_ids = request.players
        analyze_transfers = request.analyze_transfers
        analyze_captain = request.analyze_captain
        
        if not player_ids:
            raise HTTPException(status_code=400, detail="No players provided")
        
        # Async database call
        players_df = await db.get_players_with_stats()
        
        if players_df.empty:
            raise HTTPException(status_code=400, detail="No player data available. Please update data first.")
        
        team_players = players_df[players_df['id'].isin(player_ids)]
        
        if len(team_players) != len(player_ids):
            raise HTTPException(status_code=400, detail=f"Found {len(team_players)} players out of {len(player_ids)} requested")
        
        analysis_result = {
            "team_analysis": {
                "total_value": float(team_players['now_cost'].sum() / 10),
                "total_points": int(team_players['total_points'].sum()),
                "average_form": float(team_players['form'].mean()),
                "position_breakdown": team_players['position'].value_counts().to_dict()
            },
            "transfer_suggestions": [],
            "captain_suggestions": []
        }
        
        # Captain suggestions
        if analyze_captain and points_model.is_trained():
            try:
                team_predictions = points_model.predict_points(team_players)
                if not team_predictions.empty:
                    captain_candidates = team_predictions.nlargest(3, 'predicted_points')
                    analysis_result["captain_suggestions"] = [
                        {
                            "player": {
                                "id": int(row['id']),
                                "web_name": row['web_name'],
                                "team_name": row['team_name'],
                                "position": row['position'],
                                "expected_points": round(row['predicted_points'], 1)
                            },
                            "expected_points": round(row['predicted_points'] * 2, 1),
                            "reason": f"AI predicts {row['predicted_points']:.1f} points",
                            "confidence": "High" if row.get('start_probability', 0.5) > 0.8 else "Medium"
                        }
                        for _, row in captain_candidates.iterrows()
                    ]
            except Exception as e:
                logger.error(f"Error in captain analysis: {e}")
                captain_candidates = team_players.nlargest(3, 'total_points')
                analysis_result["captain_suggestions"] = [
                    {
                        "player": {
                            "id": int(player['id']),
                            "web_name": player['web_name'],
                            "team_name": player['team_name'],
                            "position": player['position'],
                            "expected_points": round(player['total_points'] / 4, 1)
                        },
                        "expected_points": round(player['total_points'] / 4 * 2, 1),
                        "reason": f"Highest points scorer ({player['total_points']} total)",
                        "confidence": "High"
                    }
                    for _, player in captain_candidates.iterrows()
                ]
        
        # Transfer suggestions
        if analyze_transfers:
            try:
                all_players = players_df[~players_df['id'].isin(player_ids)]
                
                if points_model.is_trained():
                    all_predictions = points_model.predict_points(all_players)
                    team_predictions = points_model.predict_points(team_players)
                    
                    if not all_predictions.empty and not team_predictions.empty:
                        for _, current_player in team_predictions.iterrows():
                            position_alternatives = all_predictions[
                                (all_predictions['position'] == current_player['position']) &
                                (all_predictions['now_cost'] >= current_player['now_cost'] - 10) &
                                (all_predictions['now_cost'] <= current_player['now_cost'] + 20) &
                                (all_predictions['predicted_points'] > current_player['predicted_points'])
                            ].nlargest(3, 'predicted_points')
                            
                            for _, alternative in position_alternatives.iterrows():
                                points_gain = alternative['predicted_points'] - current_player['predicted_points']
                                if points_gain > 0.5:
                                    cost_change = (alternative['now_cost'] - current_player['now_cost']) / 10
                                    analysis_result["transfer_suggestions"].append({
                                        "player_out": {
                                            "id": int(current_player['id']),
                                            "web_name": current_player['web_name'],
                                            "team_name": current_player['team_name'],
                                            "position": current_player['position'],
                                            "now_cost": int(current_player['now_cost']),
                                            "total_points": int(current_player['total_points'])
                                        },
                                        "player_in": {
                                            "id": int(alternative['id']),
                                            "web_name": alternative['web_name'],
                                            "team_name": alternative['team_name'],
                                            "position": alternative['position'],
                                            "now_cost": int(alternative['now_cost']),
                                            "total_points": int(alternative['total_points']),
                                            "expected_points": round(alternative['predicted_points'], 1)
                                        },
                                        "points_gain": round(points_gain, 1),
                                        "cost_change": round(cost_change * 10, 0),
                                        "reason": f"AI predicts {alternative['predicted_points']:.1f} vs {current_player['predicted_points']:.1f} points",
                                        "priority": "High" if points_gain > 2 else "Medium"
                                    })
            except Exception as e:
                logger.error(f"Error in transfer analysis: {e}")
            
            analysis_result["transfer_suggestions"].sort(key=lambda x: x['points_gain'], reverse=True)
            analysis_result["transfer_suggestions"] = analysis_result["transfer_suggestions"][:5]
        
        return convert_numpy_types(analysis_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing team: {str(e)}")


@router.post("/advanced-optimize", response_model=AdvancedOptimizeResponse)
async def advanced_optimize_team(
    request: AdvancedOptimizeRequest,
    db: AsyncDatabaseWrapper = Depends(get_async_db)
):
    """Advanced team optimization with comprehensive analysis"""
    try:
        # Async database call
        players_df = await db.get_players_with_stats()
        
        if players_df.empty:
            raise HTTPException(status_code=400, detail="No player data available")
        
        # Convert preferences model to ManagerPreferences
        preferences = None
        if request.preferences:
            preferences = ManagerPreferences(
                risk_tolerance=request.preferences.risk_tolerance,
                formation_preference=request.preferences.formation_preference,
                budget_allocation=request.preferences.budget_allocation or {
                    'GKP': 0.08, 'DEF': 0.25, 'MID': 0.40, 'FWD': 0.27
                },
                differential_threshold=request.preferences.differential_threshold,
                captain_strategy=request.preferences.captain_strategy,
                transfer_frequency=request.preferences.transfer_frequency
            )
        
        # Perform advanced optimization
        result = optimizer.advanced_optimize_team(
            players_df=players_df,
            budget=request.budget,
            current_team=request.players,
            free_transfers=request.free_transfers,
            use_wildcard=request.use_wildcard,
            formation=request.formation,
            preferences=preferences,
            include_fixture_analysis=request.include_fixture_analysis,
            include_price_analysis=request.include_price_analysis,
            include_strategic_planning=request.include_strategic_planning,
            starting_xi=request.starting_xi,
            chips_available=request.chips_available,
            bank_amount=request.bank_amount
        )
        
        return convert_numpy_types(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in advanced optimization: {str(e)}")


# ==========================================
# PHASE 4: ADVANCED LEARNING ENDPOINTS
# ==========================================

@router.get("/elite-insights/transfers")
async def get_elite_transfers(gameweek: int, cohort: str = "top10k_overall"):
    """Get most popular transfers among elite managers"""
    try:
        transfers = elite_analyzer.get_elite_transfers(gameweek, cohort)
        return {
            "success": True,
            "gameweek": gameweek,
            "cohort": cohort,
            "transfers": transfers
        }
    except Exception as e:
        logger.error(f"Error getting elite transfers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting elite transfers: {str(e)}")


@router.get("/elite-insights/captains")
async def get_elite_captains(gameweek: int, cohort: str = "top10k_overall"):
    """Get most popular captain choices among elite managers"""
    try:
        captains = elite_analyzer.get_elite_captains(gameweek, cohort)
        return {
            "success": True,
            "gameweek": gameweek,
            "cohort": cohort,
            "captains": captains
        }
    except Exception as e:
        logger.error(f"Error getting elite captains: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting elite captains: {str(e)}")


@router.get("/elite-insights/differentials")
async def get_elite_differentials(gameweek: int, cohort: str = "top10k_overall", ownership_threshold: float = 10.0):
    """Get differential picks (low overall ownership, high elite ownership)"""
    try:
        differentials = elite_analyzer.get_elite_differentials(gameweek, cohort, ownership_threshold)
        return {
            "success": True,
            "gameweek": gameweek,
            "cohort": cohort,
            "differentials": differentials
        }
    except Exception as e:
        logger.error(f"Error getting elite differentials: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting elite differentials: {str(e)}")


@router.get("/elite-insights/chip-timing")
async def get_chip_timing(cohort: str = "top10k_overall"):
    """Get chip usage timing from elite managers"""
    try:
        chip_timing = elite_analyzer.analyze_chip_timing(cohort)
        return {
            "success": True,
            "cohort": cohort,
            "chip_timing": chip_timing
        }
    except Exception as e:
        logger.error(f"Error getting chip timing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting chip timing: {str(e)}")


@router.post("/captain-bandit/recommend")
async def get_captain_recommendations(
    candidates: list,
    context: dict,
    gameweek: int,
    top_n: int = 5
):
    """Get captain recommendations using contextual bandit"""
    try:
        # Load bandit model
        captain_bandit.load_model()
        
        # Get recommendations
        recommendations = captain_bandit.get_top_captains(candidates, context, top_n)
        
        return {
            "success": True,
            "gameweek": gameweek,
            "recommendations": recommendations,
            "strategy": "contextual_bandit"
        }
    except Exception as e:
        logger.error(f"Error getting captain recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting captain recommendations: {str(e)}")


@router.post("/captain-bandit/train")
async def train_captain_bandit(num_gameweeks: int = 10):
    """Train captain bandit from historical data"""
    try:
        captain_bandit.batch_update_from_history(num_gameweeks)
        captain_bandit.save_model()
        
        return {
            "success": True,
            "message": f"Captain bandit trained on last {num_gameweeks} gameweeks",
            "model_saved": True
        }
    except Exception as e:
        logger.error(f"Error training captain bandit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training captain bandit: {str(e)}")


@router.get("/feature-importance/report")
async def get_feature_importance_report(gameweek: int = None):
    """Get comprehensive feature importance report"""
    try:
        report = feature_tracker.generate_feature_report(gameweek)
        return {
            "success": True,
            "report": report
        }
    except Exception as e:
        logger.error(f"Error generating feature report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating feature report: {str(e)}")


@router.get("/feature-importance/top-features")
async def get_top_features(gameweek: int = None, top_n: int = 15):
    """Get top N most important features"""
    try:
        top_features = feature_tracker.get_top_features(gameweek, top_n)
        return {
            "success": True,
            "gameweek": gameweek or "latest",
            "top_features": top_features
        }
    except Exception as e:
        logger.error(f"Error getting top features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting top features: {str(e)}")


@router.post("/ab-testing/create-experiment")
async def create_ab_experiment(
    name: str,
    description: str,
    control_strategy: str,
    treatment_strategy: str,
    allocation_ratio: float = 0.5
):
    """Create a new A/B test experiment"""
    try:
        experiment_id = ab_framework.create_experiment(
            name, description, control_strategy, treatment_strategy, allocation_ratio
        )
        
        if experiment_id:
            return {
                "success": True,
                "experiment_id": experiment_id,
                "message": f"Experiment '{name}' created successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create experiment")
            
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating experiment: {str(e)}")


@router.get("/ab-testing/experiments")
async def get_active_experiments():
    """Get all active A/B test experiments"""
    try:
        experiments = ab_framework.get_active_experiments()
        return {
            "success": True,
            "experiments": experiments
        }
    except Exception as e:
        logger.error(f"Error getting experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting experiments: {str(e)}")


@router.get("/ab-testing/results/{experiment_id}")
async def get_experiment_results(experiment_id: int, metric_name: str = None):
    """Get results for an A/B test experiment"""
    try:
        results = ab_framework.get_experiment_results(experiment_id)
        
        response = {
            "success": True,
            "experiment_id": experiment_id,
            "results": results
        }
        
        # Add statistical significance if metric specified
        if metric_name:
            significance = ab_framework.calculate_statistical_significance(experiment_id, metric_name)
            response["statistical_analysis"] = significance
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting experiment results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting experiment results: {str(e)}")

