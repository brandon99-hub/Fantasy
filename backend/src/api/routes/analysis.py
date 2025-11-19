"""Analysis and optimization API endpoints"""

from fastapi import APIRouter, HTTPException
import logging

from backend.src.core.database import FPLDatabase
from backend.src.core.optimizer import FPLOptimizer
from backend.src.models.points_model import PointsPredictor
from backend.src.models.minutes_model import MinutesPredictor
from backend.src.models.ensemble_predictor import EnsemblePredictor
from backend.src.utils.strategic_planner import ManagerPreferences
from backend.src.utils.serialization import convert_numpy_types
from backend.src.schemas.models import (
    TeamAnalysisRequest,
    TeamAnalysisResponse,
    AdvancedOptimizeRequest,
    AdvancedOptimizeResponse,
)

router = APIRouter(prefix="/api", tags=["analysis"])
logger = logging.getLogger(__name__)

# Initialize components
db = FPLDatabase()
optimizer = FPLOptimizer()
points_model = PointsPredictor()
minutes_model = MinutesPredictor()
ensemble_model = EnsemblePredictor()


@router.post("/analyze-team", response_model=TeamAnalysisResponse)
async def analyze_team(request: TeamAnalysisRequest):
    """Analyze team using AI models and provide transfer and captain suggestions"""
    try:
        player_ids = request.players
        analyze_transfers = request.analyze_transfers
        analyze_captain = request.analyze_captain
        
        if not player_ids:
            raise HTTPException(status_code=400, detail="No players provided")
        
        # Get player data
        players_df = db.get_players_with_stats()
        
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
async def advanced_optimize_team(request: AdvancedOptimizeRequest):
    """Advanced team optimization with comprehensive analysis"""
    try:
        # Get player data
        players_df = db.get_players_with_stats()
        
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

