"""
FastAPI backend for FPL AI Optimizer
Serves the TypeScript frontend and provides API endpoints
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Optional
import uvicorn
from pydantic import BaseModel

# Import existing modules
from database import FPLDatabase
from data_collector import FPLDataCollector
from models.minutes_model import MinutesPredictor
from models.points_model import PointsPredictor
from models.ensemble_predictor import EnsemblePredictor
from optimizer import FPLOptimizer
from utils.strategic_planner import ManagerPreferences
from fpl_api_client import FPLAPIClient


# Initialize components
db = FPLDatabase()
data_collector = FPLDataCollector()
minutes_model = MinutesPredictor()
points_model = PointsPredictor()
ensemble_model = EnsemblePredictor()
optimizer = FPLOptimizer()
fpl_api_client = FPLAPIClient()

# Server-side cache
players_cache = None
teams_cache = None

def get_cached_players():
    global players_cache
    if players_cache is None:
        players_cache = db.get_players_with_stats()
    return players_cache

def get_cached_teams():
    global teams_cache
    if teams_cache is None:
        teams_cache = db.get_teams()
    return teams_cache

def clear_cache():
    global players_cache, teams_cache
    players_cache = None
    teams_cache = None

# Auto-update data on startup
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting FPL AI Assistant...")
    
    # Check if we have data
    try:
        players_df = db.get_players_with_stats()
        if players_df.empty:
            print("📊 No data found. Updating from FPL API...")
            success = data_collector.update_all_data()
            if success:
                print("✅ Data updated successfully!")
            else:
                print("❌ Data update failed")
        else:
            print(f"✅ Found {len(players_df)} players in database")
    except Exception as e:
        print(f"❌ Error checking/updating data: {e}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down FPL AI Assistant...")

# Initialize FastAPI app with auto data update
app = FastAPI(
    title="FPL AI Optimizer API",
    description="Advanced Fantasy Premier League optimization with AI and mathematical optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for TypeScript frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PlayerResponse(BaseModel):
    id: int
    web_name: str
    first_name: str
    second_name: str
    position: str
    team_name: str
    now_cost: int
    total_points: int
    form: float
    selected_by_percent: float
    points_per_game: float
    goals_scored: int
    assists: int
    clean_sheets: int
    saves: int
    bonus: int
    ict_index: float
    influence: float
    creativity: float
    threat: float
    status: str
    chance_of_playing_this_round: Optional[int] = None
    chance_of_playing_next_round: Optional[int] = None
    news: Optional[str] = None

class TeamValidationRequest(BaseModel):
    player_ids: List[int]

class TeamValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    budget: dict
    structure: dict
    team_counts: dict

class SystemStatusResponse(BaseModel):
    database_connected: bool
    models_trained: bool
    last_data_update: Optional[str]
    current_gameweek: Optional[int]
    player_count: int
    upcoming_fixtures: int

# API Routes

@app.get("/")
async def read_root():
    """Serve the TypeScript frontend"""
    return FileResponse('fpl-optimizer-frontend/out/index.html')

@app.get("/api/players", response_model=List[PlayerResponse])
async def get_players(
    position: Optional[str] = Query(None, description="Filter by position (GKP, DEF, MID, FWD)"),
    team: Optional[str] = Query(None, description="Filter by team name"),
    search: Optional[str] = Query(None, description="Search player names"),
    min_price: Optional[float] = Query(None, description="Minimum price in millions"),
    max_price: Optional[float] = Query(None, description="Maximum price in millions"),
    min_ownership: Optional[float] = Query(None, description="Minimum ownership percentage"),
    min_form: Optional[float] = Query(None, description="Minimum form rating")
):
    """Get players with optional filtering"""
    try:
        players_df = get_cached_players()
        
        if players_df.empty:
            raise HTTPException(status_code=404, detail="No players found")
        
        # Apply filters
        if position:
            players_df = players_df[players_df['position'] == position]
        
        if team:
            players_df = players_df[players_df['team_name'].str.contains(team, case=False, na=False)]
        
        if search:
            search_mask = (
                players_df['web_name'].str.contains(search, case=False, na=False) |
                players_df['first_name'].str.contains(search, case=False, na=False) |
                players_df['second_name'].str.contains(search, case=False, na=False) |
                players_df['team_name'].str.contains(search, case=False, na=False)
            )
            players_df = players_df[search_mask]
        
        if min_price:
            players_df = players_df[players_df['now_cost'] >= min_price * 10]
        
        if max_price:
            players_df = players_df[players_df['now_cost'] <= max_price * 10]
        
        if min_ownership:
            players_df = players_df[players_df['selected_by_percent'] >= min_ownership]
        
        if min_form:
            players_df = players_df[players_df['form'] >= min_form]
        
        # Convert to list of dictionaries with proper handling
        players_df = players_df.infer_objects(copy=False)
        players = players_df.fillna(0).to_dict('records')
        
        # Limit results to prevent overwhelming the frontend
        return players[:100]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching players: {str(e)}")

@app.get("/api/players/search", response_model=List[PlayerResponse])
async def search_players(
    search: str = Query(..., description="Search term for player names"),
    position: Optional[str] = Query(None, description="Filter by position")
):
    """Search players by name"""
    try:
        players_df = get_cached_players()
        
        if players_df.empty:
            raise HTTPException(status_code=404, detail="No players found")
        
        # Search in multiple fields
        search_mask = (
            players_df['web_name'].str.contains(search, case=False, na=False) |
            players_df['first_name'].str.contains(search, case=False, na=False) |
            players_df['second_name'].str.contains(search, case=False, na=False)
        )
        
        filtered_df = players_df[search_mask]
        
        if position:
            filtered_df = filtered_df[filtered_df['position'] == position]
        
        # Sort by relevance (exact matches first, then by total points)
        def calculate_relevance(row):
            name_fields = [row['web_name'], row['first_name'], row['second_name']]
            exact_match = any(search.lower() == field.lower() for field in name_fields if pd.notna(field))
            starts_with = any(field.lower().startswith(search.lower()) for field in name_fields if pd.notna(field))
            
            if exact_match:
                return 1000 + row['total_points']
            elif starts_with:
                return 500 + row['total_points']
            else:
                return row['total_points']
        
        import pandas as pd
        filtered_df['relevance'] = filtered_df.apply(calculate_relevance, axis=1)
        filtered_df = filtered_df.sort_values('relevance', ascending=False)
        
        filtered_df = filtered_df.infer_objects(copy=False)
        players = filtered_df.fillna(0).head(20).to_dict('records')
        return players
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching players: {str(e)}")

@app.get("/api/teams")
async def get_teams():
    """Get all teams"""
    try:
        # Get teams from cache
        teams_df = get_cached_teams()
        
        if teams_df.empty:
            raise HTTPException(status_code=404, detail="No team data available")
        
        # Convert to list of dictionaries
        teams = teams_df.to_dict('records')
        
        return teams
        
    except Exception as e:
        logger.error(f"Error fetching teams: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching teams: {str(e)}")

@app.post("/api/team/validate", response_model=TeamValidationResponse)
async def validate_team(request: TeamValidationRequest):
    """Validate team structure and constraints"""
    try:
        players_df = db.get_players_with_stats()
        team_players = players_df[players_df['id'].isin(request.player_ids)]
        
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "budget": {"total": 0, "remaining": 100},
            "structure": {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0},
            "team_counts": {}
        }
        
        if len(team_players) != len(request.player_ids):
            validation["errors"].append("Some players not found in database")
            validation["is_valid"] = False
            return validation
        
        # Calculate budget
        total_cost = team_players['now_cost'].sum() / 10
        validation["budget"]["total"] = total_cost
        validation["budget"]["remaining"] = 100 - total_cost
        
        if total_cost > 100:
            validation["errors"].append(f"Over budget by £{total_cost - 100:.1f}M")
            validation["is_valid"] = False
        
        # Check structure
        position_counts = team_players['position'].value_counts()
        validation["structure"] = {
            "GKP": position_counts.get('GKP', 0),
            "DEF": position_counts.get('DEF', 0),
            "MID": position_counts.get('MID', 0),
            "FWD": position_counts.get('FWD', 0)
        }
        
        required_positions = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
        for pos, required in required_positions.items():
            actual = validation["structure"][pos]
            if actual != required:
                validation["errors"].append(f"Need {required} {pos}, have {actual}")
                validation["is_valid"] = False
        
        # Check team limits
        team_counts = team_players['team_name'].value_counts()
        validation["team_counts"] = team_counts.to_dict()
        
        for team_name, count in team_counts.items():
            if count > 3:
                validation["errors"].append(f"Too many players from {team_name} ({count}/3)")
                validation["is_valid"] = False
        
        return validation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating team: {str(e)}")

@app.get("/api/system/status", response_model=SystemStatusResponse)
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

@app.post("/api/system/refresh-data")
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

@app.get("/api/system/data-status")
async def get_data_status():
    """Check if we have player data"""
    try:
        players_df = db.get_players_with_stats()
        return {
            "has_data": not players_df.empty,
            "player_count": len(players_df) if not players_df.empty else 0,
            "message": "Data available" if not players_df.empty else "No data - please update first"
        }
    except Exception as e:
        return {
            "has_data": False,
            "player_count": 0,
            "message": f"Error checking data: {str(e)}"
        }

@app.get("/api/managers/search")
async def search_managers(name: str = Query(..., description="Manager name to search")):
    """Search for FPL managers by name"""
    try:
        # Use the FPL API client to search
        managers = fpl_api_client.search_managers(name)
        
        return {
            "managers": managers,
            "message": "Manager search results" if managers else f"No managers found for '{name}'"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching managers: {str(e)}")

@app.get("/api/team/manager/{manager_id}")
async def get_team_by_manager(manager_id: str):
    """Get team by manager ID using FPL API"""
    try:
        # Validate manager ID format
        if not manager_id.isdigit():
            raise HTTPException(status_code=400, detail="Manager ID must be numeric")
        
        team_data = fpl_api_client.get_manager_team(manager_id)
        
        if not team_data:
            raise HTTPException(status_code=404, detail="Manager not found or team data unavailable from FPL API")
        
        return team_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching team: {str(e)}")

@app.post("/api/analyze-team")
async def analyze_team(request: dict):
    """Analyze team using AI models and provide transfer and captain suggestions"""
    try:
        player_ids = request.get('players', [])
        analyze_transfers = request.get('analyze_transfers', True)
        analyze_captain = request.get('analyze_captain', True)
        
        if not player_ids:
            raise HTTPException(status_code=400, detail="No players provided")
        
        # Get player data
        players_df = db.get_players_with_stats()
        
        if players_df.empty:
            raise HTTPException(status_code=400, detail="No player data available. Please update data first.")
        
        team_players = players_df[players_df['id'].isin(player_ids)]
        
        if len(team_players) != len(player_ids):
            raise HTTPException(status_code=400, detail=f"Found {len(team_players)} players out of {len(player_ids)} requested. Some players not found in database.")
        
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
        
        # Use AI models if trained
        if analyze_captain and points_model.is_trained():
            try:
                # Get AI predictions for current team
                team_predictions = points_model.predict_points(team_players)
                if not team_predictions.empty:
                    # Get top 3 captain options based on AI predictions
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
                            "expected_points": round(row['predicted_points'] * 2, 1),  # Captain doubles
                            "reason": f"AI predicts {row['predicted_points']:.1f} points (form: {row.get('form', 0):.1f})",
                            "confidence": "High" if row.get('start_probability', 0.5) > 0.8 else "Medium"
                        }
                        for _, row in captain_candidates.iterrows()
                    ]
                else:
                    # Fallback to form-based
                    captain_candidates = team_players.nlargest(3, 'form')
                    analysis_result["captain_suggestions"] = [
                        {
                            "player": {
                                "id": int(player['id']),
                                "web_name": player['web_name'],
                                "team_name": player['team_name'],
                                "position": player['position'],
                                "expected_points": round(player['form'], 1)
                            },
                            "expected_points": round(player['form'] * 2, 1),
                            "reason": f"Top form player ({player['form']:.1f} avg points)",
                            "confidence": "Medium"
                        }
                        for _, player in captain_candidates.iterrows()
                    ]
            except Exception as e:
                # Fallback to form-based captain suggestions
                captain_candidates = team_players.nlargest(3, 'total_points')
                analysis_result["captain_suggestions"] = [
                    {
                        "player": {
                            "id": int(player['id']),
                            "web_name": player['web_name'],
                            "team_name": player['team_name'],
                            "position": player['position'],
                            "expected_points": round(player['total_points'] / 4, 1)  # Rough estimate
                        },
                        "expected_points": round(player['total_points'] / 4 * 2, 1),  # Captain doubles
                        "reason": f"Highest points scorer ({player['total_points']} total points)",
                        "confidence": "High"
                    }
                    for _, player in captain_candidates.iterrows()
                ]
        
        if analyze_transfers:
            try:
                # Get all players for comparison
                all_players = players_df[~players_df['id'].isin(player_ids)]
                
                # Use AI predictions if models are trained
                if points_model.is_trained():
                    all_predictions = points_model.predict_points(all_players)
                    team_predictions = points_model.predict_points(team_players)
                    
                    if not all_predictions.empty and not team_predictions.empty:
                        # AI-based transfer suggestions
                        for _, current_player in team_predictions.iterrows():
                            # Find better alternatives in same position
                            position_alternatives = all_predictions[
                                (all_predictions['position'] == current_player['position']) &
                                (all_predictions['now_cost'] >= current_player['now_cost'] - 10) &
                                (all_predictions['now_cost'] <= current_player['now_cost'] + 20) &
                                (all_predictions['predicted_points'] > current_player['predicted_points'])
                            ].nlargest(3, 'predicted_points')
                            
                            for _, alternative in position_alternatives.iterrows():
                                points_gain = alternative['predicted_points'] - current_player['predicted_points']
                                
                                if points_gain > 0.5:  # Only suggest meaningful improvements
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
                                        "cost_change": round(cost_change * 10, 0),  # Return as 0.1M units
                                        "reason": f"AI predicts {alternative['predicted_points']:.1f} vs {current_player['predicted_points']:.1f} points",
                                        "priority": "High" if points_gain > 2 else "Medium"
                                    })
                    else:
                        # Fallback to form-based suggestions
                        for _, current_player in team_players.iterrows():
                            position_alternatives = all_players[
                                (all_players['position'] == current_player['position']) &
                                (all_players['now_cost'] >= current_player['now_cost'] - 5) &
                                (all_players['now_cost'] <= current_player['now_cost'] + 10) &
                                (all_players['form'] > current_player['form'])
                            ].nlargest(2, 'form')
                            
                            for _, alternative in position_alternatives.iterrows():
                                points_gain = alternative['form'] - current_player['form']
                                if points_gain > 1:
                                    cost_change = (alternative['now_cost'] - current_player['now_cost']) / 10
                                    
                                    analysis_result["transfer_suggestions"].append({
                                        "player_out": current_player['web_name'],
                                        "player_in": alternative['web_name'],
                                        "points_gain": round(points_gain, 1),
                                        "cost_change": f"£{cost_change:+.1f}M" if abs(cost_change) >= 0.1 else "Same price",
                                        "reason": f"Better form: {alternative['form']:.1f} vs {current_player['form']:.1f}",
                                        "priority": "High" if points_gain > 2 else "Medium"
                                    })
                else:
                    # Use form-based suggestions as fallback
                    for _, current_player in team_players.iterrows():
                        position_alternatives = all_players[
                            (all_players['position'] == current_player['position']) &
                            (all_players['now_cost'] >= current_player['now_cost'] - 5) &
                            (all_players['now_cost'] <= current_player['now_cost'] + 10) &
                            (all_players['form'] > current_player['form'])
                        ].nlargest(2, 'form')
                        
                        for _, alternative in position_alternatives.iterrows():
                            points_gain = alternative['form'] - current_player['form']
                            if points_gain > 1:
                                cost_change = (alternative['now_cost'] - current_player['now_cost']) / 10
                                
                                analysis_result["transfer_suggestions"].append({
                                    "player_out": current_player['web_name'],
                                    "player_in": alternative['web_name'],
                                    "points_gain": round(points_gain, 1),
                                    "cost_change": f"£{cost_change:+.1f}M" if abs(cost_change) >= 0.1 else "Same price",
                                    "reason": f"Better form: {alternative['form']:.1f} vs {current_player['form']:.1f}",
                                    "priority": "High" if points_gain > 2 else "Medium"
                                })
            
            except Exception as e:
                print(f"Error in AI transfer analysis: {e}")
                # Fallback to basic form comparison
                pass
            
            # Sort by points gain and limit results
            analysis_result["transfer_suggestions"].sort(key=lambda x: x['points_gain'], reverse=True)
            analysis_result["transfer_suggestions"] = analysis_result["transfer_suggestions"][:5]
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing team: {str(e)}")

@app.post("/api/system/train-models")
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

@app.post("/api/advanced-optimize")
async def advanced_optimize_team(request: dict):
    """Advanced team optimization with comprehensive analysis"""
    try:
        player_ids = request.get('players', [])
        budget = request.get('budget', 100.0)
        free_transfers = request.get('free_transfers', 1)
        use_wildcard = request.get('use_wildcard', False)
        formation = request.get('formation', '3-4-3')
        
        # Manager preferences
        preferences_data = request.get('preferences', {})
        preferences = ManagerPreferences(
            risk_tolerance=preferences_data.get('risk_tolerance', 0.5),
            formation_preference=preferences_data.get('formation_preference', '3-4-3'),
            budget_allocation=preferences_data.get('budget_allocation', {
                'GKP': 0.08, 'DEF': 0.25, 'MID': 0.40, 'FWD': 0.27
            }),
            differential_threshold=preferences_data.get('differential_threshold', 5.0),
            captain_strategy=preferences_data.get('captain_strategy', 'fixture_based'),
            transfer_frequency=preferences_data.get('transfer_frequency', 'moderate')
        )
        
        # Analysis options
        include_fixture_analysis = request.get('include_fixture_analysis', True)
        include_price_analysis = request.get('include_price_analysis', True)
        include_strategic_planning = request.get('include_strategic_planning', True)
        
        if not player_ids:
            raise HTTPException(status_code=400, detail="No players provided")
        
        # Get player data
        players_df = db.get_players_with_stats()
        
        if players_df.empty:
            raise HTTPException(status_code=400, detail="No player data available. Please update data first.")
        
        # Perform advanced optimization
        result = optimizer.advanced_optimize_team(
            players_df=players_df,
            budget=budget,
            current_team=player_ids,
            free_transfers=free_transfers,
            use_wildcard=use_wildcard,
            formation=formation,
            preferences=preferences,
            include_fixture_analysis=include_fixture_analysis,
            include_price_analysis=include_price_analysis,
            include_strategic_planning=include_strategic_planning
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in advanced optimization: {str(e)}")

@app.get("/api/system/model-status")
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "FPL AI Optimizer API is running"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
