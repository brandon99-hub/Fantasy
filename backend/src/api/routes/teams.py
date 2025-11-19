"""Team-related API endpoints"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from backend.src.core.database import FPLDatabase
from backend.src.utils.serialization import convert_numpy_types
from backend.src.schemas.models import TeamValidationRequest, TeamValidationResponse

router = APIRouter(prefix="/api", tags=["teams"])
db = FPLDatabase()
logger = logging.getLogger(__name__)

# Server-side cache
teams_cache = None

def get_cached_teams():
    global teams_cache
    if teams_cache is None:
        teams_cache = db.get_teams()
    return teams_cache

def clear_cache():
    global teams_cache
    teams_cache = None


@router.get("/teams")
async def get_teams() -> List[Dict[str, Any]]:
    """Get all teams"""
    try:
        teams_df = get_cached_teams()
        
        if teams_df.empty:
            raise HTTPException(status_code=404, detail="No team data available")
        
        teams = teams_df.to_dict('records')
        return convert_numpy_types(teams)
        
    except Exception as e:
        logger.error(f"Error fetching teams: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching teams: {str(e)}")


@router.post("/team/validate", response_model=TeamValidationResponse)
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
            "GKP": int(position_counts.get('GKP', 0)),
            "DEF": int(position_counts.get('DEF', 0)),
            "MID": int(position_counts.get('MID', 0)),
            "FWD": int(position_counts.get('FWD', 0))
        }
        
        required_positions = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
        for pos, required in required_positions.items():
            actual = validation["structure"][pos]
            if actual != required:
                validation["errors"].append(f"Need {required} {pos}, have {actual}")
                validation["is_valid"] = False
        
        # Check team limits
        team_counts = team_players['team_name'].value_counts()
        validation["team_counts"] = {str(k): int(v) for k, v in team_counts.to_dict().items()}
        
        for team_name, count in team_counts.items():
            if count > 3:
                validation["errors"].append(f"Too many players from {team_name} ({count}/3)")
                validation["is_valid"] = False
        
        return convert_numpy_types(validation)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating team: {str(e)}")

