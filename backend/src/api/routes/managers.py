"""Manager search and team retrieval API endpoints"""

from fastapi import APIRouter, HTTPException, Query

from backend.src.services.fpl_api_client import FPLAPIClient
from backend.src.utils.serialization import convert_numpy_types
from backend.src.schemas.models import ManagerSearchResponse, ManagerTeamResponse

router = APIRouter(prefix="/api", tags=["managers"])
fpl_api_client = FPLAPIClient()


@router.get("/managers/search", response_model=ManagerSearchResponse)
async def search_managers(name: str = Query(..., description="Manager name to search")):
    """Search for FPL managers by name"""
    try:
        managers = fpl_api_client.search_managers(name)
        
        return convert_numpy_types({
            "managers": managers,
            "message": "Manager search results" if managers else f"No managers found for '{name}'"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching managers: {str(e)}")


@router.get("/team/manager/{manager_id}", response_model=ManagerTeamResponse)
async def get_team_by_manager(manager_id: str):
    """Get team by manager ID using FPL API"""
    try:
        if not manager_id.isdigit():
            raise HTTPException(status_code=400, detail="Manager ID must be numeric")
        
        team_data = fpl_api_client.get_manager_team(manager_id)
        
        if not team_data:
            raise HTTPException(
                status_code=404, 
                detail="Manager not found or team data unavailable from FPL API"
            )
        
        return convert_numpy_types(team_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching team: {str(e)}")

