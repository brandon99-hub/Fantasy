"""Player-related API endpoints"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from backend.src.core.database import FPLDatabase
from backend.src.utils.serialization import convert_numpy_types
from backend.src.schemas.models import PlayerResponse

router = APIRouter(prefix="/api/players", tags=["players"])
db = FPLDatabase()

# Server-side cache
players_cache = None

def get_cached_players():
    global players_cache
    if players_cache is None:
        players_cache = db.get_players_with_stats()
    return players_cache

def clear_cache():
    global players_cache
    players_cache = None


@router.get("", response_model=List[PlayerResponse])
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
        return convert_numpy_types(players[:100])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching players: {str(e)}")


@router.get("/search", response_model=List[PlayerResponse])
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
            exact_match = any(search.lower() == str(field).lower() for field in name_fields if pd.notna(field))
            starts_with = any(str(field).lower().startswith(search.lower()) for field in name_fields if pd.notna(field))
            
            if exact_match:
                return 1000 + row['total_points']
            elif starts_with:
                return 500 + row['total_points']
            else:
                return row['total_points']
        
        filtered_df['relevance'] = filtered_df.apply(calculate_relevance, axis=1)
        filtered_df = filtered_df.sort_values('relevance', ascending=False)
        
        filtered_df = filtered_df.infer_objects(copy=False)
        players = filtered_df.fillna(0).head(20).to_dict('records')
        return convert_numpy_types(players)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching players: {str(e)}")

