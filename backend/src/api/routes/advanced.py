"""
API routes for advanced analysis features
Form analysis, multi-GW planning, differentials, formation optimization
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from backend.src.core.database import FPLDatabase
from backend.src.utils.form_analyzer import FormAnalyzer
from backend.src.core.multi_gw_optimizer import MultiGWOptimizer
from backend.src.utils.differential_finder import DifferentialFinder
from backend.src.core.formation_optimizer import FormationOptimizer
from backend.src.utils.serialization import convert_numpy_types
from backend.src.core.cache import cache_result

router = APIRouter(prefix="/api/advanced", tags=["advanced"])
logger = logging.getLogger(__name__)

db = FPLDatabase()
form_analyzer = FormAnalyzer()
multi_gw_optimizer = MultiGWOptimizer()
differential_finder = DifferentialFinder()
formation_optimizer = FormationOptimizer()


@router.get("/form-analysis/{player_id}")
@cache_result(ttl=1800, key_prefix="form_analysis")
async def analyze_player_form(player_id: int):
    """
    Analyze player form momentum and trends
    
    Returns:
        - Momentum score (-1 to 1)
        - Hot/cold streak detection
        - Breakout analysis
        - Form continuation probability
    """
    try:
        # Get player history
        player_history = db.get_player_history(player_id)
        
        if player_history.empty:
            raise HTTPException(status_code=404, detail=f"No history found for player {player_id}")
        
        # Analyze form
        analysis = form_analyzer.analyze_player_form(player_history)
        
        return convert_numpy_types(analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing form for player {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/form-analysis/batch")
async def batch_analyze_form(player_ids: List[int]):
    """Analyze form for multiple players"""
    try:
        players_df = db.get_players_with_stats()
        players_df = players_df[players_df['id'].isin(player_ids)]
        
        history_df = db.get_all_player_history()
        
        # Batch analyze
        result = form_analyzer.batch_analyze_players(players_df, history_df)
        
        return convert_numpy_types(result.to_dict('records'))
        
    except Exception as e:
        logger.error(f"Error in batch form analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-gw-plan")
async def create_multi_gw_plan(
    current_team: List[int],
    horizon: int = Query(5, ge=1, le=10),
    free_transfers: int = Query(1, ge=0, le=2),
    bank: float = Query(0.0, ge=0.0),
    budget: float = Query(100.0, ge=80.0, le=120.0)
):
    """
    Create multi-gameweek transfer plan
    
    Args:
        current_team: List of 15 player IDs
        horizon: Number of gameweeks to plan (1-10)
        free_transfers: Current free transfers (0-2)
        bank: Money in bank (millions)
        budget: Total budget (millions)
    
    Returns:
        Transfer plan for each gameweek with recommendations
    """
    try:
        if len(current_team) != 15:
            raise HTTPException(status_code=400, detail="Team must have exactly 15 players")
        
        # Get player data with predictions
        players_df = db.get_players_with_stats()
        
        if players_df.empty:
            raise HTTPException(status_code=400, detail="No player data available")
        
        # Create transfer plan
        plan = multi_gw_optimizer.optimize_transfer_sequence(
            players_df=players_df,
            current_team=current_team,
            horizon=horizon,
            free_transfers=free_transfers,
            bank=bank,
            budget=budget
        )
        
        return convert_numpy_types(plan)
        
    except Exception as e:
        logger.error(f"Error creating multi-GW plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/differentials")
@cache_result(ttl=3600, key_prefix="differentials")
async def find_differentials(
    ownership_threshold: float = Query(10.0, ge=0.0, le=50.0),
    min_predicted_points: float = Query(4.0, ge=0.0),
    risk_tolerance: str = Query("medium", regex="^(low|medium|high)$"),
    position: Optional[str] = Query(None, regex="^(GKP|DEF|MID|FWD)$"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Find differential players
    
    Args:
        ownership_threshold: Max ownership % (default: 10%)
        min_predicted_points: Minimum predicted points (default: 4.0)
        risk_tolerance: low, medium, or high
        position: Filter by position (optional)
        limit: Number of results
    
    Returns:
        List of differential players with scores and risk ratings
    """
    try:
        players_df = db.get_players_with_stats()
        
        if players_df.empty:
            raise HTTPException(status_code=400, detail="No player data available")
        
        # Find differentials
        differentials = differential_finder.find_differentials(
            players_df=players_df,
            ownership_threshold=ownership_threshold,
            min_predicted_points=min_predicted_points,
            risk_tolerance=risk_tolerance,
            position=position
        )
        
        # Limit results
        differentials = differentials.head(limit)
        
        return convert_numpy_types(differentials.to_dict('records'))
        
    except Exception as e:
        logger.error(f"Error finding differentials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/template-balance")
async def analyze_template_balance(
    team: List[int],
    target_differential_count: int = Query(4, ge=0, le=10)
):
    """
    Analyze template vs differential balance in team
    
    Args:
        team: List of 15 player IDs
        target_differential_count: Target number of differentials
    
    Returns:
        Balance analysis with recommendations
    """
    try:
        if len(team) != 15:
            raise HTTPException(status_code=400, detail="Team must have exactly 15 players")
        
        players_df = db.get_players_with_stats()
        
        analysis = differential_finder.calculate_template_balance(
            team=team,
            players_df=players_df,
            target_differential_count=target_differential_count
        )
        
        return convert_numpy_types(analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing template balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rank-strategy")
async def get_rank_strategy(
    current_rank: int = Query(..., ge=1),
    target_rank: int = Query(..., ge=1),
    gameweeks_remaining: int = Query(..., ge=1, le=38)
):
    """
    Get differential strategy recommendation based on rank goals
    
    Args:
        current_rank: Current overall rank
        target_rank: Target rank to achieve
        gameweeks_remaining: Gameweeks left in season
    
    Returns:
        Strategy recommendation (conservative/balanced/aggressive)
    """
    try:
        strategy = differential_finder.rank_strategy_recommendation(
            current_rank=current_rank,
            target_rank=target_rank,
            gameweeks_remaining=gameweeks_remaining
        )
        
        return convert_numpy_types(strategy)
        
    except Exception as e:
        logger.error(f"Error calculating rank strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/formation-suggestion")
async def suggest_formation(
    squad: List[int],
    include_autosub_analysis: bool = Query(True)
):
    """
    Suggest optimal formation for squad
    
    Args:
        squad: List of 15 player IDs
        include_autosub_analysis: Whether to simulate autosubs
    
    Returns:
        Recommended formation with reasoning and alternatives
    """
    try:
        if len(squad) != 15:
            raise HTTPException(status_code=400, detail="Squad must have exactly 15 players")
        
        players_df = db.get_players_with_stats()
        squad_df = players_df[players_df['id'].isin(squad)]
        
        if len(squad_df) != 15:
            raise HTTPException(status_code=400, detail=f"Found {len(squad_df)}/15 players")
        
        # Suggest formation
        result = formation_optimizer.suggest_formation(
            squad=squad_df,
            include_autosub_analysis=include_autosub_analysis
        )
        
        return convert_numpy_types(result)
        
    except Exception as e:
        logger.error(f"Error suggesting formation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/formation-flexibility")
async def analyze_formation_flexibility(squad: List[int]):
    """
    Analyze formation flexibility of squad
    
    Args:
        squad: List of 15 player IDs
    
    Returns:
        Flexibility analysis with valid formations
    """
    try:
        if len(squad) != 15:
            raise HTTPException(status_code=400, detail="Squad must have exactly 15 players")
        
        players_df = db.get_players_with_stats()
        squad_df = players_df[players_df['id'].isin(squad)]
        
        analysis = formation_optimizer.analyze_formation_flexibility(squad_df)
        
        return convert_numpy_types(analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing formation flexibility: {e}")
        raise HTTPException(status_code=500, detail=str(e))
