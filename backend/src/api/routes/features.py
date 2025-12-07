"""
API routes for newly implemented features
xG analysis, injury risk, chip strategy, and set pieces
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from backend.src.core.database import FPLDatabase
from backend.src.utils.xg_integrator import XGIntegrator
from backend.src.models.injury_risk_predictor import InjuryRiskPredictor
from backend.src.core.chip_strategy_optimizer import ChipStrategyOptimizer
from backend.src.utils.set_piece_collector import SetPieceCollector
from backend.src.utils.serialization import convert_numpy_types
from backend.src.core.cache import cache_result

router = APIRouter(prefix="/api/features", tags=["features"])
logger = logging.getLogger(__name__)

db = FPLDatabase()
xg_integrator = XGIntegrator()
injury_predictor = InjuryRiskPredictor()
chip_optimizer = ChipStrategyOptimizer()
set_piece_collector = SetPieceCollector()


@router.get("/xg-analysis")
@cache_result(ttl=3600, key_prefix="xg_analysis")
async def get_xg_analysis():
    """Get xG/xA analysis for all players"""
    try:
        players_df = db.get_players_with_stats()
        
        if players_df.empty:
            raise HTTPException(status_code=404, detail="No player data available")
        
        # Enrich with xG data
        players_df = xg_integrator.enrich_players_with_xg(players_df)
        players_df = xg_integrator.calculate_xg_features(players_df)
        
        # Get summary
        summary = xg_integrator.get_xg_summary(players_df)
        
        return convert_numpy_types(summary)
        
    except Exception as e:
        logger.error(f"Error in xG analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/xg-overperformers")
@cache_result(ttl=3600, key_prefix="xg_overperformers")
async def get_xg_overperformers(threshold: float = Query(2.0, ge=0.0)):
    """Get players over/underperforming their xG"""
    try:
        players_df = db.get_players_with_stats()
        players_df = xg_integrator.enrich_players_with_xg(players_df)
        
        performers = xg_integrator.identify_xg_overperformers(players_df, threshold)
        
        return {
            'overperformers': convert_numpy_types(performers['overperformers'].head(20).to_dict('records')),
            'underperformers': convert_numpy_types(performers['underperformers'].head(20).to_dict('records'))
        }
        
    except Exception as e:
        logger.error(f"Error finding xG performers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/injury-risk/{player_id}")
@cache_result(ttl=1800, key_prefix="injury_risk")
async def get_injury_risk(player_id: int):
    """Get injury risk assessment for a player"""
    try:
        player_history = db.get_player_history(player_id)
        
        if player_history.empty:
            raise HTTPException(status_code=404, detail=f"No history found for player {player_id}")
        
        risk = injury_predictor.predict_injury_risk(player_history)
        
        return convert_numpy_types(risk)
        
    except Exception as e:
        logger.error(f"Error predicting injury risk for player {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/injury-risk-all")
@cache_result(ttl=3600, key_prefix="injury_risk_all")
async def get_all_injury_risks(min_risk: float = Query(0.0, ge=0.0, le=1.0)):
    """Get injury risk for all players"""
    try:
        players_df = db.get_players_with_stats()
        history_df = db.get_all_player_history()
        
        # Batch predict
        players_df = injury_predictor.batch_predict_injury_risk(players_df, history_df)
        
        # Filter by minimum risk
        high_risk = players_df[players_df['injury_risk_score'] >= min_risk]
        
        return convert_numpy_types(
            high_risk[['web_name', 'team_name', 'injury_risk_score', 'injury_risk_level', 
                       'injury_recommendation', 'injury_risk_factors']]
            .sort_values('injury_risk_score', ascending=False)
            .head(50)
            .to_dict('records')
        )
        
    except Exception as e:
        logger.error(f"Error getting all injury risks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chip-strategy")
@cache_result(ttl=7200, key_prefix="chip_strategy")
async def get_chip_strategy(
    current_gameweek: int = Query(..., ge=1, le=38),
    used_chips: Optional[List[str]] = Query(None)
):
    """Get optimal chip usage strategy"""
    try:
        fixtures_df = db.get_fixtures()
        players_df = db.get_players_with_stats()
        
        if fixtures_df.empty:
            raise HTTPException(status_code=404, detail="No fixture data available")
        
        strategy = chip_optimizer.create_season_chip_plan(
            fixtures_df=fixtures_df,
            players_df=players_df,
            current_gameweek=current_gameweek,
            used_chips=used_chips or []
        )
        
        return convert_numpy_types(strategy)
        
    except Exception as e:
        logger.error(f"Error creating chip strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chip-combinations")
async def get_chip_combinations(current_gameweek: int = Query(..., ge=1, le=38)):
    """Get chip combination synergies"""
    try:
        fixtures_df = db.get_fixtures()
        
        combinations = chip_optimizer.evaluate_chip_combinations(fixtures_df, current_gameweek)
        
        return convert_numpy_types(combinations)
        
    except Exception as e:
        logger.error(f"Error evaluating chip combinations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/set-piece-takers")
@cache_result(ttl=3600, key_prefix="set_piece_takers")
async def get_set_piece_takers(
    team: Optional[str] = Query(None),
    type: Optional[str] = Query(None, regex="^(penalties|corners|freekicks)$")
):
    """Get set piece takers with optional filtering"""
    try:
        players_df = db.get_players_with_stats()
        players_df = set_piece_collector.update_set_piece_data(players_df)
        
        takers = set_piece_collector.get_set_piece_takers(
            players_df,
            team_name=team,
            set_piece_type=type
        )
        
        return convert_numpy_types(takers.to_dict('records'))
        
    except Exception as e:
        logger.error(f"Error getting set piece takers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/set-piece-analysis")
@cache_result(ttl=3600, key_prefix="set_piece_analysis")
async def get_set_piece_analysis():
    """Get analysis of set piece takers' value"""
    try:
        players_df = db.get_players_with_stats()
        players_df = set_piece_collector.update_set_piece_data(players_df)
        
        analysis = set_piece_collector.analyze_set_piece_value(players_df)
        
        return convert_numpy_types(analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing set pieces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/set-piece-by-team")
@cache_result(ttl=3600, key_prefix="set_piece_by_team")
async def get_set_piece_by_team():
    """Get set piece takers summary by team"""
    try:
        players_df = db.get_players_with_stats()
        players_df = set_piece_collector.update_set_piece_data(players_df)
        
        summary = set_piece_collector.get_team_set_piece_summary(players_df)
        
        return convert_numpy_types(summary.to_dict('records'))
        
    except Exception as e:
        logger.error(f"Error getting team set piece summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
