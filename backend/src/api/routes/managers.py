"""Manager search and team retrieval API endpoints"""

import logging
import threading
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List

from backend.src.services.fpl_api_client import FPLAPIClient
from backend.src.core.postgres_db import PostgresManagerDB
from backend.src.core.config import get_settings
from backend.src.services.manager_crawler import ManagerCrawler
from backend.src.utils.serialization import convert_numpy_types
from backend.src.schemas.models import (
    ManagerSearchResponse, ManagerTeamResponse,
    CrawlStartRequest, CrawlStatusResponse, CrawlControlResponse,
    LeagueCrawlRequest, SeedLeaguesRequest
)

router = APIRouter(prefix="/api", tags=["managers"])
fpl_api_client = FPLAPIClient()
settings = get_settings()
logger = logging.getLogger(__name__)

# Global crawler instance (singleton pattern)
_crawler_instance: Optional[ManagerCrawler] = None

def get_crawler() -> ManagerCrawler:
    """Get or create crawler instance"""
    global _crawler_instance
    if _crawler_instance is None:
        _crawler_instance = ManagerCrawler()
    return _crawler_instance


@router.get("/managers/search", response_model=ManagerSearchResponse)
async def search_managers(
    name: str = Query(..., description="Manager name to search"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    use_index: bool = Query(True, description="Use PostgreSQL index")
):
    """Search for FPL managers by name using PostgreSQL index with fallback"""
    try:
        managers = []
        use_postgres = use_index and settings.MANAGER_INDEX_ENABLED
        
        if use_postgres:
            try:
                db = PostgresManagerDB()
                db_results = db.search_managers(name, limit=limit)
                
                # Convert to expected format
                for result in db_results:
                    managers.append({
                        'id': str(result['manager_id']),
                        'player_name': result.get('player_name', ''),
                        'team_name': result.get('team_name', ''),
                        'overall_rank': result.get('overall_rank'),
                        'total_points': result.get('total_points')
                    })
            except Exception as e:
                # Fallback to FPL API if PostgreSQL fails
                use_postgres = False
        
        # Fallback to FPL API if index is disabled or empty
        if not use_postgres or not managers:
            fpl_results = fpl_api_client.search_managers(name)
            # Deduplicate by manager ID
            seen_ids = {m['id'] for m in managers}
            for result in fpl_results:
                if result.get('id') not in seen_ids:
                    managers.append(result)
                    seen_ids.add(result.get('id'))
        
        # Limit results
        managers = managers[:limit]
        
        return convert_numpy_types({
            "managers": managers,
            "message": "Manager search results" if managers else f"No managers found for '{name}'"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching managers: {str(e)}")


@router.get("/team/manager/{manager_id}", response_model=ManagerTeamResponse)
async def get_team_by_manager(
    manager_id: str,
    force_refresh: bool = Query(False, description="Force fresh fetch from FPL API")
):
    """Get team by manager ID using FPL API with cache control"""
    try:
        if not manager_id.isdigit():
            raise HTTPException(status_code=400, detail="Manager ID must be numeric")
        
        team_data = fpl_api_client.get_manager_team(manager_id)
        
        if not team_data:
            raise HTTPException(
                status_code=404, 
                detail="Manager not found or team data unavailable from FPL API"
            )
        
        crawler = get_crawler()
        def hydrate_manager():
            try:
                crawler.crawl_manager(int(manager_id))
            except Exception as exc:
                logger.warning(f"Hydration failed for manager {manager_id}: {exc}")
        threading.Thread(target=hydrate_manager, daemon=True).start()
        
        # Return with cache-control headers to prevent stale data
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=convert_numpy_types(team_data),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching team: {str(e)}")


# Crawler management endpoints
@router.post("/managers/crawl/start", response_model=CrawlControlResponse)
async def start_crawl(request: CrawlStartRequest, background_tasks: BackgroundTasks):
    """Start manager index crawl"""
    try:
        crawler = get_crawler()
        
        # Check if already running
        status = crawler.get_status()
        if status['running'] and not status['paused']:
            return CrawlControlResponse(
                success=False,
                message="Crawler is already running"
            )
        
        # Start crawl in background thread
        def run_crawl():
            try:
                crawler.crawl_range(
                    start_id=request.start_id,
                    end_id=request.end_id,
                    batch_size=request.batch_size
                )
            except Exception as e:
                logger.error(f"Background crawl error: {str(e)}", exc_info=True)
        
        import threading
        thread = threading.Thread(target=run_crawl, daemon=True)
        thread.start()
        
        return CrawlControlResponse(
            success=True,
            message=f"Crawl started from ID {request.start_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting crawl: {str(e)}")


@router.post("/managers/crawl/resume", response_model=CrawlControlResponse)
async def resume_crawl(background_tasks: BackgroundTasks):
    """Resume interrupted crawl"""
    try:
        crawler = get_crawler()
        
        def run_resume():
            try:
                crawler.resume_crawl()
            except Exception as e:
                logger.error(f"Background resume error: {str(e)}", exc_info=True)
        
        import threading
        thread = threading.Thread(target=run_resume, daemon=True)
        thread.start()
        
        return CrawlControlResponse(
            success=True,
            message="Crawl resumed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming crawl: {str(e)}")


@router.post("/managers/crawl/pause", response_model=CrawlControlResponse)
async def pause_crawl():
    """Pause the crawler"""
    try:
        crawler = get_crawler()
        crawler.pause()
        return CrawlControlResponse(
            success=True,
            message="Crawler paused"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pausing crawl: {str(e)}")


@router.post("/managers/crawl/stop", response_model=CrawlControlResponse)
async def stop_crawl():
    """Stop the crawler"""
    try:
        crawler = get_crawler()
        crawler.stop()
        return CrawlControlResponse(
            success=True,
            message="Crawler stopped"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping crawl: {str(e)}")


@router.get("/managers/crawl/status", response_model=CrawlStatusResponse)
async def get_crawl_status():
    """Get crawler status"""
    try:
        crawler = get_crawler()
        status = crawler.get_status()
        
        return CrawlStatusResponse(
            running=status['running'],
            paused=status['paused'],
            progress=status['progress'],
            manager_count=status['manager_count'],
            message="Crawler is running" if status['running'] else "Crawler is stopped"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting crawl status: {str(e)}")


@router.post("/managers/crawl/initialize", response_model=CrawlControlResponse)
async def initialize_database():
    """Initialize PostgreSQL database schema"""
    try:
        db = PostgresManagerDB()
        
        # Test connection first
        if not db.test_connection():
            return CrawlControlResponse(
                success=False,
                message="Failed to connect to PostgreSQL database"
            )
        
        # Initialize schema
        if db.initialize_schema():
            return CrawlControlResponse(
                success=True,
                message="Database schema initialized successfully"
            )
        else:
            return CrawlControlResponse(
                success=False,
                message="Failed to initialize database schema"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing database: {str(e)}")


@router.post("/managers/crawl/league/start", response_model=CrawlControlResponse)
async def start_league_crawl(request: LeagueCrawlRequest):
    """Start crawling a specific classic league"""
    try:
        crawler = get_crawler()
        
        def run_league():
            try:
                crawler.crawl_league(
                    league_id=request.league_id,
                    max_pages=request.max_pages,
                    start_page=request.start_page
                )
            except Exception as exc:
                logger.error(f"League crawl error: {exc}", exc_info=True)
        
        thread = threading.Thread(target=run_league, daemon=True)
        thread.start()
        return CrawlControlResponse(success=True, message=f"League crawl started for {request.league_id}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error starting league crawl: {str(exc)}")


@router.post("/managers/crawl/league/resume", response_model=CrawlControlResponse)
async def resume_league_crawl(league_id: Optional[int] = None):
    """Resume a paused league crawl"""
    try:
        crawler = get_crawler()
        
        def run_resume():
            try:
                crawler.resume_league_crawl(league_id=league_id)
            except Exception as exc:
                logger.error(f"League resume error: {exc}", exc_info=True)
        
        thread = threading.Thread(target=run_resume, daemon=True)
        thread.start()
        return CrawlControlResponse(success=True, message="League crawl resume requested")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error resuming league crawl: {str(exc)}")


@router.post("/managers/crawl/seed", response_model=CrawlControlResponse)
async def seed_leagues(request: SeedLeaguesRequest):
    """Kick off a crawl for configured seed leagues"""
    try:
        crawler = get_crawler()
        league_ids = request.league_ids or settings.MANAGER_SEED_LEAGUES
        
        if not league_ids:
            return CrawlControlResponse(success=False, message="No league IDs configured")
        
        def run_seed():
            try:
                crawler.crawl_seed_leagues(league_ids=league_ids, max_pages=request.max_pages)
            except Exception as exc:
                logger.error(f"Seed crawl error: {exc}", exc_info=True)
        
        thread = threading.Thread(target=run_seed, daemon=True)
        thread.start()
        return CrawlControlResponse(success=True, message=f"Seed crawl started for {len(league_ids)} leagues")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error starting seed crawl: {str(exc)}")


@router.get("/managers/league/{league_id}", response_model=ManagerSearchResponse)
async def get_league_managers(
    league_id: int,
    limit: int = Query(1000, ge=1, le=5000)
):
    """Get cached managers for a specific league"""
    try:
        db = PostgresManagerDB()
        managers = db.get_managers_in_league(league_id, limit=limit)
        formatted = [
            {
                'id': str(item['manager_id']),
                'player_name': item.get('player_name'),
                'team_name': item.get('team_name'),
                'overall_rank': item.get('overall_rank'),
                'total_points': item.get('league_points'),
                'league_rank': item.get('league_rank')
            }
            for item in managers
        ]
        message = f"{len(formatted)} managers retrieved for league {league_id}"
        return ManagerSearchResponse(managers=formatted, message=message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error fetching league managers: {str(exc)}")

