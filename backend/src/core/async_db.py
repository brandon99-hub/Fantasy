"""
Async database wrapper for non-blocking database operations

Wraps synchronous database calls in thread pool executor to prevent blocking
the async event loop.
"""

import asyncio
import logging
from typing import Any, Callable, Optional
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for database operations
_db_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="db_worker")


async def run_in_executor(func: Callable, *args, **kwargs) -> Any:
    """
    Run a synchronous function in a thread pool executor
    
    Args:
        func: Synchronous function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result of the function call
    """
    loop = asyncio.get_event_loop()
    
    # If function has kwargs, we need to use a lambda
    if kwargs:
        return await loop.run_in_executor(
            _db_executor,
            lambda: func(*args, **kwargs)
        )
    else:
        return await loop.run_in_executor(
            _db_executor,
            func,
            *args
        )


def async_db_operation(func):
    """
    Decorator to convert synchronous database operations to async
    
    Usage:
        @async_db_operation
        def get_players(self):
            return self.db.get_players_with_stats()
        
        # Can now be called with await
        players = await get_players()
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await run_in_executor(func, *args, **kwargs)
    
    return wrapper


class AsyncDatabaseWrapper:
    """
    Async wrapper for database operations
    
    Wraps any database instance and makes all methods async by running
    them in a thread pool executor.
    """
    
    def __init__(self, db_instance):
        """
        Initialize async wrapper
        
        Args:
            db_instance: Database instance to wrap (FPLDatabase or PostgresManagerDB)
        """
        self._db = db_instance
        self._executor = _db_executor
    
    async def _run_method(self, method_name: str, *args, **kwargs):
        """Run a database method asynchronously"""
        method = getattr(self._db, method_name)
        return await run_in_executor(method, *args, **kwargs)
    
    # Common database methods wrapped as async
    
    async def get_players_with_stats(self):
        """Get all players with stats (async)"""
        return await self._run_method('get_players_with_stats')
    
    async def get_teams(self):
        """Get all teams (async)"""
        return await self._run_method('get_teams')
    
    async def get_fixtures(self, gameweek: Optional[int] = None):
        """Get fixtures (async)"""
        return await self._run_method('get_fixtures', gameweek)
    
    async def get_current_gameweek(self):
        """Get current gameweek (async)"""
        return await self._run_method('get_current_gameweek')
    
    async def get_player_history_data(self, player_id: Optional[int] = None, limit_gws: int = 10):
        """Get player history (async)"""
        return await self._run_method('get_player_history_data', player_id, limit_gws)
    
    async def test_connection(self):
        """Test database connection (async)"""
        return await self._run_method('test_connection')
    
    async def upsert_teams(self, teams_df):
        """Upsert teams (async)"""
        return await self._run_method('upsert_teams', teams_df)
    
    async def upsert_players(self, players_df):
        """Upsert players (async)"""
        return await self._run_method('upsert_players', players_df)
    
    async def upsert_gameweeks(self, gameweeks_df):
        """Upsert gameweeks (async)"""
        return await self._run_method('upsert_gameweeks', gameweeks_df)
    
    async def upsert_fixtures(self, fixtures_df):
        """Upsert fixtures (async)"""
        return await self._run_method('upsert_fixtures', fixtures_df)
    
    async def save_predictions(self, predictions_df, gameweek: int, model_version: str):
        """Save predictions (async)"""
        return await self._run_method('save_predictions', predictions_df, gameweek, model_version)
    
    def __getattr__(self, name):
        """
        Fallback for any method not explicitly wrapped
        Returns an async version of the method
        """
        if name.startswith('_'):
            # Don't wrap private methods
            return getattr(self._db, name)
        
        async def async_method(*args, **kwargs):
            return await self._run_method(name, *args, **kwargs)
        
        return async_method


def get_async_db():
    """
    FastAPI dependency for async database access
    
    Usage in routes:
        @router.get("/players")
        async def get_players(db: AsyncDatabaseWrapper = Depends(get_async_db)):
            players = await db.get_players_with_stats()
            return players
    """
    from backend.src.core.db_factory import get_db
    
    db = get_db()
    return AsyncDatabaseWrapper(db)
