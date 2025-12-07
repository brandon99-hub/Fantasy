"""
Database factory for unified database access

Provides a factory pattern to switch between SQLite and PostgreSQL databases
based on configuration, with a unified interface for both.
"""

import logging
from typing import Union, Optional
from contextlib import contextmanager
from dotenv import load_dotenv

from backend.src.core.config import get_settings
from backend.src.core.database import FPLDatabase
from backend.src.core.postgres_db import PostgresManagerDB

settings = get_settings()
logger = logging.getLogger(__name__)


class DatabaseFactory:
    """Factory for creating database instances"""
    
    _instance = None
    _use_postgres = None
    
    def __init__(self, use_postgres: Optional[bool] = None):
        """
        Initialize database factory
        
        Args:
            use_postgres: Force PostgreSQL (True) or SQLite (False). 
                         If None, uses environment variable.
        """
        if use_postgres is not None:
            self._use_postgres = use_postgres
        else:
            # Load .env file first
            load_dotenv()
            # Check environment variable or default to False
            import os
            self._use_postgres = os.getenv("USE_POSTGRES", "false").lower() == "true"
        
        logger.info(f"Database factory initialized: {'PostgreSQL' if self._use_postgres else 'SQLite'}")
    
    @classmethod
    def get_instance(cls, use_postgres: Optional[bool] = None):
        """Get singleton instance of database factory"""
        if cls._instance is None:
            cls._instance = cls(use_postgres)
        return cls._instance
    
    def get_db(self) -> Union[FPLDatabase, PostgresManagerDB]:
        """
        Get database instance based on configuration
        
        Returns:
            Database instance (either FPLDatabase or PostgresManagerDB)
        """
        if self._use_postgres:
            return PostgresManagerDB()
        else:
            return FPLDatabase()
    
    @contextmanager
    def get_connection(self):
        """
        Get database connection with automatic cleanup
        
        Yields:
            Database connection
        """
        db = self.get_db()
        
        if self._use_postgres:
            # PostgreSQL uses context manager
            with db.get_connection() as conn:
                yield conn
        else:
            # SQLite uses get_connection method
            conn = db.get_connection()
            try:
                yield conn
            finally:
                conn.close()
    
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL"""
        return self._use_postgres
    
    def is_sqlite(self) -> bool:
        """Check if using SQLite"""
        return not self._use_postgres


# Global factory instance
_factory = None


def get_db_factory(use_postgres: Optional[bool] = None) -> DatabaseFactory:
    """
    Get global database factory instance
    
    Args:
        use_postgres: Force PostgreSQL (True) or SQLite (False).
                     If None, uses environment variable.
    
    Returns:
        DatabaseFactory instance
    """
    global _factory
    if _factory is None:
        _factory = DatabaseFactory(use_postgres)
    return _factory


def get_db() -> Union[FPLDatabase, PostgresManagerDB]:
    """
    Get database instance (convenience function)
    
    Returns:
        Database instance based on configuration
    """
    factory = get_db_factory()
    return factory.get_db()


# Async database dependency for FastAPI
async def get_async_db():
    """
    FastAPI dependency for async database access
    
    Yields:
        Database instance
    """
    db = get_db()
    try:
        yield db
    finally:
        # Cleanup if needed
        pass
