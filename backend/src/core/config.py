"""Configuration management for FPL AI Optimizer"""

import os
from pathlib import Path
from typing import List
from functools import lru_cache

class Settings:
    """Application settings with environment variable support"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    BACKEND_ROOT: Path = PROJECT_ROOT / "backend"
    DATA_DIR: Path = BACKEND_ROOT / "data"
    LOGS_DIR: Path = BACKEND_ROOT / "logs"
    
    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(DATA_DIR / "fpl_data.db"))
    USE_POSTGRES: bool = os.getenv("USE_POSTGRES", "false").lower() == "true"
    
    # PostgreSQL Manager Index
    POSTGRES_CONNECTION_STRING: str = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql://postgres:Exlifes_69@localhost:5432/myfpl"
    )
    MANAGER_INDEX_ENABLED: bool = os.getenv("MANAGER_INDEX_ENABLED", "true").lower() == "true"
    CRAWLER_BATCH_SIZE: int = int(os.getenv("CRAWLER_BATCH_SIZE", "100"))
    CRAWLER_RATE_LIMIT: float = float(os.getenv("CRAWLER_RATE_LIMIT", "0.3"))  # Seconds between requests per worker (optimized)
    CRAWLER_MAX_WORKERS: int = int(os.getenv("CRAWLER_MAX_WORKERS", "12"))  # Increased for faster syncing
    CRAWLER_HEARTBEAT_INTERVAL: int = int(os.getenv("CRAWLER_HEARTBEAT_INTERVAL", "1000"))
    MANAGER_SEED_LEAGUES: List[int] = [
        int(value) for value in os.getenv(
            "MANAGER_SEED_LEAGUES",
            "131,358346,1635092,1319382,530215,101748"
        ).split(",") if value.strip().isdigit()
    ]
    MANAGER_ON_DEMAND_TTL_MINUTES: int = int(os.getenv("MANAGER_ON_DEMAND_TTL_MINUTES", "60"))
    MANAGER_SEED_REFRESH_HOURS: int = int(os.getenv("MANAGER_SEED_REFRESH_HOURS", "6"))
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    API_LOG_LEVEL: str = os.getenv("API_LOG_LEVEL", "info")
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    
    # FPL API
    FPL_BASE_URL: str = "https://fantasy.premierleague.com/api"
    FPL_REQUEST_TIMEOUT: int = int(os.getenv("FPL_REQUEST_TIMEOUT", "30"))
    FPL_RATE_LIMIT: float = float(os.getenv("FPL_RATE_LIMIT", "1.0"))
    
    # Odds API (TheOddsAPI)
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
    ODDS_API_BASE_URL: str = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
    ODDS_API_REGION: str = os.getenv("ODDS_API_REGION", "uk")
    ODDS_API_MARKETS: str = os.getenv("ODDS_API_MARKETS", "team_totals,totals")
    ODDS_API_BOOKMAKERS: str = os.getenv("ODDS_API_BOOKMAKERS", "betfair,betonlineag")
    
    # Machine Learning
    ML_MODELS_DIR: Path = DATA_DIR / "models"
    ML_TRAIN_TEST_SPLIT: float = 0.2
    ML_RANDOM_STATE: int = 42
    ML_N_ESTIMATORS: int = 200
    ML_LEARNING_RATE: float = 0.1
    ML_MAX_DEPTH: int = 6
    
    # Caching
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    CACHE_METRICS_ENABLED: bool = os.getenv("CACHE_METRICS_ENABLED", "true").lower() == "true"
    
    # Phase 2: xG Integration
    FBREF_ENABLED: bool = os.getenv("FBREF_ENABLED", "true").lower() == "true"
    FOTMOB_ENABLED: bool = os.getenv("FOTMOB_ENABLED", "false").lower() == "true"
    XG_FALLBACK_TO_ESTIMATION: bool = os.getenv("XG_FALLBACK_TO_ESTIMATION", "true").lower() == "true"
    ENABLE_ADVANCED_FEATURES: bool = os.getenv("ENABLE_ADVANCED_FEATURES", "true").lower() == "true"
    ENABLE_DYNAMIC_FDR: bool = os.getenv("ENABLE_DYNAMIC_FDR", "true").lower() == "true"
    AVAILABILITY_FILTER_MODE: str = os.getenv("AVAILABILITY_FILTER_MODE", "conservative")  # conservative, balanced, aggressive
    
    # Phase 3: Celery Background Jobs
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    CELERY_WORKERS: int = int(os.getenv("CELERY_WORKERS", "4"))
    CELERY_TASK_TIME_LIMIT: int = int(os.getenv("CELERY_TASK_TIME_LIMIT", "3600"))  # 1 hour
    
    # OR-Tools Optimization
    ORTOOLS_NUM_WORKERS: int = int(os.getenv("ORTOOLS_NUM_WORKERS", str(os.cpu_count() or 4)))
    ORTOOLS_TIME_LIMIT: float = float(os.getenv("ORTOOLS_TIME_LIMIT", "30.0"))  # seconds
    ORTOOLS_MIP_GAP: float = float(os.getenv("ORTOOLS_MIP_GAP", "0.01"))  # 1%
    ORTOOLS_LOG_PROGRESS: bool = os.getenv("ORTOOLS_LOG_PROGRESS", "false").lower() == "true"
    
    # Logging
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Path = LOGS_DIR / "fpl_optimizer.log"
    
    # Application
    APP_NAME: str = "FPL AI Optimizer"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Advanced Fantasy Premier League optimization with AI"
    
    def __init__(self):
        """Ensure required directories exist"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

