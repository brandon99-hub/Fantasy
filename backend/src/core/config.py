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
    
    # Machine Learning
    ML_MODELS_DIR: Path = DATA_DIR / "models"
    ML_TRAIN_TEST_SPLIT: float = 0.2
    ML_RANDOM_STATE: int = 42
    ML_N_ESTIMATORS: int = 200
    ML_LEARNING_RATE: float = 0.1
    ML_MAX_DEPTH: int = 6
    
    # Caching
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
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

