"""
FastAPI backend for FPL AI Optimizer
Main application entry point with route registration
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from backend.src.core.config import get_settings
from backend.src.core.database import FPLDatabase
from backend.src.core.data_collector import FPLDataCollector

# Import route modules
from backend.src.api.routes import players, teams, analysis, managers, system

# Get settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    # Startup
    logger.info("🚀 Starting FPL AI Assistant...")
    
    # Check if we have data
    try:
        db = FPLDatabase()
        data_collector = FPLDataCollector()
        
        players_df = db.get_players_with_stats()
        if players_df.empty:
            logger.info("📊 No data found. Updating from FPL API...")
            success = data_collector.update_all_data()
            if success:
                logger.info("✅ Data updated successfully!")
            else:
                logger.warning("❌ Data update failed")
        else:
            logger.info(f"✅ Found {len(players_df)} players in database")
    except Exception as e:
        logger.error(f"❌ Error checking/updating data: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down FPL AI Assistant...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(players.router)
app.include_router(teams.router)
app.include_router(analysis.router)
app.include_router(managers.router)
app.include_router(system.router)

# Add health check at root
@app.get("/health")
async def root_health_check():
    """Root health check endpoint"""
    return {"status": "healthy", "message": "FPL AI Optimizer API is running", "version": settings.APP_VERSION}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to FPL AI Optimizer API",
        "version": settings.APP_VERSION,
        "docs": "/api/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.API_LOG_LEVEL
    )

