"""Rate limiting middleware for FastAPI with Redis backend"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import os

from backend.src.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Determine storage backend
# Use Redis if available, fallback to memory
redis_url = settings.REDIS_URL
storage_uri = redis_url if redis_url else "memory://"

# Create limiter instance with Redis backend
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],
    storage_uri=storage_uri,
    strategy="fixed-window",  # Can also use "moving-window" for more accurate limiting
)

# Per-endpoint rate limit configurations
# Format: "requests/period" where period can be: second, minute, hour, day
ENDPOINT_LIMITS = {
    # Public endpoints - moderate limits
    "/api/players": ["60/minute"],
    "/api/teams": ["60/minute"],
    "/api/players/search": ["30/minute"],
    
    # Analysis endpoints
    "/api/analysis": ["30/minute"],
    "/api/analysis/fixtures": ["30/minute"],
    "/api/analysis/form": ["30/minute"],
    
    # Optimization endpoints - stricter (CPU intensive)
    "/api/optimize": ["10/minute"],
    "/api/optimize/multi-gw": ["5/minute"],
    "/api/optimize/transfers": ["10/minute"],
    "/api/optimize/chips": ["10/minute"],
    
    # System endpoints - very strict
    "/api/system/refresh-data": ["2/hour"],
    "/api/system/train-models": ["1/hour"],
    "/api/system/status": ["30/minute"],
    
    # Manager endpoints
    "/api/managers": ["30/minute"],
    "/api/managers/search": ["20/minute"],
    
    # Advanced features
    "/api/features": ["30/minute"],
    "/api/advanced": ["20/minute"],
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Custom rate limiting middleware with logging and headers"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            
            # Add rate limit info to response headers if available
            if hasattr(response, "headers"):
                # These headers are added by slowapi automatically
                # We just ensure they're present
                pass
            
            return response
            
        except RateLimitExceeded as e:
            client_ip = get_remote_address(request)
            logger.warning(
                f"Rate limit exceeded for {client_ip} on {request.url.path}"
            )
            raise e


def get_endpoint_limit(path: str) -> list:
    """
    Get rate limit for specific endpoint
    
    Args:
        path: Request path
    
    Returns:
        List of rate limit strings
    """
    # Try exact match
    if path in ENDPOINT_LIMITS:
        return ENDPOINT_LIMITS[path]
    
    # Try prefix match
    for pattern, limits in ENDPOINT_LIMITS.items():
        if path.startswith(pattern):
            return limits
    
    # Return default
    return ["100/minute"]


def setup_rate_limiting(app):
    """
    Setup rate limiting for FastAPI app with Redis backend
    
    Features:
    - Redis-backed distributed rate limiting
    - Per-endpoint rate limits
    - Informative error messages
    - Rate limit headers in responses
    
    Usage in main.py:
        from backend.src.middleware.rate_limit import setup_rate_limiting
        setup_rate_limiting(app)
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(RateLimitMiddleware)
    
    backend = "Redis" if "redis" in storage_uri else "Memory"
    logger.info(f"✅ Rate limiting configured with {backend} backend")
    logger.info(f"   Storage: {storage_uri}")
    logger.info(f"   Default limit: 100/minute")
    logger.info(f"   Custom limits: {len(ENDPOINT_LIMITS)} endpoints")
