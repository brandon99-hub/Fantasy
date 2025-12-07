"""
Rate limiting middleware for FPL Optimizer API

Implements Redis-backed rate limiting with per-endpoint configuration
and informative rate limit headers.
"""

import time
import logging
from typing import Dict, Optional, Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.src.core.cache import get_cache

logger = logging.getLogger(__name__)


# Rate limit configurations per endpoint pattern
RATE_LIMITS: Dict[str, Dict[str, int]] = {
    # Public endpoints - stricter limits
    "/api/players": {"requests": 60, "window": 60},  # 60 req/min
    "/api/teams": {"requests": 60, "window": 60},
    "/api/players/search": {"requests": 30, "window": 60},  # 30 req/min for search
    
    # Analysis endpoints - moderate limits
    "/api/analysis": {"requests": 30, "window": 60},
    "/api/analysis/fixtures": {"requests": 30, "window": 60},
    "/api/analysis/form": {"requests": 30, "window": 60},
    
    # Optimization endpoints - stricter (CPU intensive)
    "/api/optimize": {"requests": 10, "window": 60},  # 10 req/min
    "/api/optimize/multi-gw": {"requests": 5, "window": 60},  # 5 req/min
    "/api/optimize/transfers": {"requests": 10, "window": 60},
    
    # System endpoints - very strict
    "/api/system/refresh-data": {"requests": 2, "window": 3600},  # 2 req/hour
    "/api/system/train-models": {"requests": 1, "window": 3600},  # 1 req/hour
    
    # Manager endpoints - moderate
    "/api/managers": {"requests": 30, "window": 60},
    
    # Default for unspecified endpoints
    "default": {"requests": 100, "window": 60},  # 100 req/min
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis for distributed rate limiting
    
    Features:
    - Per-endpoint rate limits
    - Per-IP tracking
    - Informative rate limit headers
    - Graceful degradation if Redis unavailable
    """
    
    def __init__(self, app: ASGIApp, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.cache = get_cache()
        logger.info(f"Rate limiting {'enabled' if enabled else 'disabled'}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded IP (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client
        return request.client.host if request.client else "unknown"
    
    def _get_rate_limit_config(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for endpoint"""
        # Try exact match first
        if path in RATE_LIMITS:
            return RATE_LIMITS[path]
        
        # Try prefix match
        for pattern, config in RATE_LIMITS.items():
            if pattern != "default" and path.startswith(pattern):
                return config
        
        # Return default
        return RATE_LIMITS["default"]
    
    def _check_rate_limit(
        self, 
        client_ip: str, 
        endpoint: str, 
        limit: int, 
        window: int
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit
        
        Args:
            client_ip: Client IP address
            endpoint: API endpoint path
            limit: Maximum requests allowed
            window: Time window in seconds
        
        Returns:
            Tuple of (is_allowed, remaining, reset_time)
        """
        if not self.cache.client:
            # If Redis unavailable, allow request (graceful degradation)
            logger.warning("Redis unavailable, rate limiting disabled")
            return True, limit, int(time.time() + window)
        
        # Create rate limit key
        current_time = int(time.time())
        window_start = current_time - (current_time % window)
        key = f"ratelimit:{client_ip}:{endpoint}:{window_start}"
        
        try:
            # Get current count
            count = self.cache.client.get(key)
            current_count = int(count) if count else 0
            
            # Check if limit exceeded
            if current_count >= limit:
                reset_time = window_start + window
                return False, 0, reset_time
            
            # Increment counter
            pipe = self.cache.client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            pipe.execute()
            
            remaining = limit - current_count - 1
            reset_time = window_start + window
            
            return True, remaining, reset_time
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # On error, allow request (graceful degradation)
            return True, limit, int(time.time() + window)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/api/docs", "/api/redoc", "/api/openapi.json", "/"]:
            return await call_next(request)
        
        # Get client IP and endpoint
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        
        # Get rate limit config
        config = self._get_rate_limit_config(endpoint)
        limit = config["requests"]
        window = config["window"]
        
        # Check rate limit
        is_allowed, remaining, reset_time = self._check_rate_limit(
            client_ip, endpoint, limit, window
        )
        
        # If rate limit exceeded, return 429
        if not is_allowed:
            retry_after = reset_time - int(time.time())
            logger.warning(
                f"Rate limit exceeded for {client_ip} on {endpoint} "
                f"(limit: {limit}/{window}s)"
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please try again in {retry_after} seconds.",
                    "limit": limit,
                    "window": window,
                    "retry_after": retry_after
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(retry_after)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


def setup_rate_limiting(app, enabled: bool = True):
    """
    Setup rate limiting middleware
    
    Args:
        app: FastAPI application
        enabled: Whether to enable rate limiting (default: True)
    """
    app.add_middleware(RateLimitMiddleware, enabled=enabled)
    logger.info(f"Rate limiting middleware {'enabled' if enabled else 'disabled'}")


# Decorator for custom rate limits on specific endpoints
def rate_limit(requests: int, window: int = 60):
    """
    Decorator to set custom rate limit for specific endpoint
    
    Usage:
        @router.get("/expensive-endpoint")
        @rate_limit(requests=5, window=60)
        async def expensive_operation():
            return {"result": "data"}
    
    Note: This decorator is for documentation purposes.
    Actual rate limiting is handled by middleware using RATE_LIMITS config.
    """
    def decorator(func):
        # Store rate limit info in function metadata
        func._rate_limit = {"requests": requests, "window": window}
        return func
    return decorator
