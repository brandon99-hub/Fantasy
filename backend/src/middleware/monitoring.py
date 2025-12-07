"""Monitoring and performance tracking middleware"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Track request performance and log slow requests"""
    
    def __init__(self, app, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(duration)
        
        # Log slow requests
        if duration > self.slow_request_threshold:
            logger.warning(
                f"⚠️  Slow request: {request.method} {request.url.path} "
                f"took {duration:.2f}s (threshold: {self.slow_request_threshold}s)"
            )
        else:
            logger.debug(f"{request.method} {request.url.path} - {duration:.3f}s")
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        logger.info(f"→ {request.method} {request.url.path} from {request.client.host}")
        
        try:
            response = await call_next(request)
            logger.info(f"← {request.method} {request.url.path} - {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"✗ {request.method} {request.url.path} - Error: {str(e)}")
            raise


def setup_monitoring(app, slow_threshold: float = 1.0):
    """
    Setup monitoring middleware
    
    Args:
        app: FastAPI application
        slow_threshold: Threshold in seconds for slow request warnings
    """
    app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold=slow_threshold)
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info(f"✅ Monitoring configured (slow threshold: {slow_threshold}s)")
