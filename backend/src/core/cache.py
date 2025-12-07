"""
Redis caching service for FPL Optimizer
Provides multi-layer caching with metrics tracking and smart invalidation
"""

import redis
import json
import logging
from typing import Any, Optional, Dict, List
from functools import wraps
from datetime import timedelta
from collections import defaultdict

from backend.src.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# Multi-layer cache TTL strategy (in seconds)
CACHE_TTL_STRATEGY = {
    # Static data - long TTL
    'teams': 86400,  # 24 hours
    'gameweeks': 3600,  # 1 hour
    
    # Semi-static data - medium TTL
    'players': 1800,  # 30 minutes
    'fixtures': 1800,  # 30 minutes
    'fixture_analysis': 1800,  # 30 minutes
    
    # Dynamic data - short TTL
    'predictions': 300,  # 5 minutes
    'player_stats': 300,  # 5 minutes
    'optimization_results': 600,  # 10 minutes
    
    # Real-time data - very short TTL
    'live_scores': 60,  # 1 minute
    'price_changes': 180,  # 3 minutes
    
    # Default
    'default': 300,  # 5 minutes
}


# Cache dependency graph for smart invalidation
CACHE_DEPENDENCIES = {
    'players': ['predictions', 'optimization_results', 'player_stats'],
    'fixtures': ['fixture_analysis', 'predictions'],
    'gameweeks': ['predictions', 'optimization_results'],
    'teams': ['fixture_analysis'],
}


class CacheMetrics:
    """Track cache performance metrics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
    
    def record_hit(self):
        self.hits += 1
    
    def record_miss(self):
        self.misses += 1
    
    def record_set(self):
        self.sets += 1
    
    def record_delete(self):
        self.deletes += 1
    
    def record_error(self):
        self.errors += 1
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all cache statistics"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'errors': self.errors,
            'hit_rate': self.get_hit_rate(),
            'total_requests': self.hits + self.misses
        }
    
    def reset(self):
        """Reset all metrics"""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0


class CacheService:
    """Enhanced Redis-based caching service with multi-layer strategy"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self.client = None
        self.metrics = CacheMetrics()
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.client.ping()
            logger.info("✅ Connected to Redis successfully")
        except Exception as e:
            logger.warning(f"⚠️  Redis connection failed: {e}. Caching disabled.")
            self.client = None
    
    def _get_ttl_for_key(self, key: str) -> int:
        """Get appropriate TTL based on key prefix"""
        for prefix, ttl in CACHE_TTL_STRATEGY.items():
            if key.startswith(prefix):
                return ttl
        return CACHE_TTL_STRATEGY['default']
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                self.metrics.record_hit()
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            self.metrics.record_miss()
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache (batch operation)"""
        if not self.client or not keys:
            return {}
        
        try:
            values = self.client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = json.loads(value)
                    self.metrics.record_hit()
                else:
                    self.metrics.record_miss()
            return result
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache get_many error: {e}")
            return {}
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set value in cache with smart TTL
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (auto-determined if None)
        """
        if not self.client:
            return False
        
        try:
            # Auto-determine TTL if not provided
            if ttl is None:
                ttl = self._get_ttl_for_key(key)
            
            serialized = json.dumps(value, default=str)
            self.client.setex(key, ttl, serialized)
            self.metrics.record_set()
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def set_many(self, mapping: Dict[str, Any], ttl: int = None) -> int:
        """Set multiple values in cache (batch operation)"""
        if not self.client or not mapping:
            return 0
        
        try:
            pipe = self.client.pipeline()
            for key, value in mapping.items():
                key_ttl = ttl if ttl is not None else self._get_ttl_for_key(key)
                serialized = json.dumps(value, default=str)
                pipe.setex(key, key_ttl, serialized)
            pipe.execute()
            count = len(mapping)
            self.metrics.sets += count
            logger.debug(f"Cache SET_MANY: {count} keys")
            return count
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache set_many error: {e}")
            return 0
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.client:
            return False
        
        try:
            self.client.delete(key)
            self.metrics.record_delete()
            logger.debug(f"Cache DELETE: {key}")
            return True
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys from cache (batch operation)"""
        if not self.client or not keys:
            return 0
        
        try:
            count = self.client.delete(*keys)
            self.metrics.deletes += count
            logger.debug(f"Cache DELETE_MANY: {count} keys")
            return count
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache delete_many error: {e}")
            return 0
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern
        
        Args:
            pattern: Redis pattern (e.g., 'players:*', 'predictions:*')
        
        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                count = self.client.delete(*keys)
                self.metrics.deletes += count
                logger.info(f"Cache CLEAR_PATTERN: {pattern} ({count} keys)")
                return count
            return 0
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    def invalidate_with_dependencies(self, key_prefix: str) -> int:
        """
        Smart cache invalidation with dependency tracking
        
        When a key is invalidated, also invalidate all dependent keys
        
        Args:
            key_prefix: Prefix of keys to invalidate (e.g., 'players')
        
        Returns:
            Total number of keys invalidated
        """
        if not self.client:
            return 0
        
        total_deleted = 0
        
        # Invalidate the primary keys
        count = self.clear_pattern(f"{key_prefix}:*")
        total_deleted += count
        
        # Invalidate dependent keys
        if key_prefix in CACHE_DEPENDENCIES:
            for dependent in CACHE_DEPENDENCIES[key_prefix]:
                count = self.clear_pattern(f"{dependent}:*")
                total_deleted += count
                logger.debug(f"Invalidated dependent cache: {dependent}")
        
        logger.info(f"Smart invalidation: {key_prefix} + dependencies ({total_deleted} keys)")
        return total_deleted
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.client:
            return False
        
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get remaining TTL for key in seconds"""
        if not self.client:
            return -1
        
        try:
            return self.client.ttl(key)
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache TTL error for key {key}: {e}")
            return -1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        return self.metrics.get_stats()
    
    def reset_metrics(self):
        """Reset cache metrics"""
        self.metrics.reset()
        logger.info("Cache metrics reset")


# Global cache instance
_cache_instance = None


def get_cache() -> CacheService:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance


def cache_result(ttl: int = None, key_prefix: str = ""):
    """
    Decorator to cache function results with smart TTL
    Supports both sync and async functions
    
    Args:
        ttl: Time to live in seconds (auto-determined if None)
        key_prefix: Prefix for cache key
    
    Usage:
        @cache_result(key_prefix="players")
        async def get_players():
            return await expensive_operation()
    """
    def decorator(func):
        import asyncio
        import inspect
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache = get_cache()
                
                # Generate cache key from function name and arguments
                key_parts = [key_prefix or func.__name__]
                if args:
                    key_parts.extend([str(arg) for arg in args])
                if kwargs:
                    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                
                cache_key = ":".join(key_parts)
                
                # Try to get from cache
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Execute async function and cache result
                result = await func(*args, **kwargs)
                cache.set(cache_key, result, ttl=ttl)
                
                return result
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache = get_cache()
                
                # Generate cache key from function name and arguments
                key_parts = [key_prefix or func.__name__]
                if args:
                    key_parts.extend([str(arg) for arg in args])
                if kwargs:
                    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                
                cache_key = ":".join(key_parts)
                
                # Try to get from cache
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                cache.set(cache_key, result, ttl=ttl)
                
                return result
            
            return sync_wrapper
    
    return decorator


def invalidate_cache(key_prefix: str, include_dependencies: bool = True):
    """
    Invalidate cache for a given prefix
    
    Args:
        key_prefix: Prefix of keys to invalidate
        include_dependencies: Also invalidate dependent caches
    """
    cache = get_cache()
    if include_dependencies:
        return cache.invalidate_with_dependencies(key_prefix)
    else:
        return cache.clear_pattern(f"{key_prefix}:*")
