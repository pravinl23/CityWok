import os
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

try:
    import redis
    REDIS_AVAILABLE = True
    REDIS_URL = os.getenv("REDIS_URL")
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        redis_client = None
        REDIS_AVAILABLE = False
except ImportError:
        REDIS_AVAILABLE = False
        redis_client = None

memory_storage = defaultdict(list)


def check_rate_limit_redis(
    ip: str, 
    limit: int = 10, 
    window: int = 3600
) -> bool:
    if REDIS_AVAILABLE and redis_client:
        return _check_redis(ip, limit, window)
    else:
        return _check_memory(ip, limit, window)


def _check_redis(ip: str, limit: int, window: int) -> bool:
    try:
        key = f"ratelimit:{ip}"
        current = redis_client.get(key)
        
        if current and int(current) >= limit:
            return False
        
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        pipe.execute()
        
        return True
    
    except Exception:
        return _check_memory(ip, limit, window)


def _check_memory(ip: str, limit: int, window: int) -> bool:
    now = datetime.now()
    cutoff = now - timedelta(seconds=window)
    
    memory_storage[ip] = [
        req_time for req_time in memory_storage[ip] 
        if req_time > cutoff
    ]
    
    if len(memory_storage[ip]) >= limit:
        return False
    
    memory_storage[ip].append(now)
    return True


def get_rate_limit_info(ip: str) -> dict:
    if REDIS_AVAILABLE and redis_client:
        try:
            key = f"ratelimit:{ip}"
            current = redis_client.get(key)
            ttl = redis_client.ttl(key)
            
            return {
                "requests_used": int(current) if current else 0,
                "reset_in_seconds": ttl if ttl > 0 else 0,
                "using_redis": True
            }
        except Exception:
            pass
    
    now = datetime.now()
    cutoff = now - timedelta(seconds=3600)
    
    recent_requests = [
        req_time for req_time in memory_storage[ip]
        if req_time > cutoff
    ]
    
    return {
        "requests_used": len(recent_requests),
        "reset_in_seconds": 3600,
        "using_redis": False
    }

