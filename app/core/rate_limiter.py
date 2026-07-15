"""
app/core/rate_limiter.py

Custom in-memory sliding-window rate limiter to protect endpoints from abuse
and prevent runaway costs with external LLM API providers.
"""

import time
import logging
from collections import defaultdict
import threading
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter.
    """
    def __init__(self, rate_limit_requests: int = 60, period_seconds: int = 60):
        self.rate_limit_requests = rate_limit_requests
        self.period_seconds = period_seconds
        self._history = defaultdict(list)
        self._lock = threading.Lock()

    def is_rate_limited(self, key: str) -> bool:
        """
        Registers a request attempt for 'key' and checks if it exceeds the limit.
        """
        now = time.time()
        with self._lock:
            # Clean up old timestamps outside the sliding window
            window_start = now - self.period_seconds
            self._history[key] = [t for t in self._history[key] if t > window_start]
            
            if len(self._history[key]) >= self.rate_limit_requests:
                logger.warning("Rate limit hit for key %r: %d requests in %ds", 
                               key, len(self._history[key]), self.period_seconds)
                return True
            
            self._history[key].append(now)
            return False

    def clear(self):
        """Clears all rate limit history (useful for test isolation)."""
        with self._lock:
            self._history.clear()


from app.core.config import settings


# Pre-defined limiters for production
chat_limiter = RateLimiter(rate_limit_requests=30, period_seconds=60)
upload_limiter = RateLimiter(rate_limit_requests=10, period_seconds=60)


async def check_chat_rate_limit(request: Request):
    """
    FastAPI dependency to rate limit chat queries.
    Uses user_id (hashed API key) if available, falling back to client IP.
    """
    if settings.APP_ENV == "testing":
        return

    # Verify api key dependency runs first and sets request.state.user_id
    user_id = getattr(request.state, "user_id", None) or request.client.host
    if chat_limiter.is_rate_limited(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )


async def check_upload_rate_limit(request: Request):
    """
    FastAPI dependency to rate limit document uploads.
    """
    if settings.APP_ENV == "testing":
        return

    user_id = getattr(request.state, "user_id", None) or request.client.host
    if upload_limiter.is_rate_limited(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Upload rate limit exceeded. Please try again later."
        )
