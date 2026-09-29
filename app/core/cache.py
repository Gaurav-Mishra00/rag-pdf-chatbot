"""Thread-safe in-memory LRU query cache with TTL expiration and user isolation."""
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from app.core.config import settings


class QueryCache:
    """Thread-safe LRU cache storing normalized query results per user."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Any, str]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(user_id: str, query: str) -> str:
        normalized = " ".join(query.strip().lower().split())
        return hashlib.sha256(f"{user_id}:{normalized}".encode("utf-8")).hexdigest()

    def get(self, user_id: str, query: str) -> Optional[Any]:
        if not settings.QUERY_CACHE_ENABLED:
            return None

        key = self._make_key(user_id, query)
        now = time.time()

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            timestamp, value, cached_user = self._cache[key]
            if cached_user != user_id or (now - timestamp) > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, user_id: str, query: str, value: Any) -> None:
        if not settings.QUERY_CACHE_ENABLED:
            return

        key = self._make_key(user_id, query)
        now = time.time()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                # Evict oldest
                self._cache.popitem(last=False)

            self._cache[key] = (now, value, user_id)

    def clear_for_user(self, user_id: str) -> int:
        """Evicts all cached queries belonging to a specific user (e.g. after uploading or deleting docs)."""
        removed = 0
        with self._lock:
            keys_to_remove = [k for k, (_, _, uid) in self._cache.items() if uid == user_id]
            for k in keys_to_remove:
                del self._cache[k]
                removed += 1
        return removed

    def clear_all(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = round(self._hits / total, 3) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
            }


# Singleton query cache instance
query_cache = QueryCache(
    max_size=settings.QUERY_CACHE_MAX_SIZE,
    ttl_seconds=settings.QUERY_CACHE_TTL_SECONDS,
)
