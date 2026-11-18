"""Redis-backed answer cache and per-client rate limiter.

Both fail OPEN: if Redis is down we skip caching / limiting rather than
taking the whole API down. Redis is an optimisation here, not a dependency
for correctness.
"""
import hashlib
import json
import logging
import time

from app.metrics import CACHE

log = logging.getLogger(__name__)


class MemoryStore:
    """Tiny in-process stand-in for Redis (local dev and tests only:
    not shared between replicas, so not used in docker compose / k8s)."""

    def __init__(self):
        self.data, self.expiry = {}, {}

    def _alive(self, key):
        if key in self.expiry and self.expiry[key] < time.time():
            self.data.pop(key, None), self.expiry.pop(key, None)
        return key in self.data

    def get(self, key):
        return self.data[key] if self._alive(key) else None

    def setex(self, key, ttl, value):
        self.data[key], self.expiry[key] = value, time.time() + ttl

    def incr(self, key):
        self.data[key] = int(self.get(key) or 0) + 1
        return self.data[key]

    def expire(self, key, ttl):
        self.expiry[key] = time.time() + ttl

    def ping(self):
        return True


def make_store(redis_url: str):
    if not redis_url:
        return MemoryStore()
    import redis
    return redis.Redis.from_url(redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)


def cache_key(question: str) -> str:
    normalized = " ".join(question.lower().split())
    return "answer:v1:" + hashlib.sha256(normalized.encode()).hexdigest()


class AnswerCache:
    def __init__(self, store, ttl_s: int):
        self.store, self.ttl_s = store, ttl_s

    def get(self, question: str) -> dict | None:
        if self.ttl_s <= 0:
            return None
        try:
            raw = self.store.get(cache_key(question))
        except Exception as e:  # redis.RedisError, connection refused, ...
            log.warning("cache get failed: %s", e)
            CACHE.labels("error").inc()
            return None
        CACHE.labels("hit" if raw else "miss").inc()
        return json.loads(raw) if raw else None

    def set(self, question: str, value: dict):
        if self.ttl_s <= 0:
            return
        try:
            self.store.setex(cache_key(question), self.ttl_s, json.dumps(value))
        except Exception as e:
            log.warning("cache set failed: %s", e)


class RateLimiter:
    """Fixed window: at most `limit` requests per client per minute.
    INCR + EXPIRE is atomic enough here and easy to reason about."""

    def __init__(self, store, limit_per_min: int):
        self.store, self.limit = store, limit_per_min

    def allow(self, client_id: str) -> bool:
        key = f"ratelimit:{client_id}:{int(time.time() // 60)}"
        try:
            count = self.store.incr(key)
            if count == 1:
                self.store.expire(key, 60)
            return count <= self.limit
        except Exception as e:
            log.warning("rate limiter unavailable, allowing request: %s", e)
            return True
