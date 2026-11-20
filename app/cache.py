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

    def delete(self, key):
        self.data.pop(key, None), self.expiry.pop(key, None)

    def rpush(self, key, value):
        self.data.setdefault(key, []).append(value)

    def ltrim(self, key, start, end):
        items = self.data.get(key, [])
        self.data[key] = items[start:] if end == -1 else items[start:end + 1]

    def lrange(self, key, start, end):
        items = self.data.get(key, [])
        return items[start:] if end == -1 else items[start:end + 1]

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


def cache_key(question: str, layer: str = "answer") -> str:
    normalized = " ".join(question.lower().split())
    return f"{layer}:v1:" + hashlib.sha256(normalized.encode()).hexdigest()


class AnswerCache:
    """Two layers share this class (different key prefix and TTL):
      answer  - the final validated answer: a hit skips retrieval AND the LLM
      context - the reranked passages: a hit skips retrieval (the slow, CPU part)
    """

    def __init__(self, store, ttl_s: int, layer: str = "answer"):
        self.store, self.ttl_s, self.layer = store, ttl_s, layer

    def get(self, question: str) -> dict | None:
        if self.ttl_s <= 0:
            return None
        try:
            raw = self.store.get(cache_key(question, self.layer))
        except Exception as e:  # redis.RedisError, connection refused, ...
            log.warning("cache get failed: %s", e)
            CACHE.labels(self.layer, "error").inc()
            return None
        CACHE.labels(self.layer, "hit" if raw else "miss").inc()
        return json.loads(raw) if raw else None

    def set(self, question: str, value: dict):
        if self.ttl_s <= 0:
            return
        try:
            self.store.setex(cache_key(question, self.layer), self.ttl_s, json.dumps(value))
        except Exception as e:
            log.warning("cache set failed: %s", e)

    def delete(self, question: str):
        try:
            self.store.delete(cache_key(question, self.layer))
        except Exception as e:
            log.warning("cache delete failed: %s", e)


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
