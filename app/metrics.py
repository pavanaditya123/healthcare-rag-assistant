"""Prometheus metrics. Every pipeline stage is timed separately so a latency
regression can be traced to sparse search, dense search, rerank or the LLM."""
from prometheus_client import Counter, Histogram

STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds", "Latency of each pipeline stage", ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20),
)
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "End-to-end /ask latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20),
)
REQUESTS = Counter("rag_requests_total", "Requests by route taken", ["route"])
CACHE = Counter("rag_cache_total", "Answer cache lookups", ["result"])          # hit / miss / error
LLM_CALLS = Counter("rag_llm_calls_total", "LLM calls", ["model", "outcome"])  # ok / error
DEGRADED = Counter("rag_degraded_total", "Sources-only answers", ["reason"])
RATE_LIMITED = Counter("rag_rate_limited_total", "Requests rejected by rate limiter")
