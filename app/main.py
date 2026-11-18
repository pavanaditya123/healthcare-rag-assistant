"""Stateless FastAPI service. All state lives in Qdrant (index) and Redis
(cache, rate-limit counters), so any number of replicas can run behind a
load balancer and the HPA can add or remove pods freely.

    uvicorn app.main:app --port 8000
"""
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator

from app.answer import Assistant
from app.cache import AnswerCache, RateLimiter, make_store
from app.config import Settings, settings
from app.metrics import RATE_LIMITED, REQUEST_LATENCY, REQUESTS
from app.safety import redact

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")


SOURCES = {"GHR", "GARD", "NIDDK", "NINDS", "MPlusHealthTopics", "NIHSeniorHealth", "CancerGov", "NHLBI", "CDC"}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    # Optional metadata filters, e.g. {"sources": ["GHR"], "qtypes": ["treatment"]}
    sources: list[str] | None = None
    qtypes: list[str] | None = None

    @field_validator("sources")
    @classmethod
    def known_sources(cls, v):
        unknown = set(v or []) - SOURCES
        if unknown:
            raise ValueError(f"unknown sources {sorted(unknown)}; choose from {sorted(SOURCES)}")
        return v


def create_app(assistant: Assistant | None = None, store=None, s: Settings = settings) -> FastAPI:
    """Dependencies can be injected so tests run without models or Redis."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.assistant is None:  # real startup: load models once per process
            from app.llm import make_llm
            from app.retrieval import Encoders, HybridRetriever, make_client

            retriever = HybridRetriever(make_client(s), Encoders(s), s)
            app.state.assistant = Assistant(retriever, make_llm(s), s)
            # Warm-up: the first ONNX run is slow (graph init, memory allocation),
            # so pay that cost before readiness instead of on a user's request.
            if retriever.is_ready():
                retriever.search("what is diabetes")
                log.info("warm-up query done")
        yield

    app = FastAPI(title="Healthcare RAG Assistant", lifespan=lifespan)
    app.state.assistant = assistant
    app.state.store = store if store is not None else make_store(s.redis_url)
    app.state.cache = AnswerCache(app.state.store, s.cache_ttl_s)
    app.state.limiter = RateLimiter(app.state.store, s.rate_limit_per_min)

    def prepare(body: AskRequest, request: Request) -> tuple[str, str]:
        """Rate limit, then redact. Returns (redacted question, cache key)."""
        client = request.client.host if request.client else "unknown"
        if not app.state.limiter.allow(client):
            RATE_LIMITED.inc()
            raise HTTPException(429, "Too many requests, try again in a minute.")
        question = redact(body.question)  # nothing below this line sees raw PHI
        # Filters are part of the cache key: same question + different filter = different answer.
        return question, f"{question}|{sorted(body.sources or [])}|{sorted(body.qtypes or [])}"

    def finish(result: dict, cache_q: str, question: str, start: float, cached: bool):
        if not cached and result["route"] != "degraded":  # don't cache outages; retry the LLM next time
            app.state.cache.set(cache_q, result)
        REQUESTS.labels(result["route"]).inc()
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.observe(elapsed)
        log.info("ask route=%s cached=%s ms=%.0f q=%r", result["route"], cached, elapsed * 1000, question)

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        # Plain `def`: FastAPI runs it in a thread pool, so blocking model and
        # HTTP calls don't block the event loop.
        start = time.perf_counter()
        question, cache_q = prepare(body, request)
        cached = app.state.cache.get(cache_q)
        result = cached or app.state.assistant.answer(question, sources=body.sources, qtypes=body.qtypes).to_dict()
        finish(result, cache_q, question, start, cached=bool(cached))
        return {**result, "cached": bool(cached)}

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, request: Request):
        """Server-sent events: sources first, then tokens, then a validated "done".
        The user sees text after retrieval + first token instead of the full answer."""
        start = time.perf_counter()
        question, cache_q = prepare(body, request)
        cached = app.state.cache.get(cache_q)

        def events():
            if cached:
                final = {"type": "done", **cached}
                yield f"data: {json.dumps({**final, 'cached': True})}\n\n"
            else:
                for event in app.state.assistant.stream(question, sources=body.sources, qtypes=body.qtypes):
                    if event["type"] == "done":
                        final = event
                    yield f"data: {json.dumps(event)}\n\n"
            finish({k: v for k, v in final.items() if k not in ("type", "cached")}, cache_q, question, start, bool(cached))

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/healthz")
    def healthz():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        """Readiness: models loaded and the index exists. K8s only routes traffic when this passes."""
        a = app.state.assistant
        if a is None or not a.retriever.is_ready():
            raise HTTPException(503, "index not ready")
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
