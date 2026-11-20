import json

from fastapi.testclient import TestClient

from app.answer import Assistant
from app.cache import MemoryStore
from app.config import Settings
from app.llm import LLMError
from app.main import create_app
from tests.fakes import FakeLLM, FakeRetriever


def make(llm=None, retriever=None, limit=100):
    retriever = retriever or FakeRetriever()
    s = Settings(rate_limit_per_min=limit)
    assistant = Assistant(retriever, llm or FakeLLM(), s)
    return TestClient(create_app(assistant=assistant, store=MemoryStore(), s=s)), retriever


def test_ask_then_cache_hit():
    c, retriever = make()
    first = c.post("/ask", json={"question": "What is glaucoma?"}).json()
    second = c.post("/ask", json={"question": "what is   GLAUCOMA?"}).json()
    assert first["route"] == "answered" and first["cached"] is False
    assert second["cached"] is True and len(retriever.queries) == 1


def test_phi_is_redacted_before_pipeline():
    c, retriever = make()
    c.post("/ask", json={"question": "I'm reachable at 555-123-4567, what is glaucoma?"})
    assert "555" not in retriever.queries[0] and "[PHONE]" in retriever.queries[0]


def test_rate_limit():
    c, _ = make(limit=2)
    codes = [c.post("/ask", json={"question": "What is glaucoma?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_degraded_answers_are_not_cached():
    c, retriever = make(llm=FakeLLM(error=LLMError("down")))
    for _ in range(2):
        assert c.post("/ask", json={"question": "What is glaucoma?"}).json()["route"] == "degraded"
    assert len(retriever.queries) == 2


def test_health_ready_metrics():
    c, _ = make(retriever=FakeRetriever(ready=False))
    assert c.get("/healthz").status_code == 200
    assert c.get("/readyz").status_code == 503
    c.post("/ask", json={"question": "What is glaucoma?"})
    assert "rag_requests_total" in c.get("/metrics").text


def test_rejects_empty_question():
    c, _ = make()
    assert c.post("/ask", json={"question": ""}).status_code == 422


def test_filters_reach_retriever_and_are_validated():
    c, retriever = make()
    ok = c.post("/ask", json={"question": "What is glaucoma?", "sources": ["GHR"], "qtypes": ["treatment"]})
    assert ok.status_code == 200 and retriever.filters == (["GHR"], ["treatment"])
    assert c.post("/ask", json={"question": "What is glaucoma?", "sources": ["WebMD"]}).status_code == 422


def stream_events(c, question):
    r = c.post("/ask/stream", json={"question": question})
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def test_stream_sends_sources_then_tokens_then_done():
    c, _ = make()
    events = stream_events(c, "What is glaucoma?")
    types = [e["type"] for e in events]
    assert types[0] == "sources" and types[-1] == "done" and "token" in types
    assert events[-1]["route"] == "answered" and "[1]" in events[-1]["answer"]
    # second call is served from the cache as a single "done" event
    again = stream_events(c, "What is glaucoma?")
    assert [e["type"] for e in again] == ["done"] and again[0]["cached"] is True


def test_stream_replaces_uncited_answer_with_sources_only():
    c, _ = make(llm=FakeLLM(text="Glaucoma is bad."))
    done = stream_events(c, "What is glaucoma?")[-1]
    assert done["route"] == "degraded" and "optic nerve" in done["answer"]


def test_stream_emergency_has_no_tokens():
    c, _ = make()
    events = stream_events(c, "I want to end my life")
    assert [e["type"] for e in events] == ["done"] and "988" in events[0]["answer"]


def test_overloaded_pod_returns_fast_503_with_retry_after():
    c, _ = make(retriever=FakeRetriever(overloaded=True))
    for path in ("/ask", "/ask/stream"):
        r = c.post(path, json={"question": "What is glaucoma?"})
        assert r.status_code == 503 and r.headers["retry-after"] == "1"


def test_request_id_is_returned_and_reused():
    c, _ = make()
    r = c.post("/ask", json={"question": "What is glaucoma?"})
    assert r.headers["x-request-id"] == r.json()["request_id"] and len(r.json()["request_id"]) == 32
    r = c.post("/ask", json={"question": "What is asthma?"}, headers={"X-Request-ID": "gateway-123"})
    assert r.headers["x-request-id"] == "gateway-123" and r.json()["request_id"] == "gateway-123"


def test_logs_are_json_with_request_id():
    import logging
    from app.logs import JsonFormatter, request_id
    token = request_id.set("abc")
    rec = logging.LogRecord("api", logging.INFO, __file__, 1, "ask", None, None)
    rec.fields = {"route": "answered"}
    line = json.loads(JsonFormatter().format(rec))
    request_id.reset(token)
    assert line["request_id"] == "abc" and line["route"] == "answered" and line["msg"] == "ask"
