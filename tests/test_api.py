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
