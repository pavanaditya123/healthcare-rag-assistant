import httpx
import pytest

from app.config import Settings
from app.llm import GroqLLM, LLMError

S = Settings(groq_api_key="test", primary_model="big", fallback_model="small", llm_retries=1)
OK = {"choices": [{"message": {"content": "answer [1]"}}]}


def client(handler):
    return GroqLLM(S, http=httpx.Client(transport=httpx.MockTransport(handler)), backoff_s=0)


def test_retries_primary_after_timeout():
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json=OK)

    assert client(handler).complete([]) == ("answer [1]", "big")
    assert len(calls) == 2


def test_falls_back_to_second_model():
    def handler(req):
        model = req.read().decode()
        return httpx.Response(503) if '"big"' in model else httpx.Response(200, json=OK)

    assert client(handler).complete([])[1] == "small"


def test_non_retryable_error_skips_straight_to_fallback():
    seen = []

    def handler(req):
        body = req.read().decode()
        seen.append("big" if '"big"' in body else "small")
        return httpx.Response(400) if '"big"' in body else httpx.Response(200, json=OK)

    client(handler).complete([])
    assert seen == ["big", "small"]


def test_raises_when_everything_fails():
    with pytest.raises(LLMError):
        client(lambda req: httpx.Response(500)).complete([])


def test_stream_parses_sse_and_falls_back_before_first_token():
    sse = (
        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"[1]"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(req):
        body = req.read().decode()
        if '"big"' in body:
            return httpx.Response(503)
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    out = list(client(handler).stream([]))
    assert "".join(t for t, _ in out) == "Hello [1]" and out[0][1] == "small"
