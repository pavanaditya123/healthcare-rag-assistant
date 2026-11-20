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


def test_describe_handles_every_error_shape():
    import httpx
    from app.llm import describe
    req = httpx.Request("POST", "https://x")
    for body, want in [({"error": {"message": "model gone"}}, "model gone"),
                       ([{"error": {"code": 404, "message": "not found"}}], "not found"),
                       ("plain text", "plain text")]:
        resp = httpx.Response(404, json=body, request=req) if not isinstance(body, str) else httpx.Response(404, text=body, request=req)
        e = httpx.HTTPStatusError("x", request=req, response=resp)
        assert describe(e) == f"HTTP 404: {want}"


def test_429_waits_for_retry_after_then_succeeds(monkeypatch):
    import httpx
    from app.config import Settings
    from app.llm import GroqLLM, retry_after
    waits, calls = [], {"n": 0}
    monkeypatch.setattr("app.llm.time.sleep", waits.append)

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "TPM"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok [1]"}}]})

    llm = GroqLLM(Settings(llm_retries=1, groq_api_key="k"), http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.complete([{"role": "user", "content": "hi"}]) == ("ok [1]", llm.s.primary_model)
    assert waits == [7.0]
    assert retry_after(httpx.Response(429, headers={"retry-after": "999"})) == 30.0
    assert retry_after(httpx.Response(429)) is None
