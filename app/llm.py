"""LLM client for Groq (OpenAI-compatible chat API).

Resilience, in order:
  1. hard timeout on every call
  2. retry the primary model on timeouts / 429 / 5xx, with backoff
  3. fall back to a smaller, faster model
  4. if both fail, raise LLMError -> caller serves a sources-only answer
"""
import json
import logging
import os
import time

import httpx

from app.config import Settings
from app.metrics import LLM_CALLS, STAGE_LATENCY

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
log = logging.getLogger(__name__)


def describe(e: Exception) -> str:
    """Short, key-free description of an LLM failure for the logs:
    the HTTP status and the provider's error message, never request headers."""
    if isinstance(e, httpx.HTTPStatusError):
        try:
            body = e.response.json()
            body = body[0] if isinstance(body, list) and body else body  # Gemini wraps errors in a list
            err = body.get("error", body) if isinstance(body, dict) else body
            msg = err.get("message", err) if isinstance(err, dict) else err
        except ValueError:
            msg = e.response.text
        return f"HTTP {e.response.status_code}: {str(msg)[:200]}"
    return f"{type(e).__name__}: {str(e)[:200]}"


class LLMError(Exception):
    pass


class GroqLLM:
    def __init__(self, s: Settings, http: httpx.Client | None = None, backoff_s: float = 0.5,
                 extra_body: dict | None = None):
        self.s = s
        self.extra_body = extra_body or {}  # e.g. {"response_format": {"type": "json_object"}} for the judge
        self.http = http or httpx.Client(timeout=s.llm_timeout_s)
        self.backoff_s = backoff_s

    def _call(self, model: str, messages: list[dict]) -> str:
        resp = self.http.post(
            self.s.groq_url,
            headers={"Authorization": f"Bearer {self.s.groq_api_key}"},
            json={"model": model, "messages": messages, "temperature": 0, "max_tokens": self.s.llm_max_tokens,
                  **self.extra_body},
            timeout=self.s.llm_timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def complete(self, messages: list[dict]) -> tuple[str, str]:
        """Return (text, model_used) or raise LLMError."""
        for model in (self.s.primary_model, self.s.fallback_model):
            for attempt in range(self.s.llm_retries + 1):
                try:
                    with STAGE_LATENCY.labels("llm").time():
                        text = self._call(model, messages)
                    LLM_CALLS.labels(model, "ok").inc()
                    return text, model
                except httpx.HTTPStatusError as e:
                    LLM_CALLS.labels(model, "error").inc()
                    log.warning("llm call failed model=%s attempt=%d %s", model, attempt, describe(e))
                    if e.response.status_code not in RETRYABLE_STATUS:
                        break  # e.g. 400 model decommissioned, 401 bad key: retrying won't help
                except (httpx.TimeoutException, httpx.TransportError, KeyError, ValueError) as e:
                    LLM_CALLS.labels(model, "error").inc()
                    log.warning("llm call failed model=%s attempt=%d %s", model, attempt, describe(e))
                if attempt < self.s.llm_retries:
                    time.sleep(self.backoff_s * 2 ** attempt)
        raise LLMError("primary and fallback models failed")

    def stream(self, messages: list[dict]):
        """Yield (text_delta, model) as tokens arrive (server-sent events from Groq).

        Falls back to the second model only if the first fails BEFORE sending
        any token; retrying mid-answer would repeat text the user already saw.
        """
        for model in (self.s.primary_model, self.s.fallback_model):
            started = False
            t0 = time.perf_counter()
            try:
                with self.http.stream(
                    "POST", self.s.groq_url,
                    headers={"Authorization": f"Bearer {self.s.groq_api_key}"},
                    json={"model": model, "messages": messages, "temperature": 0,
                          "max_tokens": self.s.llm_max_tokens, "stream": True},
                    timeout=self.s.llm_timeout_s,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                        if delta:
                            if not started:
                                STAGE_LATENCY.labels("llm_first_token").observe(time.perf_counter() - t0)
                                started = True
                            yield delta, model
                LLM_CALLS.labels(model, "ok").inc()
                return
            except (httpx.HTTPError, KeyError, ValueError) as e:
                LLM_CALLS.labels(model, "error").inc()
                log.warning("llm stream failed model=%s %s", model, describe(e))
                if started:
                    raise LLMError("stream broke mid-answer") from e
        raise LLMError("primary and fallback models failed")


class StubLLM:
    """Deterministic stand-in for tests and load tests (no network, no key).

    Quotes the first sentence of source [1], so its answers are grounded and
    exercise the same output guardrails a real answer goes through.
    """

    def __init__(self, latency_s: float = 0.0):
        self.latency_s = latency_s

    @staticmethod
    def text_for(messages: list[dict]) -> str:
        user = messages[-1]["content"]
        source = user.split("[1] ", 1)[1].split("\n", 1)[1] if "[1] " in user else "No sources."
        first = source.split("\n[2]")[0].replace("\n", " ").split(". ")[0][:300].rstrip(".")
        return f"{first}. [1]"

    def complete(self, messages: list[dict]) -> tuple[str, str]:
        time.sleep(self.latency_s)  # simulate LLM latency in load tests
        return self.text_for(messages), "stub"

    def stream(self, messages: list[dict]):
        time.sleep(self.latency_s / 3)  # first token arrives before the full answer
        for word in self.text_for(messages).split(" "):
            yield word + " ", "stub"


def make_llm(s: Settings):
    if s.llm_provider == "stub":
        return StubLLM(latency_s=float(os.getenv("STUB_LATENCY_S", "0")))
    return GroqLLM(s)
