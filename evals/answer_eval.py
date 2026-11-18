"""End-to-end check with the real Groq LLM (runs only when GROQ_API_KEY is set).

Measures how often the LLM's answer passes citation validation, how often
we fall back to sources-only mode, and LLM latency. Called from
retrieval_eval.py so it reuses the index that is already built.
"""
import time

from app.answer import Assistant
from app.llm import GroqLLM


def run(retriever, questions: list[str], s) -> dict:
    a = Assistant(retriever, GroqLLM(s), s)
    routes, models, latencies = {}, {}, []
    for q in questions:
        t0 = time.perf_counter()
        ans = a.answer(q)
        latencies.append(time.perf_counter() - t0)
        routes[ans.route] = routes.get(ans.route, 0) + 1
        if ans.model:
            models[ans.model] = models.get(ans.model, 0) + 1
        time.sleep(1)  # stay well under Groq free-tier rate limits
    latencies.sort()
    generated = routes.get("answered", 0) + routes.get("degraded", 0)
    return {
        "n": len(questions),
        "routes": routes,
        "models": models,
        "citation_valid_rate": round(routes.get("answered", 0) / generated, 3) if generated else None,
        "p50_s": round(latencies[len(latencies) // 2], 2),
        "p95_s": round(latencies[int(len(latencies) * 0.95) - 1], 2),
    }
