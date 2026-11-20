"""Smoke test of every configured LLM through the REAL production code path.

    GROQ_API_KEY=... GEMINI_API_KEY=... python scripts/llm_smoke.py

1. key shape + model lists (catches revoked keys and retired model names)
2. generator: primary and fallback each answer a real RAG prompt via GroqLLM,
   then the answer goes through the output guardrails
3. judges: each returns valid JSON for the real judge prompt

Never prints keys. A few thousand tokens in total.
"""
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import guardrails  # noqa: E402
from app.answer import build_messages, citations_valid  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm import GroqLLM, describe  # noqa: E402
from app.retrieval import Hit  # noqa: E402
from evals.judge_eval import GEMINI_URL, JUDGE_PROMPT, judge_client, pick_judges, score  # noqa: E402

GROQ_MODELS = "https://api.groq.com/openai/v1/models"
GEMINI_MODELS = GEMINI_URL.replace("/chat/completions", "/models")
PASSAGE = ("Glaucoma\nGlaucoma is a group of diseases that damage the optic nerve. Treatment usually "
           "includes prescription eye drops, laser surgery or conventional surgery to lower eye pressure.")
HIT = Hit("NEI_0000001_Sec2", 5.0, {"focus": "Glaucoma", "url": "https://www.nei.nih.gov/glaucoma", "text": PASSAGE})
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}]   {name:40} {detail}")


def key_shape(key: str) -> str:
    if not key:
        return "MISSING"
    issues = [n for n, bad in [("surrounding whitespace", key != key.strip()), ("quotes", '"' in key or "'" in key)] if bad]
    return f"present, {len(key)} chars" + (f", has {', '.join(issues)}" if issues else "")


def list_models(url, key):
    r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=20)
    r.raise_for_status()
    return {m["id"].removeprefix("models/") for m in r.json().get("data", [])}


def main():
    s = Settings()
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    print(f"GROQ_API_KEY: {key_shape(s.groq_api_key)}\nGEMINI_API_KEY: {key_shape(gemini_key)}")
    listed = {}
    for name, url, key in [("groq", GROQ_MODELS, s.groq_api_key), ("gemini", GEMINI_MODELS, gemini_key)]:
        if key:
            try:
                listed[name] = list_models(url, key.strip())
                print(f"  {name} /models: {len(listed[name])} models")
            except Exception as e:  # noqa: BLE001 - diagnostic script
                print(f"  {name} /models: {describe(e)}")

    # 2. generator, real prompt, real client, real guardrails
    for role, model in [("generator primary", s.primary_model), ("generator fallback", s.fallback_model)]:
        llm = GroqLLM(replace(s, primary_model=model, fallback_model=model, llm_retries=0))
        t0 = time.perf_counter()
        try:
            text = llm._call(model, build_messages("How is glaucoma treated?", [HIT]))
            ms = (time.perf_counter() - t0) * 1000
            problems = [p for p, bad in [
                ("empty (reasoning used all tokens?)", not text.strip()),
                ("citations invalid", not citations_valid(text, 1)),
                (f"unsafe: {guardrails.unsafe_output(text)}", bool(guardrails.unsafe_output(text))),
            ] if bad]
            support = guardrails.min_support(text, [PASSAGE])
            report(f"{role}: {model}", not problems,
                   f"{ms:.0f} ms, {len(text)} chars, grounding support {support}" +
                   (f", {'; '.join(problems)}" if problems else "") +
                   f" | {text.strip()[:90]!r}")
        except Exception as e:  # noqa: BLE001
            report(f"{role}: {model}", False, describe(e))

    # 3. judges, real prompt, must return parseable JSON with all metrics
    for spec in pick_judges(os.getenv("JUDGES") or "qwen/qwen3.8-27b,gemini:gemini-3.5-flash-lite"):
        prompt = JUDGE_PROMPT.format(question="How is glaucoma treated?", passages=f"[1] Glaucoma: {PASSAGE}",
                                     reference=PASSAGE, answer="Glaucoma is treated with eye drops, laser or surgery. [1]")
        t0 = time.perf_counter()
        try:
            text, _ = judge_client(spec, replace(s, llm_retries=0)).complete([{"role": "user", "content": prompt}])
            m = score(json.loads(text))
            ok = m["faithfulness"] is not None and m["context_recall"] is not None
            report(f"judge: {spec}", ok, f"{(time.perf_counter() - t0) * 1000:.0f} ms, faithfulness={m['faithfulness']}")
        except Exception as e:  # noqa: BLE001
            report(f"judge: {spec}", False, describe(e))

    for name, ids in sorted(listed.items()):
        usable = sorted(ids) if name == "groq" else sorted(i for i in ids if "flash" in i or "pro" in i)
        print(f"{name} models available to this key: {', '.join(usable[:30])}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
