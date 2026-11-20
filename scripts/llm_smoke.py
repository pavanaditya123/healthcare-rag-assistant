"""One tiny request to every configured LLM, to diagnose keys and model names.

    GROQ_API_KEY=... GEMINI_API_KEY=... python scripts/llm_smoke.py

Prints the HTTP status, the provider's error message and whether each model is
in the provider's model list. Never prints keys. Costs a few tokens.
"""
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.llm import describe  # noqa: E402

GROQ = "https://api.groq.com/openai/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
CHECKS = [  # (provider, base url, key env var, model)
    ("groq", GROQ, "GROQ_API_KEY", os.getenv("LLM_PRIMARY_MODEL", "llama-3.3-70b-versatile")),
    ("groq", GROQ, "GROQ_API_KEY", os.getenv("LLM_FALLBACK_MODEL", "llama-3.1-8b-instant")),
    ("groq", GROQ, "GROQ_API_KEY", "openai/gpt-oss-120b"),
    ("gemini", GEMINI, "GEMINI_API_KEY", "gemini-2.5-flash"),
]


def key_shape(key: str) -> str:
    """Enough to spot a pasting mistake (quotes, spaces, empty), nothing secret."""
    if not key:
        return "MISSING"
    issues = [name for name, bad in [("leading/trailing whitespace", key != key.strip()),
                                     ("quote characters", '"' in key or "'" in key)] if bad]
    return f"present, {len(key)} chars" + (f", has {', '.join(issues)}" if issues else "")


def main():
    listed = {}
    for provider, base, env in {(p, b, e) for p, b, e, _ in CHECKS}:
        key = os.getenv(env, "")
        print(f"{env}: {key_shape(key)}")
        if not key:
            continue
        try:
            r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key.strip()}"}, timeout=20)
            r.raise_for_status()
            listed[provider] = {m["id"].removeprefix("models/") for m in r.json().get("data", [])}
            print(f"  {provider} /models: HTTP 200, {len(listed[provider])} models")
        except Exception as e:  # noqa: BLE001 - diagnostic script, report everything
            print(f"  {provider} /models: {describe(e)}")

    ok = True
    for provider, base, env, model in CHECKS:
        key = os.getenv(env, "").strip()
        if not key:
            continue
        in_list = "listed" if model in listed.get(provider, set()) else "NOT in model list"
        try:
            r = httpx.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "max_tokens": 5, "temperature": 0,
                                 "messages": [{"role": "user", "content": "Reply with the word OK."}]},
                           timeout=60)
            r.raise_for_status()
            print(f"[OK]   {provider:6} {model:28} ({in_list})")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[FAIL] {provider:6} {model:28} ({in_list}) {describe(e)}")
    for provider, ids in sorted(listed.items()):
        usable = sorted(ids) if provider == "groq" else sorted(i for i in ids if "flash" in i or "pro" in i)
        print(f"{provider} models available to this key: {', '.join(usable[:25])}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
