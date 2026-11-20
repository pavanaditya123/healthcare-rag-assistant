"""End-to-end check of every feature against a RUNNING server (real models,
real Qdrant, real Redis when REDIS_URL is set). Unit tests use fakes; this
proves the pieces work together.

    python scripts/e2e_check.py --url http://localhost:8000

Exits 1 if any check fails. Works with LLM_PROVIDER=stub (CI) or groq.
"""
import argparse
import json
import random
import string
import sys
import time

import httpx

results = []
NIH_DOMAINS = ("nih.gov", "nihseniorhealth.gov", "cdc.gov")


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def stream(c, body):
    with c.stream("POST", "/ask/stream", json=body) as r:
        return [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    # Unique letters per run, so an earlier run's cache can't fake a pass. (Not digits:
    # a 10-digit timestamp is redacted as a phone number and every run would collide.)
    run = "run " + "".join(random.choice(string.ascii_lowercase) for _ in range(8))
    c = httpx.Client(base_url=args.url, timeout=60)

    # --- health ---
    check("liveness /healthz", c.get("/healthz").status_code == 200)
    check("readiness /readyz (models + index loaded)", c.get("/readyz").status_code == 200)

    # --- answer + citations + request id ---
    q = f"What causes glaucoma? ({run})"
    t0 = time.perf_counter()
    r = c.post("/ask", json={"question": q})
    first_ms = (time.perf_counter() - t0) * 1000
    a = r.json()
    check("answer returned with NIH sources", r.status_code == 200 and a["route"] == "answered"
          and a["sources"] and all(any(d in s["url"] for d in NIH_DOMAINS) for s in a["sources"]),
          f"route={a.get('route')}, {len(a.get('sources', []))} sources")
    check("retrieval is relevant (glaucoma in top source)", "glaucoma" in a["sources"][0]["title"].lower(),
          a["sources"][0]["title"])
    check("answer cites a provided source", "[1]" in a["answer"] or "[2]" in a["answer"])
    check("request id in body and header", a["request_id"] == r.headers.get("x-request-id") and a["request_id"])
    rid = a["request_id"]

    # --- answer cache ---
    t0 = time.perf_counter()
    again = c.post("/ask", json={"question": q.upper()}).json()   # normalised: case-insensitive
    hit_ms = (time.perf_counter() - t0) * 1000
    check("answer cache hit on repeat", again["cached"] is True, f"{first_ms:.0f} ms -> {hit_ms:.0f} ms")

    # --- feedback: thumbs-down evicts, then context cache serves passages ---
    fb = c.post("/feedback", json={"request_id": rid, "rating": "down", "comment": "too vague, call 555-123-4567"})
    check("thumbs-down recorded and evicts cached answer", fb.status_code == 200 and fb.json()["cache_evicted"] is True,
          fb.text[:80])
    check("second vote on same answer rejected (409)",
          c.post("/feedback", json={"request_id": rid, "rating": "up"}).status_code == 409)
    check("vote on unknown request id rejected (404)",
          c.post("/feedback", json={"request_id": "does-not-exist", "rating": "up"}).status_code == 404)
    t0 = time.perf_counter()
    regen = c.post("/ask", json={"question": q}).json()
    regen_ms = (time.perf_counter() - t0) * 1000
    check("after thumbs-down: answer regenerated, passages from context cache",
          regen["cached"] is False and regen["context_cached"] is True,
          f"cached={regen['cached']}, context_cached={regen['context_cached']}, {regen_ms:.0f} ms")

    # --- streaming ---
    events = stream(c, {"question": f"How is asthma treated? ({run})"})
    types = [e["type"] for e in events]
    check("stream: sources -> tokens -> done", types[0] == "sources" and types[-1] == "done" and "token" in types,
          f"{len(events)} events")
    done = events[-1]
    check("stream done carries request id usable for feedback", done.get("request_id") and
          c.post("/feedback", json={"request_id": done["request_id"], "rating": "up"}).status_code == 200)

    # --- metadata filters ---
    f = c.post("/ask", json={"question": f"Is cystic fibrosis inherited? ({run})", "sources": ["GHR"]}).json()
    ids = [s["id"] for s in f.get("sources", [])]
    check("source filter: only GHR passages", ids and all(i.startswith("GHR_") for i in ids), ", ".join(ids[:3]))
    check("unknown source rejected (422)",
          c.post("/ask", json={"question": "What is asthma?", "sources": ["WebMD"]}).status_code == 422)

    # --- safety routes (never reach the LLM) ---
    e = c.post("/ask", json={"question": "My dad has crushing chest pain right now"}).json()
    check("emergency routed to helplines, no sources", e["route"] == "emergency" and "911" in e["answer"]
          and not e["sources"])
    d = c.post("/ask", json={"question": "How many mg of ibuprofen should I take?"}).json()
    check("dosing question refused", d["route"] == "diagnosis_dose", d["route"])
    o = c.post("/ask", json={"question": "How do I fix my car brakes?"}).json()
    check("out-of-scope not answered", o["route"] in ("out_of_scope", "no_info"), o["route"])

    # --- PHI: the redacted question is what gets cached / traced ---
    p1 = c.post("/ask", json={"question": f"I'm at 555-867-5309, what is lupus? ({run})"}).json()
    p2 = c.post("/ask", json={"question": f"I'm at 555-000-1111, what is lupus? ({run})"}).json()
    check("PHI redacted before cache key (different phone, same cached answer)", p2["cached"] is True)

    # --- metrics ---
    m = c.get("/metrics").text
    for name in ("rag_stage_latency_seconds", "rag_cache_total", "rag_feedback_total", "rag_requests_total"):
        check(f"metric {name} exported", name in m)

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
