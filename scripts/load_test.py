"""Minimal async load test for /ask. Exits 1 if latency or error gates fail.

    python scripts/load_test.py --url http://localhost:8000 --requests 400 --concurrency 20

Run against a server started with LLM_PROVIDER=stub so we measure our own
service (retrieval, rerank, cache, rate limiting), not Groq's latency.
"""
import argparse
import asyncio
import json
import os
import random
import time

import httpx

QUESTIONS = [
    "What is glaucoma?", "What causes asthma?", "How is high blood pressure treated?",
    "What are the symptoms of Parkinson's disease?", "Is cystic fibrosis inherited?",
    "How many people are affected by sickle cell disease?", "What is the outlook for ALS?",
    "How is hepatitis C spread?", "What are the risk factors for COPD?", "What is Marfan syndrome?",
    "How can I prevent osteoporosis?", "What causes migraines?", "What is chronic kidney disease?",
    "How is Lyme disease treated?", "What are the stages of Alzheimer's disease?", "What is epilepsy?",
]


async def worker(client, url, queue, results):
    while True:
        try:
            q = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/ask", json={"question": q})
            results.append((time.perf_counter() - t0, r.status_code))
        except httpx.HTTPError:
            results.append((time.perf_counter() - t0, 0))


async def run(args):
    rng = random.Random(0)
    queue = asyncio.Queue()
    for i in range(args.requests):
        # Append a counter to most questions so we exercise the real pipeline, not just the cache.
        q = rng.choice(QUESTIONS)
        queue.put_nowait(q if rng.random() < args.repeat_ratio else f"{q} ({i})")
    results = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=30) as client:
        await asyncio.gather(*(worker(client, args.url, queue, results) for _ in range(args.concurrency)))
    wall = time.perf_counter() - t0

    # 503 = deliberate load shedding (pod at capacity), counted separately from errors.
    lat = sorted(t * 1000 for t, code in results if code == 200) or [0.0]
    shed = sum(1 for _, code in results if code == 503)
    errors = sum(1 for _, code in results if code not in (200, 503))
    pct = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))]  # noqa: E731
    summary = {
        "requests": len(results), "concurrency": args.concurrency, "rps": round(len(results) / wall, 1),
        "p50_ms": round(pct(0.50)), "p95_ms": round(pct(0.95)), "p99_ms": round(pct(0.99)),
        "error_rate": round(errors / len(results), 4),
        "shed_rate": round(shed / len(results), 4),   # latency percentiles are for served (200) requests
    }
    print(json.dumps(summary, indent=2))
    failures = []
    if summary["p95_ms"] > args.max_p95_ms:
        failures.append(f"p95 {summary['p95_ms']}ms > {args.max_p95_ms}ms")
    if summary["shed_rate"] > args.max_shed_rate:
        failures.append(f"shed rate {summary['shed_rate']:.2%} > {args.max_shed_rate:.2%}")
    if summary["error_rate"] > args.max_error_rate:
        failures.append(f"error rate {summary['error_rate']:.2%} > {args.max_error_rate:.2%}")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"### Load test: {args.name}\n\n```\n" + json.dumps(summary, indent=2) + "\n```\n"
                    + ("**FAIL:** " + "; ".join(failures) if failures else "**PASS**") + "\n")
    if failures:
        print("FAIL:", "; ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--repeat-ratio", type=float, default=0.3, help="share of exact repeats (cache hits)")
    ap.add_argument("--max-p95-ms", type=float, default=3000)
    ap.add_argument("--max-error-rate", type=float, default=0.01)
    ap.add_argument("--max-shed-rate", type=float, default=0.0, help="share of 503s allowed (overload test)")
    ap.add_argument("--name", default="stub LLM")
    asyncio.run(run(ap.parse_args()))
