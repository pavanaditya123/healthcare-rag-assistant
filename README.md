# Healthcare RAG Assistant

A question-answering service over **NIH consumer health content (MedQuAD)** that
answers only from cited NIH sources, routes emergencies to helplines without ever
calling the LLM, refuses diagnosis and dosing questions, and keeps working
(in a degraded, sources-only mode) when the LLM is down.

Every claim about quality is backed by an eval that runs in GitHub Actions.

```
question ─► PHI redaction ─► safety router ──► emergency  → helplines (911 / 988 / Poison Control), no LLM
                                    │       ├─► diagnosis / dose → refusal, no LLM
                                    │       └─► out of scope     → refusal, no LLM
                                    ▼
                         Redis answer cache (hit → return)
                                    ▼
              BM25 (sparse) ──┐
                              ├─► RRF fusion ─► cross-encoder rerank ─► score < threshold → "no reliable info"
       bge-small (dense) ─────┘        (all vectors in Qdrant)
                                    ▼
             Groq LLM (timeout → retry → fallback model) ─► citation validator
                                    │                              │
                                    └── LLM down / bad citations ──┴─► sources-only answer
```

## Results

Full corpus (16,407 chunks, 5,476 NIH pages), real models, CPU only. Reproduce with
`python -m evals.retrieval_eval`; CI re-runs it on every change to `app/` or `evals/`
and fails if MedQuAD reranked recall@5 drops below 0.90.

**MedQuAD held-out questions (n = 1,000)**

| Stage | recall@5 | MRR |
|---|---|---|
| BM25 (sparse) | 0.754 | 0.497 |
| bge-small (dense) | 0.880 | 0.731 |
| RRF fusion | 0.864 | 0.627 |
| **RRF + cross-encoder rerank** | **0.925** | **0.791** |

Reranker ceiling (fused recall@20) is 0.959, so the reranker recovers most of what fusion finds.

**TREC 2017 LiveQA-Med, real consumer questions — SMALL (38 questions, 96 relevant pairs)**

| Stage | recall@5 | hit@5 | MRR |
|---|---|---|---|
| BM25 | 0.239 | 0.395 | 0.281 |
| dense | 0.414 | 0.605 | 0.436 |
| RRF fusion | 0.393 | 0.553 | 0.398 |
| **+ rerank** | **0.523** | **0.763** | **0.503** |

The qrels judge 39 questions / 106 pairs after the copyright exclusions, but 10 of those
pairs (e.g. `NIHSeniorHealth_0000062_Sec6`) are not in the public MedQuAD XML (the qrels
were made on an earlier build), so 38 / 96 are scorable. Real questions are long and messy
("I have a 3 year old son with…"), which is why every stage drops sharply versus MedQuAD.

What the stage-by-stage numbers say:
- **Dense beats BM25 by a wide margin**, on both datasets. Consumer wording rarely matches NIH wording.
- **RRF alone does not beat dense here.** BM25 is weak enough that fusing it in costs a little at the top. Fusion still widens the candidate pool, which is what the reranker needs.
- **The cross-encoder is the biggest single gain** (+6 points recall@5 on MedQuAD, +13 on LiveQA).
- On the 38 NIST *paraphrased* LiveQA questions the reranker scored lower than fusion (0.439 vs 0.520 recall@5). With 38 questions that's within noise, but it's reported rather than hidden.

**Speed vs accuracy (same index, MedQuAD, 300 questions)**

| Candidates per retriever | Reranker reads | recall@5 | rerank p50 | search p95 |
|---|---|---|---|---|
| 20 | full passage | 0.924 | 1,883 ms | 3,463 ms |
| 20 | 1,000 chars | 0.929 | 1,103 ms | 2,464 ms |
| 20 | 500 chars | 0.926 | 782 ms | 2,174 ms |
| **10 (default)** | **1,000 chars (default)** | **0.926** | **746 ms** | **2,063 ms** |

On MedQuAD, reranking 10 candidates instead of 20 costs nothing measurable and cuts rerank
time by 32%. The full CI eval at the 10-candidate default shows the trade-off is **not free on
real questions**:

| Reranked recall@5 | 20 candidates (local) | 10 candidates (CI, default) |
|---|---|---|
| MedQuAD (n = 1,000) | 0.925 | 0.921 |
| LiveQA-Med (n = 38) | 0.523 | 0.449 |
| LiveQA paraphrased (n = 38) | 0.439 | 0.514 |

Messy consumer questions put the right passage deeper in the fused list, so a smaller pool
loses it. With 38 questions the gap is noisy (the paraphrased set moved the other way), but
it is the honest cost of the latency win. 10 is the default because it keeps normal-load p95
near 1 s on CPU. With a GPU, or a pod with more cores, set `CANDIDATES=20`.
(Headline tables above were run at 20.)
Latencies are from a Windows laptop running the eval alone; BM25 + dense together take ~25 ms on
the CI runner, the reranker is ~98% of search time.

**"No reliable info" threshold (best cross-encoder score)**

| Threshold | In-domain questions refused | Out-of-scope questions answered |
|---|---|---|
| −6 | 0.0% | 17.4% |
| −4 | 0.0% | 0.0% |
| **0 (default)** | **0.2%** | **0.0%** |
| 4 | 5.1% | 0.0% |

**Safety set, end to end (100 questions)**

| True label | Outcome |
|---|---|
| emergency (25) | 25 → helplines, LLM never called |
| diagnosis / dose (24) | 24 → refused, LLM never called |
| out of scope (23) | 20 → refused by rule, 3 → "no reliable info" via the score threshold, **0 answered** |
| normal (28) | 27 answered, 1 "no reliable info" |

Router-only confusion matrix (gated in CI): emergency recall 100%, no emergency or dosing
question ever reaches the LLM, accuracy 97%.

**Load (CI, 2-vCPU runner, stubbed LLM).** The first run at 20 concurrent users gave p95 = 20.9 s
from CPU contention in the reranker. That finding led to per-process inference limits and load
shedding (see *Reliability and scale*). CI now gates a normal-load phase (1 user per core) on
p95 and an overload phase on shedding fast instead of slowing down.

| Load phase (CI, 2 vCPU, 2 workers) | req/s | p50 | p95 | errors | shed (503) |
|---|---|---|---|---|---|
| Before limits: 20 concurrent users | 1.6 | 16.9 s | **20.9 s** | 0% | — |
| Normal: 2 concurrent users (gate p95 ≤ 3 s) | 3.1 | 706 ms | **1,116 ms** | 0% | 0% |
| Overload: 20 concurrent users | 23.3 | 31 ms | 2,473 ms (served) | 0% | 24% |

Under overload the pod no longer slows everyone down: requests it can serve stay
bounded, the rest get an immediate 503 to retry elsewhere, and in Kubernetes the HPA
adds pods. Going from 20 to 10 rerank candidates took normal-load p95 from 3.3 s to 1.1 s.

## Data

| | |
|---|---|
| Corpus | [MedQuAD](https://github.com/abachaa/MedQuAD) (CC BY 4.0), 9 NIH sources (GHR, GARD, NIDDK, NINDS, MedlinePlus Health Topics, NIHSeniorHealth, CancerGov, NHLBI, CDC) |
| Excluded | `10_MPlus_ADAM`, `11_MPlusDrugs`, `12_MPlusHerbs` — their answers were removed from the public release for copyright |
| Size | 5,476 NIH pages → **16,407 chunks** (one per QA pair) |
| Chunk id | Same as the official answer ids: `GHR_0000804_Sec2` = source GHR, Document id 0000804, QAPair pid 2 |
| Citations | Every chunk carries the URL of its NIH page |

**Why one chunk per QA pair?** MedQuAD documents are already split into
question-specific sections ("What causes X?", "How is X treated?"), and the
LiveQA relevance judgments are at exactly this level, so no extra splitting is
needed and evaluation maps 1:1 to chunks. Trade-off: about 12% of answers are
longer than 350 words, and bge-small / the cross-encoder only see the first 512
tokens of those (BM25 sees everything).

**The question text is never indexed** — only the topic and the answer. That
keeps the MedQuAD eval honest: the index has never seen any eval question.

## Evals

| Eval | What it measures | Where |
|---|---|---|
| MedQuAD held-out questions | 1,000 sampled questions → the QA pair they came from. recall@5, MRR per stage | `evals/retrieval_eval.py`, eval workflow |
| TREC 2017 LiveQA-Med | Real consumer questions with NIST/NLM judgments of MedQuAD answers; judgment 3 or 4 = relevant. **Small: 39 questions / 106 relevant pairs** after the copyright exclusions, 38 / 96 scorable against the public XML | same |
| Safety set | 100 hand-written questions (emergency / diagnosis-dose / out-of-scope / normal) → router confusion matrix, **gated in CI** | `evals/safety_eval.py`, CI workflow |
| Real LLM answers | Citation-validation pass rate and latency with Groq (only when `GROQ_API_KEY` is set) | `evals/answer_eval.py` |
| Load test | p50/p95/p99 and error rate with a stubbed LLM, **gated in CI** | `scripts/load_test.py`, CI workflow |

Each retrieval stage is measured separately (BM25 alone, dense alone, RRF, RRF +
rerank), plus the reranker's ceiling (fused recall@20: the reranker can only
reorder what fusion found).

Honest caveats:
- MedQuAD questions are templated ("What causes X ?"), so that benchmark is easier than real traffic. LiveQA is the realistic one, but at 39 questions a difference of a few points is noise.
- The safety set was written alongside the rules, so it is a **regression test** for known cases, not an estimate of real-world recall.

## Safety design

| Risk | Control |
|---|---|
| PHI in logs or sent to a third-party LLM | Regex redaction (email, phone, SSN, dates, MRN/member id, street address, "my name is …") runs first; nothing downstream sees the raw text, including the cache key and the logs |
| Emergencies | Rule-based router, checked before anything else. Self-harm phrases always trigger; acute symptoms ("chest pain", "can't breathe", "face drooping") trigger only with a person cue ("my dad…", "I…") so "What are the signs of a stroke?" is still answered. Returns a fixed message with 911, 988 and Poison Control — the LLM never sees the question |
| Diagnosis / dosing | Refused by rule ("do I have…", "how many mg…", "should I stop taking…"), no LLM call |
| Hallucination | LLM sees only the top-5 NIH passages; every paragraph must cite `[n]` and every `n` must be a real source, otherwise the answer is replaced by the sources-only answer |
| Questions the corpus can't answer | If the best cross-encoder score is below `MIN_RERANK_SCORE`, reply "no reliable information" instead of letting the LLM improvise. This is also the second net for out-of-scope questions the rules miss |

Why rules and not a classifier for routing? For emergencies a regex you can read
and unit-test beats a model you have to trust, and it adds ~0 ms. The cost is
recall on phrasings nobody wrote a rule for — that is exactly what the safety
set is there to grow.

## API

```bash
# full answer
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How is cystic fibrosis treated?"}'

# metadata filters (applied inside Qdrant, to both retrievers)
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Is it inherited?", "sources": ["GHR", "GARD"], "qtypes": ["inheritance"]}'

# streaming (server-sent events): sources -> tokens -> validated "done"
curl -N localhost:8000/ask/stream -H 'content-type: application/json' \
  -d '{"question": "What causes glaucoma?"}'
```

**Metadata filtering.** Each chunk stores `source` (GHR, GARD, NIDDK, …) and
`qtype` (information, causes, treatment, inheritance, …) with Qdrant payload
indexes. Filters are applied *before* the vector search (pre-filtering), so you
still get the full candidate list after filtering. Unknown sources are rejected
with 422, and filters are part of the cache key.

## Latency: what makes answers fast

| Technique | Why it helps |
|---|---|
| Redis answer cache | Repeated questions skip retrieval and the LLM entirely (ms instead of seconds) |
| BM25 and dense search run in parallel | They are independent; total = max(sparse, dense) instead of the sum |
| Reranker reads at most `RERANK_MAX_CHARS` (1,000) per passage | Cross-encoder cost grows with tokens; the eval's speed table shows the recall cost of each setting |
| Candidates per retriever (`CANDIDATES`, 10) | Fewer candidates = fewer cross-encoder pairs; same speed table |
| Streaming (`/ask/stream`) | Sources arrive right after retrieval and text starts at the LLM's first token, instead of waiting for the whole answer. Time-to-first-token is a metric (`stage="llm_first_token"`) |
| `LLM_MAX_TOKENS=300` | Answers are capped at ~150 words anyway; generation time scales with output tokens |
| Warm-up query at startup | The first ONNX run is slow; pods pay it before readiness, not on a user request |
| Models baked into the image, one worker per pod, HPA on CPU | No downloads at start-up; scale out with replicas |

Streaming and citations: the citation check needs the full text, so it runs at
the end. If it fails, the final `done` event carries the sources-only answer and
the client replaces what it showed. This is the trade-off for faster first text.

## Reliability and scale

- **Stateless API** (FastAPI). State lives in Qdrant (index) and Redis (cache, rate-limit counters), so replicas can be added or killed at any time.
- **Redis answer cache** keyed on the *redacted*, normalised question. Degraded answers are not cached, so the next request retries the LLM.
- **Rate limiting**: fixed window per client per minute (Redis `INCR` + `EXPIRE`).
- **Load shedding.** Embedding + reranking is CPU-bound ONNX work. The first CI load
  test (20 concurrent users on a 2-vCPU runner) had 0% errors but **p95 = 20.9 s**:
  every request ran the cross-encoder at once and they all slowed each other down
  (rerank went from ~0.5 s alone to ~18 s under contention). Now each process runs at
  most `MAX_INFLIGHT` searches with `ONNX_THREADS` threads; extra requests wait up to
  `QUEUE_TIMEOUT_S` and then get a fast `503 Retry-After: 1`. Served requests stay fast,
  `rag_shed_total` / `rag_inflight_searches` show the pressure, and the HPA adds pods.
  CI runs two load phases: normal load (gated on p95) and overload (must shed, not slow down).
- Both **fail open**: if Redis is down the API keeps answering, just without cache/limits.
- **LLM resilience**: 10 s timeout → retry with backoff on timeout/429/5xx → fallback model (`llama-3.1-8b-instant`) → sources-only answer.
- **Prometheus metrics** at `/metrics`: latency per stage (`sparse`, `dense`, `fusion`, `rerank`, `llm`, `llm_first_token`), end-to-end latency, routes taken, cache hit/miss, LLM errors per model, degraded answers by reason, rate-limited requests.
- **Kubernetes**: Deployment with readiness (`/readyz`: models loaded + index exists) and liveness probes, CPU-based HPA (2–10 pods, 70% target; retrieval and rerank are CPU-bound ONNX), Qdrant StatefulSet, Redis, one-off ingest Job. Models are baked into the image so new pods don't download anything.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                # unit tests, fake models, ~10 s
python -m evals.safety_eval              # safety confusion matrix + gates

bash scripts/download_data.sh            # MedQuAD + LiveQA at pinned commits
python -m evals.retrieval_eval           # full retrieval eval (~downloads models on first run)
```

With Docker:

```bash
export GROQ_API_KEY=...                  # or LLM_PROVIDER=stub to run without a key
docker compose up -d --build
docker compose run --rm ingest           # build the index once
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"What causes glaucoma?"}'
```

Kubernetes:

```bash
kubectl create secret generic groq --from-literal=GROQ_API_KEY=...
kubectl apply -f deploy/k8s/
```

## Project layout

```
app/
  data.py        MedQuAD XML -> chunks
  retrieval.py   BM25 + dense in Qdrant, RRF, cross-encoder rerank
  safety.py      PHI redaction, router, fixed safety messages
  llm.py         Groq client: timeout, retry, fallback; stub for tests
  answer.py      pipeline: route -> retrieve -> threshold -> LLM -> validate citations
  cache.py       Redis answer cache + rate limiter (fail open)
  metrics.py     Prometheus metrics
  main.py        FastAPI app
  ingest.py      build the index
evals/           retrieval, safety and LLM-answer evals
scripts/         data download, load test
deploy/          k8s manifests, Prometheus config
```

## Configuration

All settings are environment variables (see `app/config.py`): `QDRANT_URL`,
`REDIS_URL`, `GROQ_API_KEY`, `LLM_PROVIDER` (`groq` / `stub`),
`LLM_PRIMARY_MODEL`, `LLM_FALLBACK_MODEL`, `LLM_TIMEOUT_S`, `LLM_RETRIES`,
`LLM_MAX_TOKENS`, `MIN_RERANK_SCORE`, `TOP_K`, `CANDIDATES`, `RERANK_MAX_CHARS`,
`CACHE_TTL_S`, `RATE_LIMIT_PER_MIN`, `MAX_INFLIGHT`, `QUEUE_TIMEOUT_S`, `ONNX_THREADS`.

## Not medical advice

This is a portfolio project. Answers are general information from NIH pages and
are not a substitute for a clinician.

Data: MedQuAD — Ben Abacha & Demner-Fushman, *A Question-Entailment Approach to
Question Answering*, BMC Bioinformatics 2019 (CC BY 4.0). TREC 2017 LiveQA
Medical — Ben Abacha et al., TREC 2017.
