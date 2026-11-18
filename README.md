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

RESULTS_PLACEHOLDER

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
| TREC 2017 LiveQA-Med | Real consumer questions with NIST/NLM judgments of MedQuAD answers; judgment 3 or 4 = relevant. **Small: 39 questions, 106 relevant pairs** after the copyright exclusions | same |
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
| Candidates per retriever (`CANDIDATES`, 20) | Fewer candidates = fewer cross-encoder pairs; same speed table |
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
`CACHE_TTL_S`, `RATE_LIMIT_PER_MIN`.

## Not medical advice

This is a portfolio project. Answers are general information from NIH pages and
are not a substitute for a clinician.

Data: MedQuAD — Ben Abacha & Demner-Fushman, *A Question-Entailment Approach to
Question Answering*, BMC Bioinformatics 2019 (CC BY 4.0). TREC 2017 LiveQA
Medical — Ben Abacha et al., TREC 2017.
