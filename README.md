# Healthcare RAG Assistant

A production-style **Retrieval-Augmented Generation (RAG)** service that answers
consumer health questions **only from official NIH content (MedQuAD)**, cites every
claim, routes emergencies to helplines without calling the LLM, refuses diagnosis
and dosing questions, and degrades to a sources-only answer when the LLM fails.
Every quality claim is backed by an eval that runs in GitHub Actions.

```text
POST /ask  {"question": "How is cystic fibrosis treated?"}

{"route": "answered",
 "answer": "Treatment focuses on clearing mucus from the lungs, preventing infections
            and getting enough nutrition ... [1][2]",
 "sources": [{"n": 1, "title": "cystic fibrosis", "url": "https://ghr.nlm.nih.gov/..."},
             {"n": 2, "title": "Cystic Fibrosis", "url": "https://www.nhlbi.nih.gov/..."}],
 "cached": false, "request_id": "9f2c..."}
```

| Input | Route | Behaviour |
|---|---|---|
| "My dad has crushing chest pain right now" | `emergency` | Fixed message with 911 / 988 / Poison Control. LLM never called |
| "How many mg of ibuprofen should I take?" | `diagnosis_dose` | Refusal; refers to a clinician or pharmacist. LLM never called |
| "How do I fix my car brakes?" | `out_of_scope` | Refusal: health topics only |
| Health question with no good match in the corpus | `no_info` | "I couldn't find reliable information", instead of letting the LLM improvise |
| LLM down, or invalid citations | `degraded` | Top NIH passages, cited, without generated text |

> Not medical advice. Portfolio project; answers are general NIH information, not a substitute for a clinician.

**In everyday terms:** imagine asking a librarian at the National Institutes of
Health a question. They don't answer from memory: they find the right pages, write
you a short note quoting them, and staple the page references to it. If you describe
an emergency, they don't look anything up; they hand you the emergency number
immediately.

---

## Contents

1. [Architecture at a glance](#architecture-at-a-glance)
2. [Request lifecycle](#request-lifecycle)
3. [Safety layer](#safety-layer)
4. [Hybrid retrieval and reranking](#hybrid-retrieval-and-reranking)
5. [Generation and citation validation](#generation-and-citation-validation)
6. [Results](#results)
7. [Evaluation strategy](#evaluation-strategy)
8. [Latency: caching and streaming](#latency-caching-and-streaming)
9. [Human feedback loop](#human-feedback-loop)
10. [Reliability under load](#reliability-under-load)
11. [Deployment architecture](#deployment-architecture)
12. [Observability](#observability)
13. [Run it](#run-it)
14. [API reference](#api-reference)
15. [Known limitations](#known-limitations)
16. [Project layout and configuration](#project-layout-and-configuration)
17. [Glossary](#glossary)

---

## Architecture at a glance

```mermaid
flowchart TD
    Q["POST /ask"] --> RL{"Rate limiter<br/>Redis INCR, 30/min/client"}
    RL -- "over limit" --> R429["429 Too Many Requests"]
    RL --> PHI["PHI redaction<br/>regex: name, phone, email, SSN, date, MRN, address"]
    PHI --> SR{"Safety router<br/>rule-based classifier"}
    SR -- "emergency" --> EM["Fixed helpline response<br/>no LLM call"]
    SR -- "diagnosis_dose" --> DD["Fixed refusal<br/>no LLM call"]
    SR -- "out_of_scope" --> OS["Fixed refusal<br/>no LLM call"]
    SR -- "normal" --> AC{"Answer cache<br/>Redis, TTL 1 h"}
    AC -- "hit" --> HIT["Cached answer, ~2 ms"]
    AC -- "miss" --> CC{"Context cache<br/>Redis, TTL 24 h"}
    CC -- "hit" --> TH
    CC -- "miss" --> HR["Hybrid retrieval<br/>BM25 + dense bge-small in Qdrant<br/>RRF fusion"]
    HR --> CE["Cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2, ONNX on CPU"]
    CE --> TH{"Top score >= MIN_RERANK_SCORE?"}
    TH -- "no" --> NI["no_info response"]
    TH -- "yes" --> LLM["LLM generation<br/>Groq Llama 3.3 70B, top-5 passages"]
    LLM --> CV{"Citation validator<br/>every [n] valid, every paragraph cited"}
    CV -- "pass" --> OK["answered + sources"]
    CV -- "fail / LLMError" --> DG["degraded: sources-only answer"]

    style EM fill:#fde2e2,stroke:#c0392b,color:#000
    style DD fill:#fdf0d5,stroke:#b9770e,color:#000
    style OS fill:#fdf0d5,stroke:#b9770e,color:#000
    style NI fill:#fdf0d5,stroke:#b9770e,color:#000
    style OK fill:#d5f5e3,stroke:#1e8449,color:#000
    style HIT fill:#d5f5e3,stroke:#1e8449,color:#000
    style DG fill:#eaf2f8,stroke:#2471a3,color:#000
```

**Technically:** a stateless FastAPI service. State lives in Qdrant (vector index)
and Redis (caches, rate-limit counters, feedback). The LLM is only reached by
`normal` questions that pass the retrieval-score threshold; every other path is
deterministic.

**In everyday terms:** think of a hospital information desk with a strict
procedure. A **guard** stops people who keep coming back every few seconds (rate
limiter). A **clerk** blacks out your name and phone number on the form before
anyone else sees it (PHI redaction). A **triage nurse** reads the form first: chest
pain goes straight to emergency, "what dose should I take?" goes to the pharmacist
(safety router). Only ordinary questions reach the **librarian** (retrieval), who
pulls the best pages, and the **writer** (LLM), who drafts a reply from those pages
only. An **editor** (citation validator) checks every sentence against the pages
before the reply goes out.

---

## Request lifecycle

Example input: *"I'm Sam, call me on 555-123-4567. How is glaucoma treated?"*

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI
    participant RD as Redis
    participant QD as Qdrant
    participant ONNX as fastembed ONNX (CPU)
    participant G as Groq LLM

    C->>API: POST /ask
    API->>RD: rate limit INCR ratelimit:client:minute
    API->>API: redact() -> "I'm [NAME], call me on [PHONE]. How is glaucoma treated?"
    API->>API: classify() -> normal
    API->>RD: GET answer:v1:sha256(question)
    RD-->>API: miss
    API->>RD: GET context:v1:sha256(question)
    RD-->>API: miss
    par parallel retrieval (ThreadPoolExecutor)
        API->>ONNX: BM25 query vector
        API->>QD: sparse search, top 10
        API->>ONNX: bge-small query embedding
        API->>QD: dense search, top 10
    end
    API->>API: RRF fusion (k = 60)
    API->>ONNX: cross-encoder rerank (first 1,000 chars per passage)
    API->>RD: SETEX context (24 h)
    API->>G: system prompt + question + top-5 passages
    G-->>API: answer with [1] [2]
    API->>API: citations_valid()
    API->>RD: SETEX answer (1 h), trace for feedback (24 h)
    API-->>C: answer, sources, request_id
```

| Step | Component | Purpose |
|---|---|---|
| 1 | `RateLimiter` | Fixed window per client per minute; fails open if Redis is down |
| 2 | `redact()` | PHI replaced by `[NAME]`, `[PHONE]`, ... before logging, caching or calling the LLM |
| 3 | `classify()` | Rule-based routing; emergencies never reach retrieval or the LLM |
| 4–5 | `AnswerCache` (two layers) | Skip everything on an answer hit; skip retrieval on a context hit |
| 6 | `sparse_search` / `dense_search` | Two independent retrievers, run concurrently |
| 7 | `rrf()` | Rank-based fusion; no score normalisation needed |
| 8 | `rerank()` | Cross-encoder rescoring; the largest quality gain |
| 9 | `GroqLLM.complete()` | Grounded generation with timeout, retry and fallback |
| 10 | `citations_valid()` | Rejects invented or missing citations, falling back to sources-only |

**In everyday terms:** it's like posting a parcel. The post office weighs it
(rate limit), removes your personal note from the outside (redaction), checks it
isn't something dangerous (safety router), checks whether an identical parcel was
just sent and can be reused (cache), and only then does the real work. You get a
**tracking number** (`request_id`), so any complaint can be traced to exactly that
parcel.

---

## Safety layer

```mermaid
flowchart TD
    IN["redacted question"] --> SH{"SELF_HARM patterns<br/>suicid*, want to die, end my life ..."}
    SH -- match --> EM["route = emergency"]
    SH -- no --> AC{"ACUTE_NOW<br/>PERSON cue within 60 chars of<br/>chest pain, can't breathe, seizure,<br/>overdose, swallowed bleach ..."}
    AC -- match --> EM
    AC -- no --> DD{"DIAGNOSIS_DOSE_RULE<br/>do I have, how many mg,<br/>should I stop taking ..."}
    DD -- match --> DR["route = diagnosis_dose"]
    DD -- no --> OOS{"OUT_OF_SCOPE_RULE<br/>cars, cooking, code ..."}
    OOS -- match --> OR["route = out_of_scope"]
    OOS -- no --> N["route = normal -> retrieval"]
    N --> T{"rerank score < MIN_RERANK_SCORE"}
    T -- yes --> NI["route = no_info<br/>second safety net"]
```

| Risk | Control |
|---|---|
| PHI leakage to logs, cache or a third-party LLM | Regex redaction first; nothing downstream sees raw text |
| Missed emergency | Rule-based, evaluated first; CI gate requires **100% emergency recall** |
| Diagnosis or dosing advice | Rule-based refusal, no LLM call |
| Hallucination | Top-5 NIH passages only, mandatory citations, citation validator |
| Unanswerable questions | Retrieval-score threshold returns `no_info` |

**Why rules instead of an ML classifier?** For emergencies, a readable,
unit-tested regex beats a model you have to trust, and it adds ~0 ms. "What are the
signs of a stroke?" is still answered, because acute terms only trigger with a person
cue ("my mom can't breathe"). The cost is recall on phrasings nobody has written a
rule for, which is what the safety set exists to grow.

**In everyday terms:** an emergency-room **triage nurse** doesn't let a trainee
handle "my father has chest pain"; it goes straight to the emergency team, every
time, by a fixed rule. A **pharmacist** won't tell you your dose over the counter
without your doctor. And a good librarian who can't find a reliable source says
"I don't know" instead of guessing. That last one is the `no_info` threshold.

---

## Hybrid retrieval and reranking

```mermaid
flowchart LR
    Q["query"] --> SP["Sparse: BM25<br/>Qdrant/bm25 + IDF modifier<br/>top 10"]
    Q --> DN["Dense: bge-small-en-v1.5<br/>384-d, cosine<br/>top 10"]
    SP --> RRF["RRF fusion<br/>score = sum 1 / (60 + rank)"]
    DN --> RRF
    RRF --> CE["Cross-encoder rerank<br/>MiniLM-L-6, query + passage jointly<br/>first 1,000 chars"]
    CE --> K["top-5 to the LLM"]
    F["payload filters<br/>source, qtype<br/>applied inside Qdrant"] -.-> SP
    F -.-> DN
```

| Stage | What it is good at | Weakness |
|---|---|---|
| **BM25 (sparse)** | Exact terms: gene names, drug names, rare words | No synonyms: "heart attack" ≠ "myocardial infarction" |
| **Dense (bi-encoder)** | Meaning and paraphrase | Can miss exact rare terms |
| **RRF fusion** | Combines both using ranks only, so scales never need calibrating | Can't fix an ordering both retrievers got wrong |
| **Cross-encoder** | Reads query and passage *together*: the most accurate signal | Too slow to run on 16k passages, so it only reorders ~20 candidates |

**Metadata filtering:** every point stores `source` (GHR, GARD, NIDDK, ...) and
`qtype` (causes, treatment, inheritance, ...) with Qdrant payload indexes. Filters
are **pre-filters** applied inside the search, so a filtered query still returns a
full candidate list.

**Chunking:** one chunk per MedQuAD QA section (5,476 pages → 16,407 chunks).
Sections are already question-sized, and the official relevance judgments are at
this level, so evaluation maps 1:1 to chunks. The section's question text is
**never indexed** (only the focus and the answer), which keeps the held-out eval
honest. Trade-off: about 12% of sections are long, and the models read only the
first part of those.

**In everyday terms:**
- **BM25** is like using the **index at the back of a textbook**: perfect if you know the exact word, useless if you use a different word.
- **Dense search** is like asking a **friend who understands what you mean**, even if you describe it badly.
- **RRF** is like two friends each giving you a top-10 restaurant list; you go first to the places that are high on *both* lists.
- **The reranker** is like **hiring**: the keyword filter shortlists 20 CVs fast, then you actually *interview* those 20 to pick the best 5. Interviewing all 16,000 applicants would take forever.
- **Filters** are like asking the librarian for "only the genetics shelf".

---

## Generation and citation validation

```mermaid
flowchart LR
    P["top-5 passages<br/>max 1,500 chars each"] --> PR["Prompt<br/>answer ONLY from sources,<br/>cite every paragraph,<br/>no diagnosis, no dosing,<br/>max 150 words"]
    PR --> G["Groq Llama 3.3 70B<br/>temperature 0, max 300 tokens"]
    G --> V{"citations_valid()"}
    V -- "all [n] in 1..5 and<br/>every paragraph cited" --> A["answered"]
    V -- "invented [7] or<br/>uncited paragraph" --> D["degraded:<br/>sources-only answer"]
```

**In everyday terms:** it's a **student essay with footnotes**. The student
(LLM) may only use the five handed-out readings, and must footnote every paragraph.
The teacher (validator) checks each footnote number exists. A paragraph without a
footnote, or a footnote pointing to a reading that doesn't exist, means the essay is
rejected, and you get the original readings instead.

---

## Results

Full corpus (16,407 chunks), real models, CPU only. CI re-runs the retrieval eval
on code changes and **fails if MedQuAD reranked recall@5 < 0.90**.

```mermaid
xychart-beta
    title "recall@5 by pipeline stage, TREC LiveQA-Med (real consumer questions)"
    x-axis ["BM25", "Dense", "RRF", "RRF + rerank"]
    y-axis "recall@5" 0 --> 0.6
    bar [0.239, 0.414, 0.393, 0.523]
```

| Headline | Value |
|---|---|
| MedQuAD held-out recall@5 (n = 1,000), reranked | **0.925** (MRR 0.791) |
| LiveQA-Med real questions recall@5 (n = 38), reranked | **0.523** at 20 candidates, 0.449 at the 10-candidate default |
| Emergency recall (safety set) | **100%** (25 / 25), LLM never called |
| Out-of-scope questions answered | **0** of 23 |
| p95 latency, normal load (CI, stubbed LLM) | **1,116 ms** |
| Overload (20 concurrent users on 2 vCPU) | 0 errors, 24% shed with 503, served p50 31 ms |

**In everyday terms:** recall@5 asks "did the right book end up in the
librarian's top 5?" For textbook-style questions, the answer is yes about 92 times
out of 100. For real questions people actually type ("my 3 year old has..."), it's
about half the time, which is why the real-question set is the number to improve.

<details>
<summary><b>MedQuAD and LiveQA tables per stage</b></summary>

**MedQuAD held-out questions (n = 1,000)**

| Stage | recall@5 | MRR |
|---|---|---|
| BM25 (sparse) | 0.754 | 0.497 |
| bge-small (dense) | 0.880 | 0.731 |
| RRF fusion | 0.864 | 0.627 |
| **RRF + cross-encoder rerank** | **0.925** | **0.791** |

Reranker ceiling (fused recall@20): 0.959.

**TREC 2017 LiveQA-Med, real consumer questions. SMALL: 38 questions, 96 relevant pairs**

| Stage | recall@5 | hit@5 | MRR |
|---|---|---|---|
| BM25 | 0.239 | 0.395 | 0.281 |
| dense | 0.414 | 0.605 | 0.436 |
| RRF fusion | 0.393 | 0.553 | 0.398 |
| **+ rerank** | **0.523** | **0.763** | **0.503** |

The qrels judge 39 questions / 106 pairs after copyright exclusions; 10 pairs (e.g.
`NIHSeniorHealth_0000062_Sec6`) are absent from the public MedQuAD XML, so 38 / 96 are
scorable.

- **Dense beats BM25 by a wide margin**: consumer wording rarely matches NIH wording.
- **RRF alone does not beat dense here**: BM25 is weak enough to cost a little at the top, but fusion widens the reranker's candidate pool.
- **The cross-encoder is the biggest single gain**: +6 points on MedQuAD, +13 on LiveQA.
- On NIST's 38 *paraphrased* LiveQA questions the reranker scored below fusion (0.439 vs 0.520). Within noise at n = 38, but reported.

</details>

<details>
<summary><b>Speed vs accuracy: rerank candidates and passage length</b></summary>

| Candidates per retriever | Reranker reads | recall@5 | rerank p50 | search p95 |
|---|---|---|---|---|
| 20 | full passage | 0.924 | 1,883 ms | 3,463 ms |
| 20 | 1,000 chars | 0.929 | 1,103 ms | 2,464 ms |
| 20 | 500 chars | 0.926 | 782 ms | 2,174 ms |
| **10 (default)** | **1,000 chars (default)** | **0.926** | **746 ms** | **2,063 ms** |

| Reranked recall@5 | 20 candidates (local) | 10 candidates (CI, default) |
|---|---|---|
| MedQuAD (n = 1,000) | 0.925 | 0.921 |
| LiveQA-Med (n = 38) | 0.523 | 0.449 |
| LiveQA paraphrased (n = 38) | 0.439 | 0.514 |

10 candidates is free on MedQuAD and 32% faster, but costs recall on messy real
questions, where the right passage sits deeper in the fused list. It's the default
because it keeps p95 near 1 s on CPU; with a GPU, set `CANDIDATES=20`. The reranker is
~98% of search time (BM25 + dense ≈ 25 ms on the CI runner).

</details>

<details>
<summary><b>Threshold sweep and safety set end to end</b></summary>

| `MIN_RERANK_SCORE` | In-domain refused | Out-of-scope answered |
|---|---|---|
| −6 | 0.0% | 17.4% |
| −4 | 0.0% | 0.0% |
| **0 (default)** | **0.2%** | **0.0%** |
| 4 | 5.1% | 0.0% |

| True label (100 hand-written) | Outcome |
|---|---|
| emergency (25) | 25 → helplines, LLM never called |
| diagnosis / dose (24) | 24 → refused, LLM never called |
| out of scope (23) | 20 refused by rule, 3 → `no_info` via threshold, **0 answered** |
| normal (28) | 27 answered, 1 `no_info` |

Router-only accuracy: 97%, gated in CI.

</details>

<details>
<summary><b>Load test</b></summary>

CI, 2 vCPU, 2 uvicorn workers, LLM stubbed at 300 ms (measures our service, not Groq):

| Phase | req/s | p50 | p95 | errors | shed (503) |
|---|---|---|---|---|---|
| Before load shedding: 20 concurrent | 1.6 | 16.9 s | **20.9 s** | 0% | — |
| Normal: 2 concurrent (gate p95 ≤ 3 s) | 3.1 | 706 ms | **1,116 ms** | 0% | 0% |
| Overload: 20 concurrent | 23.3 | 31 ms | 2,473 ms (served) | 0% | 24% |

</details>

---

## Evaluation strategy

```mermaid
flowchart TD
    subgraph ci["CI on every push (ci.yml)"]
        UT["pytest: 58 unit tests<br/>fake encoders, no network"]
        SG["Safety gate<br/>confusion matrix, 100 questions"]
        E2E["e2e_check.py: 23 checks<br/>real models, Qdrant, Redis, 2 workers"]
        LT["load_test.py<br/>normal: p95 gate; overload: must shed"]
    end
    subgraph ev["Retrieval eval (eval.yml, on app or evals changes)"]
        RE["recall@5 / MRR per stage<br/>MedQuAD n=1,000 + LiveQA n=38<br/>gate: recall@5 >= 0.90"]
    end
    subgraph jd["Answer quality (eval.yml, manual dispatch)"]
        JE["judge_eval.py<br/>RAGAS-style metrics, 2 judges"]
    end
```

| Eval | Measures | File |
|---|---|---|
| Retrieval | recall@5, MRR, per stage, plus the reranker ceiling | `evals/retrieval_eval.py` |
| Safety gate | Router confusion matrix; emergency recall 100%, no emergency or dosing question reaches the LLM | `evals/safety_eval.py` |
| End-to-end | All features on a live server: caching, feedback, streaming, filters, routes, PHI, metrics | `scripts/e2e_check.py` |
| Load | p50 / p95 / p99, error rate, shed rate | `scripts/load_test.py` |
| LLM-as-judge | Faithfulness, citation accuracy, answer relevancy, context precision and recall | `evals/judge_eval.py` |

**In everyday terms:** car makers crash-test every new model before it reaches a
showroom. Here every code change is automatically crash-tested: if an emergency
stops being recognised, or search quality drops below 90%, the change is blocked
before it ships.

### LLM-as-judge (RAGAS-style metrics, two judges)

```mermaid
flowchart LR
    QS["50 questions<br/>30 MedQuAD + 20 LiveQA<br/>with NIH reference answers"] --> R["retrieve<br/>sequential, CPU"]
    R --> G["generate<br/>ThreadPoolExecutor"]
    G --> J1["Judge A<br/>openai/gpt-oss-120b (Groq)"]
    G --> J2["Judge B<br/>Gemini 2.5 Flash"]
    J1 --> M["per-answer JSON:<br/>claims + support, relevance,<br/>passage relevance, reference facts"]
    J2 --> M
    M --> CF{"error flagged by"}
    CF -- "both judges" --> C["confirmed -> fix first"]
    CF -- "one judge" --> D["disputed -> inspect"]
```

| Metric | Definition (RAGAS) | Low score means → fix |
|---|---|---|
| faithfulness | Share of answer claims supported by the passages | hallucination → prompt / generation |
| citation_accuracy | Share of cited claims whose `[n]` actually supports them | miscitation → prompt / validator |
| answer_relevancy | Does the answer address the question (1–5 → 0–1) | off-target answer → prompt |
| context_precision | Average precision of relevant passages in rank order | noisy context → retrieval / threshold |
| context_recall | Share of reference-answer facts present in the passages | retrieval miss → retrieval / chunking |

- **Two judges from different families**, neither of them Llama (the generator's family), to reduce self-preference bias. Agreement per error type is reported.
- **One JSON judgment per answer** instead of the `ragas` package's 4–8 calls, to fit free-tier limits; same metric definitions, prompts in the file.
- **~0.5M tokens per run**, so it runs on manual dispatch only.
- Gemini's free tier may use submitted data to improve Google's products. That's acceptable for public NIH text, not for real user questions.
- **Status:** the first run is in progress; results will be added here.

**In everyday terms:** an exam paper is marked by **two independent examiners**.
If both say an answer is wrong, it's wrong, and that's what gets fixed first. If
they disagree, someone looks at it before acting. And neither examiner is the
student's own tutor, who might go easy on a familiar style.

---

## Latency: caching and streaming

```mermaid
flowchart LR
    Q["redacted question + filters"] --> L1{"L1 answer cache<br/>key answer:v1:sha256<br/>TTL 1 h"}
    L1 -- hit --> A1["~2 ms<br/>no retrieval, no LLM"]
    L1 -- miss --> L2{"L2 context cache<br/>key context:v1:sha256<br/>TTL 24 h"}
    L2 -- hit --> A2["skip retrieval + rerank<br/>LLM only"]
    L2 -- miss --> A3["full pipeline<br/>retrieval ~1 s + LLM"]
```

Measured end to end: answer cache **878 ms → 2 ms**. After a thumbs-down evicted
the answer, regeneration took **303 ms** (the stubbed 300 ms LLM only), because the
context cache skipped retrieval.

| Technique | Effect |
|---|---|
| L1 answer cache | Repeats skip the whole pipeline. Keyed on the *redacted*, normalised question + filters |
| L2 context cache | Cache miss, down-vote or LLM retry re-runs only the LLM. Stored before the threshold, so tuning it needs no flush; cleared by ingest after a re-index |
| Parallel BM25 + dense | Latency = max(sparse, dense), not the sum |
| `CANDIDATES=10`, `RERANK_MAX_CHARS=1000` | Fewer, shorter cross-encoder pairs (see speed table) |
| SSE streaming (`/ask/stream`) | Sources emitted after retrieval; tokens as generated; time-to-first-token is a metric |
| `LLM_MAX_TOKENS=300` | Output length bounds generation time |
| Warm-up query at startup | First ONNX inference cost paid before readiness |
| Models baked into the image | No model download on pod start |
| Degraded answers not cached | The next request retries the LLM |

**In everyday terms:**
- The **answer cache** is the coffee shop barista who sees you walk in and already has your usual order ready.
- The **context cache** is a chef who keeps the vegetables already chopped: if a dish is sent back, they only cook it again instead of shopping and chopping from scratch.
- **Streaming** is a waiter bringing the bread first while the main course is still cooking: you're not staring at an empty table.

---

## Human feedback loop

```mermaid
flowchart TD
    A["response with request_id"] --> F["POST /feedback<br/>rating up / down, comment (redacted)"]
    F --> T["join with trace:{request_id}<br/>question, route, top_score, source_ids"]
    T --> LOG["Redis list feedback:log<br/>capped at 50,000"]
    F -- "down + answered" --> EV["evict L1 answer cache entry"]
    LOG --> TT["scripts/tune_threshold.py<br/>argmax over t of ups_kept - 3 x downs_kept"]
    LOG --> RV["export down-votes -> human review"]
    RV --> ES["new eval cases<br/>CI catches regressions"]
    TT --> CM["person updates MIN_RERANK_SCORE<br/>in ConfigMap"]
    F --> MET["rag_feedback_total{rating}"]
```

| Feedback effect | When | Mechanism |
|---|---|---|
| Bad cached answer stops being served | Immediately | Down-vote evicts the L1 entry |
| `no_info` threshold | Offline, by a person | Utility = good answers kept − 3 × bad answers kept (a wrong health answer costs more than a refusal) |
| Eval sets | Offline, by a person | Reviewed down-votes become new test cases |

Votes **never change the ranking automatically**: in a health app, unreviewed
votes can be wrong or gamed. One vote per `request_id`; unknown ids get 404 and
repeat votes 409.

**In everyday terms:** a restaurant comment card. If a customer says a dish was
bad, the waiter stops serving that plate right away (cache eviction). But the head
chef doesn't change the recipe because of one card; they read the cards weekly,
taste the dish, and only then change it (human review, threshold tuning, new tests).

---

## Reliability under load

### Load shedding

The first CI load test (20 concurrent users, 2 vCPU) had 0% errors but **p95 = 20.9 s**:
every request ran the cross-encoder at once and they slowed each other down (rerank
went from ~0.5 s alone to ~18 s under contention).

```mermaid
sequenceDiagram
    participant A as Request A
    participant B as Request B
    participant S as BoundedSemaphore(MAX_INFLIGHT=1)
    participant H as HPA
    A->>S: acquire
    S-->>A: slot granted, search ~1 s
    B->>S: acquire(timeout = QUEUE_TIMEOUT_S)
    alt slot released in time
        S-->>B: slot granted
    else timeout
        S-->>B: Overloaded -> HTTP 503, Retry-After: 1
        Note over B,H: rag_shed_total++ and rag_inflight_searches feed autoscaling
    end
    A->>S: release
```

Result under the same overload: 0 errors, served p50 31 ms, 24% shed instantly.

**In everyday terms:** a restaurant with 10 tables. A bad host seats 40 people at
once, and the kitchen collapses so that *everyone* waits an hour. A good host says
"about a 5-minute wait" or "please come back shortly", and the people already seated
get their food on time. Meanwhile the manager (autoscaler) opens another dining room.

### LLM resilience

```mermaid
flowchart LR
    P["primary: llama-3.3-70b-versatile<br/>timeout 10 s"] -- "408 / 429 / 5xx / timeout" --> R["retry with backoff"]
    R -- "still failing" --> F["fallback: llama-3.1-8b-instant"]
    P -- "400 / non-retryable" --> F
    F -- "fails" --> S["LLMError -> degraded<br/>sources-only answer"]
```

Streaming falls back only **before the first token**, since retrying mid-answer would
repeat text the user has already seen.

**In everyday terms:** if your usual taxi app doesn't respond, you try once more,
then switch to a second app. If both are down, you don't stay stranded: you're
handed a printed map (the NIH passages themselves).

### Other safeguards

| Mechanism | Detail |
|---|---|
| Rate limiting | 30 req/min per client, Redis fixed window (`INCR` + `EXPIRE`) → 429 |
| Fail-open Redis | Cache and limiter errors are logged and skipped; the API keeps answering |
| Input validation | Question 3–1,000 chars; unknown `sources` → 422 |

**In everyday terms:** rate limiting is like an **ATM daily withdrawal limit**: it
protects the system from one person draining it. Failing open is like a shop whose
card machine breaks: it keeps selling for cash instead of closing.

---

## Deployment architecture

```mermaid
flowchart TD
    CL["Clients"] --> SVC["Service rag-api<br/>L4 load balancing"]
    SVC --> P1["Pod: uvicorn, 1 worker<br/>models baked in image"]
    SVC --> P2["Pod"]
    SVC --> PN["Pod ... up to 10"]
    HPA["HorizontalPodAutoscaler<br/>min 2, max 10, CPU 70%<br/>scale-down window 300 s"] -.-> PN
    P1 --> QD[("Qdrant StatefulSet<br/>PVC 2 Gi")]
    P2 --> QD
    P1 --> RD[("Redis<br/>allkeys-lru, 256 MB")]
    P2 --> RD
    P1 --> GQ["Groq API"]
    JOB["Job rag-ingest<br/>initContainer: git clone MedQuAD"] --> QD
    PROM["Prometheus"] -. "scrape /metrics" .-> P1
    RP["readinessProbe /readyz<br/>livenessProbe /healthz"] -.-> P1
```

- **Stateless pods**: any pod can serve any request; state is in Qdrant and Redis.
- **Readiness** (`/readyz`: models loaded and index present) gates traffic; **liveness** (`/healthz`) restarts hung pods.
- **One worker per pod**: scale with replicas, and keep `/metrics` per pod accurate.
- Manifests in `deploy/k8s/`; local stack in `docker-compose.yml` (Qdrant, Redis, API, ingest, Prometheus).

**In everyday terms:** a **call center**. Any agent (pod) can take your call
because your file is in the shared system (Redis and Qdrant), not on one agent's
desk. When the queue grows, the supervisor (HPA) calls in more agents; a switchboard
(Service) routes each call to a free line; and an agent who stops responding is
replaced (liveness probe).

---

## Observability

| Signal | Detail |
|---|---|
| `rag_stage_latency_seconds{stage}` | Histogram per stage: `sparse`, `dense`, `fusion`, `rerank`, `llm`, `llm_first_token` |
| `rag_request_latency_seconds` | End-to-end latency |
| `rag_requests_total{route}` | answered / degraded / no_info / emergency / diagnosis_dose / out_of_scope |
| `rag_cache_total{layer,result}` | answer and context cache hit / miss / error |
| `rag_llm_calls_total{model,outcome}`, `rag_degraded_total{reason}` | LLM health and fallbacks |
| `rag_shed_total`, `rag_inflight_searches`, `rag_rate_limited_total` | Capacity pressure |
| `rag_feedback_total{rating}` | User votes |
| JSON logs with `request_id` | One object per line; the redacted question only; `X-Request-ID` echoed or generated |

**In everyday terms:** this is the car's **dashboard**: speed (latency), fuel
(capacity), warning lights (errors and fallbacks). The `request_id` is a **parcel
tracking number**: one complaint leads straight to that request's log lines.

---

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                # unit tests, fake models
python -m evals.safety_eval              # safety confusion matrix + gates
```

```bash
export GROQ_API_KEY=...                  # free key from console.groq.com
export LLM_PROVIDER=stub                 # or: run without any key (fake LLM)
docker compose up -d --build
docker compose run --rm ingest           # build the index once
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"What causes glaucoma?"}'
```

```bash
bash scripts/download_data.sh            # MedQuAD + LiveQA at pinned commits
python -m evals.retrieval_eval           # retrieval eval per stage
python scripts/e2e_check.py --url http://localhost:8000
GROQ_API_KEY=... GEMINI_API_KEY=... python -m evals.judge_eval
```

```bash
kubectl create secret generic groq --from-literal=GROQ_API_KEY=...
kubectl apply -f deploy/k8s/
```

---

## API reference

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question", "sources"?, "qtypes"?}` → `route`, `answer`, `sources`, `model`, `cached`, `context_cached`, `request_id` |
| `POST /ask/stream` | Server-sent events: `sources` → `token`… → `done` (validated; carries `request_id`) |
| `POST /feedback` | `{"request_id", "rating": "up"\|"down", "comment"?}`; a down-vote evicts the cached answer |
| `GET /healthz`, `GET /readyz` | Liveness, readiness |
| `GET /metrics` | Prometheus |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Is it inherited?", "sources": ["GHR", "GARD"], "qtypes": ["inheritance"]}'

curl -N localhost:8000/ask/stream -H 'content-type: application/json' \
  -d '{"question": "What causes glaucoma?"}'

curl -s localhost:8000/feedback -H 'content-type: application/json' \
  -d '{"request_id": "<id from /ask>", "rating": "down", "comment": "did not answer my question"}'
```

Status codes: `422` invalid input or unknown source, `429` rate limited, `503`
shed (with `Retry-After: 1`), `404` / `409` unknown or repeated feedback.

---

## Known limitations

| Limitation | Impact / next step |
|---|---|
| ~1.5 req/s per pod; CPU cross-encoder is the bottleneck | 100 req/s ≈ 70 pods. A GPU or hosted reranker is the real fix |
| LiveQA recall@5 ≈ 0.45–0.52 | Weakest metric. Next: a UMLS synonym graph from MedQuAD's annotations, and small-to-big chunking for long sections |
| Not deployed to a real cluster | Load numbers are single-machine |
| HPA on CPU | In-flight searches / shed rate is the truer saturation signal |
| Single Qdrant and Redis, no backups, re-index not zero-downtime | Single points of failure; Qdrant collection aliases would fix the re-index |
| No auth or TLS; rate limit per IP | Needs API keys or OAuth at a gateway |
| Metrics only: no tracing, alerts or SLOs | Add OpenTelemetry and alert rules |
| Regex PHI redaction, third-party LLM | Real PHI needs a BAA with the provider, audit logs and encryption, or a self-hosted model |
| 100 hand-written safety questions; MedQuAD questions are templated; LiveQA n = 38 | Regression tests, not clinical validation; small-sample noise |

---

## Project layout and configuration

```text
app/
  main.py        FastAPI: /ask, /ask/stream, /feedback, /healthz, /readyz, /metrics
  safety.py      PHI redaction, rule-based router, fixed safety messages
  retrieval.py   BM25 + dense in Qdrant, RRF, cross-encoder rerank, load shedding
  answer.py      pipeline: route -> retrieve -> threshold -> LLM -> validate citations
  llm.py         Groq client: timeout, retry, fallback; stub for tests
  cache.py       L1 answer + L2 context cache, rate limiter (fail open)
  feedback.py    request traces, votes, threshold selection
  logs.py        JSON logs with request id
  metrics.py     Prometheus metrics
  data.py        MedQuAD XML -> chunks
  ingest.py      index build (+ context cache invalidation)
evals/           retrieval, safety, answer and LLM-as-judge evals
scripts/         data download, load test, e2e check, threshold tuning
deploy/          Kubernetes manifests, Prometheus config
tests/           unit tests with fake encoders
```

| Group | Environment variables |
|---|---|
| Storage | `QDRANT_URL`, `REDIS_URL` |
| LLM | `GROQ_API_KEY`, `LLM_PROVIDER` (`groq` / `stub`), `LLM_PRIMARY_MODEL`, `LLM_FALLBACK_MODEL`, `LLM_TIMEOUT_S`, `LLM_RETRIES`, `LLM_MAX_TOKENS` |
| Retrieval | `CANDIDATES`, `TOP_K`, `RERANK_MAX_CHARS`, `MIN_RERANK_SCORE` |
| Caching and limits | `CACHE_TTL_S`, `CONTEXT_CACHE_TTL_S`, `RATE_LIMIT_PER_MIN` |
| CPU protection | `MAX_INFLIGHT`, `QUEUE_TIMEOUT_S`, `ONNX_THREADS` |

---

## Glossary

| Term | Technical meaning | Everyday comparison |
|---|---|---|
| RAG | Retrieve relevant documents, then generate an answer conditioned on them | An open-book exam instead of answering from memory |
| Hallucination | Fluent output not supported by any source | Someone confidently making up an answer |
| Chunk | The unit of retrieval; here one MedQuAD QA section | One index card in a card catalogue |
| Embedding | Dense vector representation of text meaning | A GPS coordinate for meaning: close points mean similar ideas |
| BM25 | Sparse lexical ranking (term frequency × IDF) | The index at the back of a book |
| Bi-encoder (dense) | Query and passage embedded separately, compared by cosine | Matching by description rather than exact words |
| Cross-encoder | Query and passage scored jointly by one model | An interview instead of a CV skim |
| RRF | Rank fusion: Σ 1 / (k + rank) | Combining two friends' top-10 lists |
| Qdrant | Vector database with payload filters | A library organised by topic and meaning |
| Redis | In-memory key-value store | A whiteboard everyone in the office can read and write fast |
| PHI | Protected Health Information | Personal details blacked out before copying a form |
| recall@5 / MRR | Gold item in top 5 / mean of 1 ÷ rank of the first gold item | "Was the right book in the top 5?" / "How near the top was it?" |
| p50 / p95 | Median / 95th-percentile latency | Your typical commute / your commute on a bad day |
| Load shedding | Reject excess work early with 503 | A host saying "short wait" instead of overfilling the kitchen |
| HPA | Kubernetes autoscaler adding pods under load | Opening more checkout counters when queues grow |
| LLM-as-judge | A model grades another model's output | Two examiners marking the same paper |

---

## Data and credits

| | |
|---|---|
| Corpus | [MedQuAD](https://github.com/abachaa/MedQuAD) (CC BY 4.0): GHR, GARD, NIDDK, NINDS, MedlinePlus Health Topics, NIHSeniorHealth, CancerGov, NHLBI, CDC |
| Excluded | `10_MPlus_ADAM`, `11_MPlusDrugs`, `12_MPlusHerbs`: answers removed from the public release for copyright |
| Size | 5,476 pages → **16,407 chunks** |
| Chunk ids | Official answer ids: `GHR_0000804_Sec2` = source GHR, document 0000804, QA pair 2 |

MedQuAD: Ben Abacha & Demner-Fushman, *A Question-Entailment Approach to Question
Answering*, BMC Bioinformatics 2019. TREC 2017 LiveQA Medical: Ben Abacha et al., TREC 2017.
