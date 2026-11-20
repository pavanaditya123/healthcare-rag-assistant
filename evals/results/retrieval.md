### Retrieval eval — 16407 chunks from 5476 NIH pages (indexed in 1492s)

#### MedQuAD held-out questions (n=1000)

| stage | recall@5 | hit@5 | MRR |
|---|---|---|---|
| sparse | 0.754 | 0.770 | 0.497 |
| dense | 0.880 | 0.904 | 0.731 |
| fused | 0.864 | 0.882 | 0.627 |
| reranked | 0.925 | 0.953 | 0.791 |

Reranker ceiling (fused recall@20): 0.959

Latency ms (p50 / p95): dense 259.7 / 324.8; sparse 374.0 / 542.5; fusion 0.0 / 0.1; rerank 431.6 / 580.8; total 810.0 / 1019.4

#### TREC LiveQA-Med real consumer questions — SMALL: 38 questions, 96 relevant pairs (n=38)

| stage | recall@5 | hit@5 | MRR |
|---|---|---|---|
| sparse | 0.239 | 0.395 | 0.281 |
| dense | 0.414 | 0.605 | 0.436 |
| fused | 0.393 | 0.553 | 0.398 |
| reranked | 0.523 | 0.763 | 0.503 |

Reranker ceiling (fused recall@20): 0.659

Latency ms (p50 / p95): dense 245.2 / 301.8; sparse 446.3 / 618.6; fusion 0.0 / 0.1; rerank 430.9 / 519.5; total 866.4 / 1085.8

#### LiveQA, NIST paraphrased versions of the same questions (n=38)

| stage | recall@5 | hit@5 | MRR |
|---|---|---|---|
| sparse | 0.417 | 0.579 | 0.383 |
| dense | 0.460 | 0.658 | 0.482 |
| fused | 0.520 | 0.684 | 0.554 |
| reranked | 0.439 | 0.632 | 0.494 |

Reranker ceiling (fused recall@20): 0.698

Latency ms (p50 / p95): dense 249.8 / 296.6; sparse 379.1 / 521.5; fusion 0.0 / 0.1; rerank 414.6 / 570.6; total 798.7 / 929.6

#### Speed vs accuracy (MedQuAD, first 300 questions)

| candidates/retriever | rerank chars | recall@5 | MRR | rerank p50 ms | total p95 ms |
|---|---|---|---|---|---|
| 20 | full | 0.924 | 0.769 | 1883.2 | 3463.0 |
| 20 | 1000 | 0.929 | 0.795 | 1103.4 | 2463.7 |
| 20 | 500 | 0.926 | 0.804 | 781.7 | 2174.2 |
| 10 | 1000 | 0.926 | 0.790 | 746.4 | 2063.3 |

#### Score threshold (top cross-encoder score)

| threshold | in-domain refused | out-of-scope answered |
|---|---|---|
| -6 | 0.0% | 17.4% |
| -4 | 0.0% | 0.0% |
| -2 | 0.1% | 0.0% |
| 0 | 0.2% | 0.0% |
| 2 | 0.8% | 0.0% |
| 4 | 5.1% | 0.0% |

#### Safety set end-to-end (stub LLM, MIN_RERANK_SCORE=0.0)

```
{
 "emergency": {
  "emergency": 25
 },
 "diagnosis_dose": {
  "diagnosis_dose": 24
 },
 "out_of_scope": {
  "out_of_scope": 20,
  "no_info": 3
 },
 "normal": {
  "answered": 27,
  "no_info": 1
 }
}
```
