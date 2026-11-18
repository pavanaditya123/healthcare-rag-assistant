"""Retrieval eval: recall@5 and MRR for every pipeline stage.

    python -m evals.retrieval_eval --data data --sample 1000

Datasets
  1. MedQuAD held-out questions -> the QA pair they came from. Question text
     is never indexed, so every question is unseen by the index. Many
     MedQuAD questions are templated ("What causes X ?"), so this is the
     easier benchmark.
  2. TREC 2017 LiveQA-Med: real consumer questions, with NIST/NLM judgments
     of MedQuAD answers. Judgment 3 (Incomplete) or 4 (Excellent) = relevant.
     After excluding ADAM/Drugs/Herbs: 39 questions, 106 relevant pairs.
     SMALL - treat differences of a few points as noise.

Stages: sparse (BM25), dense (bge-small), fused (RRF), reranked (cross-encoder).
Also reports the reranker's ceiling (fused recall@20) and a score-threshold
analysis used to pick MIN_RERANK_SCORE.
"""
import argparse
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from qdrant_client import QdrantClient

from app.answer import Assistant
from app.config import Settings
from app.data import EXCLUDED_SOURCES, load_medquad
from app.llm import StubLLM
from app.retrieval import Encoders, HybridRetriever
from app.safety import OUT_OF_SCOPE
from evals.safety_eval import load as load_safety_set

STAGES = ["sparse", "dense", "fused", "reranked"]


# ------------------------------------------------------------------ datasets
def medquad_questions(chunks, n, seed=42):
    """Sample n questions; gold = every chunk with the identical answer
    (MedQuAD repeats some answers across documents)."""
    by_answer = {}
    for c in chunks:
        by_answer.setdefault(c.answer, set()).add(c.id)
    pool = [c for c in chunks if c.question]
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    return [{"qid": c.id, "question": c.question, "gold": by_answer[c.answer]} for c in sample]


def liveqa_questions(questions_xml, qrels_txt, corpus_ids):
    gold = {}
    for line in Path(qrels_txt).read_text().splitlines():
        qid, judgment, answer_file = line.split()
        answer_id = answer_file.removesuffix(".txt")
        if judgment[0] in "34" and answer_id.split("_")[0] not in EXCLUDED_SOURCES:
            gold.setdefault(qid, set()).add(answer_id)

    out = []
    for q in ET.parse(questions_xml).getroot().iter("NLM-QUESTION"):
        qid = q.get("qid").removeprefix("TQ")
        if qid not in gold:
            continue
        orig = q.find("Original-Question")
        text = " ".join(f"{orig.findtext('SUBJECT') or ''}. {orig.findtext('MESSAGE') or ''}".split())
        if not gold[qid] & corpus_ids:
            continue
        out.append({"qid": qid, "question": text, "gold": gold[qid] & corpus_ids,
                    "paraphrase": (q.findtext("NIST-PARAPHRASE") or "").strip()})
    return out


# ------------------------------------------------------------------ metrics
def score_ranking(ids, gold, k=5):
    top = ids[:k]
    rr = next((1 / r for r, i in enumerate(ids, 1) if i in gold), 0.0)
    return {"recall@5": len(gold & set(top)) / len(gold), "hit@5": float(bool(gold & set(top))), "mrr": rr}


def pct(values, p):
    v = sorted(values)
    return round(v[min(len(v) - 1, int(len(v) * p))], 1) if v else 0.0


def evaluate(retriever, questions, field="question"):
    totals = {s: {"recall@5": 0.0, "hit@5": 0.0, "mrr": 0.0} for s in STAGES}
    ceiling, top_scores, total_ms = 0.0, [], []
    stage_ms = {}
    for q in questions:
        t0 = time.perf_counter()
        res = retriever.search(q[field])
        total_ms.append(1000 * (time.perf_counter() - t0))
        for stage, ms in res.timings_ms.items():
            stage_ms.setdefault(stage, []).append(ms)
        for s in STAGES:
            for m, v in score_ranking([h.id for h in getattr(res, s)], q["gold"]).items():
                totals[s][m] += v
        ceiling += len(q["gold"] & {h.id for h in res.fused}) / len(q["gold"])
        top_scores.append(res.reranked[0].score if res.reranked else -99)
    n = len(questions)
    return {
        "n": n,
        "stages": {s: {m: round(v / n, 4) for m, v in totals[s].items()} for s in STAGES},
        "fused_recall@20": round(ceiling / n, 4),
        # per-stage latency; sparse and dense run in parallel, so total < sum
        "latency_ms": {st: {"p50": pct(v, 0.5), "p95": pct(v, 0.95)}
                       for st, v in {**stage_ms, "total": total_ms}.items()},
        "top_rerank_scores": top_scores,
    }


def threshold_table(in_domain, out_domain, thresholds=(-6, -4, -2, 0, 2, 4)):
    rows = []
    for t in thresholds:
        rows.append({"threshold": t,
                     "in_domain_refused": round(sum(s < t for s in in_domain) / len(in_domain), 3),
                     "out_of_scope_answered": round(sum(s >= t for s in out_domain) / len(out_domain), 3)})
    return rows


def safety_end_to_end(retriever, s):
    """Run the safety set through the full pipeline (stub LLM) and count what
    happens to out-of-scope questions the regex router misses."""
    a = Assistant(retriever, StubLLM(), s)
    rows = load_safety_set()
    outcomes = {}
    for r in rows:
        route = a.answer(r["question"]).route
        outcomes.setdefault(r["label"], {}).setdefault(route, 0)
        outcomes[r["label"]][route] += 1
    oos_total = sum(outcomes.get(OUT_OF_SCOPE, {}).values())
    oos_answered = outcomes.get(OUT_OF_SCOPE, {}).get("answered", 0)
    return {"outcomes": outcomes, "out_of_scope_answered": oos_answered, "out_of_scope_total": oos_total}


# ------------------------------------------------------------------ report
def table(name, r):
    lines = [f"#### {name} (n={r['n']})", "", "| stage | recall@5 | hit@5 | MRR |", "|---|---|---|---|"]
    for s in STAGES:
        m = r["stages"][s]
        lines.append(f"| {s} | {m['recall@5']:.3f} | {m['hit@5']:.3f} | {m['mrr']:.3f} |")
    lat = r["latency_ms"]
    lines.append(f"\nReranker ceiling (fused recall@20): {r['fused_recall@20']:.3f}\n")
    lines.append("Latency ms (p50 / p95): " + "; ".join(f"{st} {v['p50']} / {v['p95']}" for st, v in lat.items()) + "\n")
    return "\n".join(lines)


def speed_tradeoff(retriever, base, questions):
    """Same index, different speed knobs: how much recall do we give up for speed?"""
    rows = []
    for cands, cap in [(20, 100_000), (20, 1000), (20, 500), (10, 1000)]:
        retriever.s = replace(base, candidates_per_retriever=cands, rerank_max_chars=cap)
        r = evaluate(retriever, questions)
        rows.append({"candidates": cands, "rerank_max_chars": cap if cap < 100_000 else "full",
                     "recall@5": r["stages"]["reranked"]["recall@5"], "mrr": r["stages"]["reranked"]["mrr"],
                     "rerank_p50_ms": r["latency_ms"]["rerank"]["p50"], "total_p95_ms": r["latency_ms"]["total"]["p95"]})
    retriever.s = base
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--out", default="evals/results/retrieval.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--index-path", default="data/eval_index")
    ap.add_argument("--speed-sample", type=int, default=300, help="questions for the speed trade-off table")
    ap.add_argument("--answer-sample", type=int, default=30, help="questions sent to Groq if GROQ_API_KEY is set")
    ap.add_argument("--min-medquad-recall", type=float, default=0.0, help="CI gate on reranked recall@5")
    args = ap.parse_args()
    data = Path(args.data)

    s = Settings(collection="eval")
    chunks = load_medquad(data / "MedQuAD")
    if args.limit:  # smoke tests only; reported numbers always use the full corpus
        chunks = chunks[: args.limit]
    # On-disk index, reused across runs (re-embedding 16k chunks is the slow part).
    client = QdrantClient(path=args.index_path)
    retriever = HybridRetriever(client, Encoders(s), s)
    t0 = time.time()
    if not (client.collection_exists(s.collection) and client.count(s.collection).count == len(chunks)):
        retriever.create_collection()
        for start in range(0, len(chunks), 2048):
            retriever.index(chunks[start:start + 2048])
            print(f"indexed {min(start + 2048, len(chunks))}/{len(chunks)}", flush=True)
    index_s = time.time() - t0

    corpus_ids = {c.id for c in chunks}
    mq = medquad_questions(chunks, args.sample)
    lq = liveqa_questions(
        data / "LiveQA/TestDataset/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml",
        data / "qrels/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt",
        corpus_ids,
    )
    results = {
        "corpus": {"chunks": len(chunks), "pages": len({c.url for c in chunks}), "index_seconds": round(index_s)},
        "medquad": evaluate(retriever, mq),
        "liveqa": evaluate(retriever, lq),
        "liveqa_paraphrased": evaluate(retriever, lq, field="paraphrase"),
    }
    results["liveqa"]["relevant_pairs"] = sum(len(q["gold"]) for q in lq)
    results["speed_tradeoff"] = speed_tradeoff(retriever, s, mq[: args.speed_sample])

    oos = [r["question"] for r in load_safety_set() if r["label"] == OUT_OF_SCOPE]
    oos_scores = [retriever.search(q).reranked[0].score for q in oos]
    results["threshold"] = threshold_table(results["medquad"]["top_rerank_scores"], oos_scores)
    results["safety_end_to_end"] = safety_end_to_end(retriever, s)
    if os.getenv("GROQ_API_KEY") and args.answer_sample:
        from evals.answer_eval import run as answer_check
        qs = [q["question"] for q in mq[: args.answer_sample // 2]] + [q["question"] for q in lq[: args.answer_sample // 2]]
        results["llm_answers"] = answer_check(retriever, qs, Settings(collection="eval"))

    lq_r = results["liveqa"]
    report = [
        f"### Retrieval eval — {len(chunks)} chunks from {results['corpus']['pages']} NIH pages (indexed in {round(index_s)}s)", "",
        table("MedQuAD held-out questions", results["medquad"]),
        table(f"TREC LiveQA-Med real consumer questions — SMALL: {lq_r['n']} questions, {lq_r['relevant_pairs']} relevant pairs",
              lq_r),
        table("LiveQA, NIST paraphrased versions of the same questions", results["liveqa_paraphrased"]),
        f"#### Speed vs accuracy (MedQuAD, first {args.speed_sample} questions)", "",
        "| candidates/retriever | rerank chars | recall@5 | MRR | rerank p50 ms | total p95 ms |", "|---|---|---|---|---|---|",
        *[f"| {r['candidates']} | {r['rerank_max_chars']} | {r['recall@5']:.3f} | {r['mrr']:.3f} | {r['rerank_p50_ms']} | {r['total_p95_ms']} |"
          for r in results["speed_tradeoff"]],
        "",
        "#### Score threshold (top cross-encoder score)", "",
        "| threshold | in-domain refused | out-of-scope answered |", "|---|---|---|",
        *[f"| {r['threshold']} | {r['in_domain_refused']:.1%} | {r['out_of_scope_answered']:.1%} |" for r in results["threshold"]],
        "",
        f"#### Safety set end-to-end (stub LLM, MIN_RERANK_SCORE={s.min_rerank_score})", "",
        "```", json.dumps(results["safety_end_to_end"]["outcomes"], indent=1), "```",
    ]
    if "llm_answers" in results:
        report += ["", "#### Real Groq answers (MedQuAD + LiveQA questions)", "", "```",
                   json.dumps(results["llm_answers"], indent=1), "```"]
    text = "\n".join(report)
    print(text)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for r in ("medquad", "liveqa", "liveqa_paraphrased"):
        results[r].pop("top_rerank_scores")
    Path(args.out).write_text(json.dumps(results, indent=2))
    Path(args.out).with_suffix(".md").write_text(text + "\n")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")

    got = results["medquad"]["stages"]["reranked"]["recall@5"]
    if got < args.min_medquad_recall:
        print(f"GATE FAILED: MedQuAD reranked recall@5 {got:.3f} < {args.min_medquad_recall}")
        sys.exit(1)


if __name__ == "__main__":
    main()
