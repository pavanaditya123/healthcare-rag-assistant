"""Answer-quality eval with LLM-as-judge: RAGAS-style metrics, two judges in parallel.

    GROQ_API_KEY=... GEMINI_API_KEY=... python -m evals.judge_eval --medquad 30 --liveqa 20

Pipeline:
  1. retrieve for every question (local index, real models)
  2. generate answers with the production prompt + Groq        (parallel)
  3. two DIFFERENT judge models grade every answer              (parallel)
  4. compute metrics, judge agreement, and an error breakdown

Metrics (definitions follow RAGAS, computed from one JSON judgment per answer):
  faithfulness       share of the answer's claims supported by the passages
  citation_accuracy  share of cited claims whose [n] really supports them
  answer_relevancy   does the answer address the question (1-5, scaled to 0-1)
  context_precision  are relevant passages ranked at the top (RAGAS average precision)
  context_recall     share of the reference answer's facts found in the passages

Why two judges: one judge has its own biases (it favours answers in its own
style). An error both judges flag is "confirmed" and is what we fix first;
low agreement tells us the metric itself is unreliable.

This uses RAGAS's metric definitions, not the `ragas` package: one judge call
per answer (instead of ~4-8) keeps us inside Groq free-tier limits, and every
prompt is visible here.
"""
import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from app.answer import Assistant
from app.config import Settings
from app.data import load_medquad
from app.llm import GroqLLM
from app.retrieval import SearchResult

JUDGE_PROMPT = """You are grading a medical question-answering system. Be strict and literal.

QUESTION:
{question}

RETRIEVED PASSAGES:
{passages}

REFERENCE ANSWER (written by NIH, treat as ground truth):
{reference}

SYSTEM ANSWER:
{answer}

Return ONLY this JSON:
{{
  "claims": [{{"claim": "<one factual statement from the SYSTEM ANSWER>",
               "cited": [<the [n] numbers the answer attached to it>],
               "supported_by": [<passage numbers that actually state it; [] if none>]}}],
  "answer_relevance": <1-5: 5 = directly and fully answers the question, 1 = does not address it>,
  "passage_relevant": [<true/false for each passage in order: does it help answer the question?>],
  "reference_facts": [{{"fact": "<one key fact from the REFERENCE ANSWER>",
                        "in_passages": <true/false: stated in the retrieved passages?>}}]
}}
List at most 8 claims and at most 6 reference facts. Skip "see your doctor" style advice as a claim."""


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def judge_client(spec: str, s: Settings) -> GroqLLM:
    """"model" = a Groq model; "gemini:model" = Google's OpenAI-compatible endpoint.
    Same client, different URL and key, so retries/backoff behave the same for both."""
    provider, _, model = spec.rpartition(":") if spec.startswith("gemini:") else ("groq", "", spec)
    url, key = (GEMINI_URL, os.getenv("GEMINI_API_KEY", "")) if provider == "gemini" else (s.groq_url, s.groq_api_key)
    # Reasoning models (gpt-oss, Gemini 2.5) spend hidden tokens before the JSON; leave room.
    js = replace(s, groq_url=url, groq_api_key=key, primary_model=model, fallback_model=model,
                 llm_max_tokens=4000, llm_timeout_s=120, llm_retries=4)
    return GroqLLM(js, backoff_s=4, extra_body={"response_format": {"type": "json_object"}})


def pick_judges(spec: str) -> list[str]:
    judges = spec.split(",")
    if not os.getenv("GEMINI_API_KEY") and any(j.startswith("gemini:") for j in judges):
        print("GEMINI_API_KEY not set: using llama-3.3-70b-versatile as the second judge "
              "(same family as the generator, so expect it to be more lenient)")
        judges = [("llama-3.3-70b-versatile" if j.startswith("gemini:") else j) for j in judges]
    return judges


# ---------- metrics from one judgment (pure functions, unit tested) ----------
def average_precision(relevant: list[bool]) -> float | None:
    """RAGAS context precision: mean of precision@k over the ranks k that are relevant."""
    hits, total = 0, 0.0
    for k, rel in enumerate(relevant, start=1):
        if rel:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def score(j: dict) -> dict:
    claims = j.get("claims") or []
    facts = j.get("reference_facts") or []
    cited = [c for c in claims if c.get("cited")]
    return {
        "faithfulness": sum(bool(c.get("supported_by")) for c in claims) / len(claims) if claims else None,
        "citation_accuracy": (sum(bool(set(c["cited"]) & set(c.get("supported_by") or [])) for c in cited)
                              / len(cited)) if cited else None,
        "answer_relevancy": (float(j["answer_relevance"]) - 1) / 4 if j.get("answer_relevance") else None,
        "context_precision": average_precision([bool(x) for x in j.get("passage_relevant") or []]),
        "context_recall": sum(bool(f.get("in_passages")) for f in facts) / len(facts) if facts else None,
        "unsupported_claims": [c["claim"] for c in claims if not c.get("supported_by")],
    }


# Error taxonomy: each maps to a different fix.
ERRORS = {
    "hallucination":     lambda m: m["faithfulness"] is not None and m["faithfulness"] < 1.0,   # -> prompt / generation
    "miscitation":       lambda m: m["citation_accuracy"] is not None and m["citation_accuracy"] < 1.0,
    "retrieval_miss":    lambda m: m["context_recall"] is not None and m["context_recall"] < 0.5,  # -> retrieval
    "noisy_context":     lambda m: m["context_precision"] is not None and m["context_precision"] < 0.5,
    "off_target_answer": lambda m: m["answer_relevancy"] is not None and m["answer_relevancy"] < 0.5,
}


def errors_of(m: dict) -> set[str]:
    return {name for name, test in ERRORS.items() if m and test(m)}


# ---------- running ----------
class Precomputed:
    """Retriever stand-in that returns hits we already computed (so generation can run in threads)."""

    def __init__(self, hits_by_q):
        self.hits_by_q = hits_by_q

    def search(self, q, sources=None, qtypes=None):
        return SearchResult(reranked=self.hits_by_q[q])


def load_questions(chunks, n_medquad, n_liveqa, data_dir):
    from evals.retrieval_eval import liveqa_questions, medquad_questions
    by_id = {c.id: c for c in chunks}
    out = [{"set": "medquad", "question": q["question"],
            "reference": by_id[sorted(q["gold"])[0]].answer[:2000]}
           for q in medquad_questions(chunks, n_medquad, seed=7)]
    lq_xml = Path(data_dir) / "LiveQA/TestDataset/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"
    qrels = (Path(data_dir) / "qrels/QA-TestSet-LiveQA-Med-Qrels-2479-Answers"
             / "All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt")
    if n_liveqa and lq_xml.exists():
        lq, _ = liveqa_questions(lq_xml, qrels, set(by_id))
        for q in lq[:n_liveqa]:
            ref = "\n\n".join(by_id[g].answer[:1000] for g in sorted(q["gold"])[:2])
            out.append({"set": "liveqa", "question": q["question"], "reference": ref})
    return out


def judge_one(judge: GroqLLM, item: dict) -> dict | None:
    passages = "\n\n".join(f"[{s['n']}] {s['title']}: {item['passage_text'][i]}"
                           for i, s in enumerate(item["sources"]))
    prompt = JUDGE_PROMPT.format(question=item["question"], passages=passages or "(none)",
                                 reference=item["reference"], answer=item["answer"])
    try:
        text, _ = judge.complete([{"role": "user", "content": prompt}])
        return score(json.loads(text))
    except Exception as e:  # LLMError, bad JSON, missing keys: count it, don't crash the eval
        return {"judge_error": str(e)[:200]}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def summarise(items, judges):
    report = {"n": len(items), "routes": {}, "judges": {}, "agreement": {}, "errors": {}}
    for it in items:
        report["routes"][it["route"]] = report["routes"].get(it["route"], 0) + 1
    graded = [it for it in items if it["route"] == "answered"]
    for jm in judges:
        ms = [it["judgments"][jm] for it in graded if "judge_error" not in it["judgments"][jm]]
        report["judges"][jm] = {k: mean([m[k] for m in ms]) for k in
                                ("faithfulness", "citation_accuracy", "answer_relevancy",
                                 "context_precision", "context_recall")}
        report["judges"][jm]["judge_errors"] = len(graded) - len(ms)
    # Per error type: flagged by both judges (confirmed) or by one (disputed).
    for name in ERRORS:
        both = one = agree = n = 0
        for it in graded:
            flags = [name in errors_of(it["judgments"][jm]) for jm in judges
                     if "judge_error" not in it["judgments"][jm]]
            if len(flags) < 2:
                continue
            n += 1
            agree += flags[0] == flags[1]
            both += all(flags)
            one += any(flags) and not all(flags)
        report["errors"][name] = {"confirmed": both, "disputed": one, "of": n}
        report["agreement"][name] = round(agree / n, 3) if n else None
    return report


def to_markdown(report, items, judges):
    lines = [f"### LLM-as-judge answer quality ({report['n']} questions, judges: {', '.join(judges)})", "",
             f"Routes: {report['routes']}", "",
             "| metric | " + " | ".join(judges) + " |", "|---" * (len(judges) + 1) + "|"]
    for k in ("faithfulness", "citation_accuracy", "answer_relevancy", "context_precision", "context_recall"):
        lines.append(f"| {k} | " + " | ".join(str(report["judges"][j][k]) for j in judges) + " |")
    lines += ["", "| error type | confirmed (both judges) | disputed (one judge) | judge agreement |",
              "|---|---|---|---|"]
    for name, e in report["errors"].items():
        lines.append(f"| {name} | {e['confirmed']}/{e['of']} | {e['disputed']}/{e['of']} | {report['agreement'][name]} |")
    lines += ["", "**Confirmed hallucinations (unsupported claims):**"]
    for it in items:
        if it["route"] != "answered":
            continue
        if all("hallucination" in errors_of(it["judgments"][j]) for j in judges):
            claims = it["judgments"][judges[0]].get("unsupported_claims", [])[:2]
            lines.append(f"- *{it['question'][:90]}* -> {claims}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--index-path", default="data/eval_index")
    ap.add_argument("--medquad", type=int, default=30)
    ap.add_argument("--liveqa", type=int, default=20)
    ap.add_argument("--judges", default="openai/gpt-oss-120b,gemini:gemini-2.5-flash",
                    help="two judges, comma separated; 'gemini:<model>' uses GEMINI_API_KEY. "
                         "Pick families different from the Llama generator.")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    ap.add_argument("--out", default="evals/results/judge")
    args = ap.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set")

    from qdrant_client import QdrantClient
    from app.retrieval import Encoders, HybridRetriever

    s = Settings(collection="eval")
    chunks = load_medquad(f"{args.data}/MedQuAD")
    retriever = HybridRetriever(QdrantClient(path=args.index_path), Encoders(s), s)
    items = load_questions(chunks, args.medquad, args.liveqa, args.data)

    # 1. retrieval (CPU-bound, sequential)
    t0 = time.time()
    hits = {it["question"]: retriever.search(it["question"]).reranked[: s.top_k] for it in items}
    print(f"retrieved {len(items)} questions in {time.time() - t0:.0f}s")

    # 2. generation (network-bound, parallel)
    gen = Assistant(Precomputed(hits), GroqLLM(replace(s, llm_retries=3), backoff_s=3), s)

    def generate(it):
        a = gen.answer(it["question"])
        return {**it, "route": a.route, "answer": a.answer, "sources": a.sources, "model": a.model,
                "passage_text": [h.payload["text"][:1500] for h in hits[it["question"]]]}

    with ThreadPoolExecutor(args.workers) as pool:
        items = list(pool.map(generate, items))
    print(f"generated answers; routes: {summarise(items, [])['routes']}")

    # 3. two judges, every (answer, judge) pair in parallel
    judges = pick_judges(args.judges)
    clients = {jm: judge_client(jm, s) for jm in judges}
    for it in items:
        it["judgments"] = {}
    jobs = [(it, jm) for it in items if it["route"] == "answered" for jm in judges]
    with ThreadPoolExecutor(args.workers) as pool:
        for (it, jm), m in zip(jobs, pool.map(lambda job: judge_one(clients[job[1]], job[0]), jobs)):
            it["judgments"][jm] = m

    report = summarise(items, judges)
    md = to_markdown(report, items, judges)
    print(md)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".json").write_text(json.dumps({"report": report, "items": items}, indent=2), encoding="utf-8")
    Path(args.out + ".md").write_text(md, encoding="utf-8")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    main()
