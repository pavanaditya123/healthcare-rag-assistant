"""Pick the "no reliable info" cut-off (MIN_RERANK_SCORE) from user votes,
and export down-voted questions for human review as new eval cases.

    REDIS_URL=redis://localhost:6379/0 python scripts/tune_threshold.py
    python scripts/tune_threshold.py --file feedback.jsonl --down-cost 3

Nothing is changed automatically: the script prints a recommendation, a person
sets MIN_RERANK_SCORE in the ConfigMap, and the change goes through review.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import feedback  # noqa: E402
from app.cache import make_store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="JSONL of feedback records (default: read from REDIS_URL)")
    ap.add_argument("--down-cost", type=float, default=3.0, help="a bad answer costs this many good ones")
    ap.add_argument("--min-votes", type=int, default=30)
    ap.add_argument("--review-out", default="evals/results/feedback_review.jsonl")
    args = ap.parse_args()

    if args.file:
        records = [json.loads(x) for x in Path(args.file).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        records = feedback.load_log(make_store(os.environ["REDIS_URL"]))
    ups = sum(r["rating"] == "up" for r in records)
    print(f"{len(records)} votes: {ups} up, {len(records) - ups} down")

    best, table = feedback.best_threshold(records, args.down_cost, args.min_votes)
    if best is None:
        print(f"Not enough votes on generated answers (need {args.min_votes}); keep the eval-tuned threshold.")
    else:
        print(f"\n| MIN_RERANK_SCORE | up kept | down kept | refused | utility (down cost {args.down_cost:g}) |")
        print("|---|---|---|---|---|")
        for row in sorted(table, key=lambda r: -r["utility"])[:10]:  # 10 best cut-offs
            mark = " <- best" if row["threshold"] == best else ""
            print(f"| {row['threshold']:.3f} | {row['ups_kept']} | {row['downs_kept']} | {row['refused']} | "
                  f"{row['utility']:g}{mark} |")
        print(f"\nRecommended MIN_RERANK_SCORE={best:.3f}")

    # Down-voted questions -> review queue. A reviewer adds the good ones to the
    # eval sets (with the right gold passage or safety label).
    down = [{"question": r["question"], "route": r["route"], "top_score": r.get("top_score"),
             "source_ids": r.get("source_ids"), "comment": r.get("comment")}
            for r in records if r["rating"] == "down"]
    Path(args.review_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.review_out).write_text("".join(json.dumps(d) + "\n" for d in down), encoding="utf-8")
    print(f"{len(down)} down-voted questions written to {args.review_out} for review")


if __name__ == "__main__":
    main()
