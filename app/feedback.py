"""Human feedback (thumbs up / down) and what the system does with it.

Every answered request leaves a short trace (redacted question, route, top
rerank score, source ids) under its request id for a day. A vote is joined to
that trace, so we know WHICH answer, at WHICH retrieval score, was rated.

What feedback changes:
  * immediately: a thumbs-down evicts that answer from the cache, so the
    next user gets a freshly generated answer instead of the bad one
  * offline:     best_threshold() picks the "no reliable info" cut-off
    (MIN_RERANK_SCORE) from real votes - see scripts/tune_threshold.py
  * offline:     down-voted questions are exported as new eval cases

What it deliberately does NOT change: ranking. Raw votes in a health app can
be wrong or gamed; a person reviews them before they become eval cases or
rule changes.
"""
import json
import logging

log = logging.getLogger(__name__)

TRACE_TTL_S = 86400
LOG_KEY, LOG_MAX = "feedback:log", 50_000


def save_trace(store, rid: str, trace: dict):
    try:
        store.setex(f"trace:{rid}", TRACE_TTL_S, json.dumps(trace))
    except Exception as e:  # feedback is optional; never fail the answer because of it
        log.warning("trace save failed: %s", e)


def load_trace(store, rid: str) -> dict | None:
    raw = store.get(f"trace:{rid}")
    return json.loads(raw) if raw else None


def record(store, rid: str, record: dict) -> bool:
    """Returns False if this request was already voted on."""
    if store.get(f"voted:{rid}"):
        return False
    store.setex(f"voted:{rid}", TRACE_TTL_S, "1")
    store.rpush(LOG_KEY, json.dumps(record))
    store.ltrim(LOG_KEY, -LOG_MAX, -1)  # keep the newest LOG_MAX votes
    return True


def load_log(store) -> list[dict]:
    return [json.loads(x) for x in store.lrange(LOG_KEY, 0, -1)]


def best_threshold(records: list[dict], down_cost: float = 3.0, min_votes: int = 30):
    """Choose MIN_RERANK_SCORE from votes on generated answers.

    Raising the cut-off turns low-score answers into "no reliable info".
    That loses the up-voted ones and removes the down-voted ones. A wrong
    health answer is worse than a refusal, so each down-vote removed is worth
    `down_cost` up-votes kept:

        utility(t) = ups kept(score >= t) - down_cost * downs kept(score >= t)

    Returns (best threshold or None if too few votes, table of candidates).
    """
    votes = [(r["top_score"], r["rating"]) for r in records
             if r.get("route") == "answered" and r.get("top_score") is not None]
    if len(votes) < min_votes:
        return None, []
    candidates = sorted({s for s, _ in votes} | {min(s for s, _ in votes) - 1e-6})
    table = []
    for t in candidates:
        ups = sum(1 for s, r in votes if s >= t and r == "up")
        downs = sum(1 for s, r in votes if s >= t and r == "down")
        refused = sum(1 for s, _ in votes if s < t)
        table.append({"threshold": t, "ups_kept": ups, "downs_kept": downs,
                      "refused": refused, "utility": ups - down_cost * downs})
    best = max(table, key=lambda row: (row["utility"], -row["threshold"]))  # ties: refuse less
    return best["threshold"], table
