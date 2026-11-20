from app.feedback import best_threshold


def votes(pairs):
    return [{"route": "answered", "top_score": s, "rating": r} for s, r in pairs]


def test_threshold_cuts_off_low_score_downvotes():
    # Low-score answers are mostly bad, high-score ones mostly good.
    recs = votes([(-2.0, "down")] * 10 + [(-1.0, "down")] * 5 + [(-1.0, "up")] * 2 + [(3.0, "up")] * 20)
    t, table = best_threshold(recs, down_cost=3)
    assert t == 3.0 and table


def test_keeps_everything_when_votes_are_good():
    t, _ = best_threshold(votes([(-2.0, "up")] * 15 + [(4.0, "up")] * 20))
    assert t < -2.0  # nothing refused


def test_needs_enough_votes():
    assert best_threshold(votes([(1.0, "up")] * 5)) == (None, [])
