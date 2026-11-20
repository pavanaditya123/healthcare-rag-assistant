from evals.judge_eval import average_precision, errors_of, score, summarise


def test_average_precision_rewards_relevant_passages_ranked_first():
    assert average_precision([True, True, False]) == 1.0
    assert average_precision([False, True]) == 0.5
    assert average_precision([False, False]) == 0.0


def test_score_and_error_types():
    j = {"claims": [{"claim": "a", "cited": [1], "supported_by": [1]},
                    {"claim": "b", "cited": [2], "supported_by": []}],
         "answer_relevance": 5, "passage_relevant": [True, False],
         "reference_facts": [{"fact": "x", "in_passages": True}, {"fact": "y", "in_passages": False},
                             {"fact": "z", "in_passages": False}]}
    m = score(j)
    assert m["faithfulness"] == 0.5 and m["citation_accuracy"] == 0.5 and m["answer_relevancy"] == 1.0
    assert m["unsupported_claims"] == ["b"]
    assert errors_of(m) == {"hallucination", "miscitation", "retrieval_miss"}


def test_confirmed_vs_disputed_errors():
    bad = score({"claims": [{"claim": "b", "cited": [1], "supported_by": []}], "answer_relevance": 5,
                 "passage_relevant": [True], "reference_facts": [{"fact": "x", "in_passages": True}]})
    good = score({"claims": [{"claim": "a", "cited": [1], "supported_by": [1]}], "answer_relevance": 5,
                  "passage_relevant": [True], "reference_facts": [{"fact": "x", "in_passages": True}]})
    items = [{"route": "answered", "judgments": {"j1": bad, "j2": bad}},
             {"route": "answered", "judgments": {"j1": bad, "j2": good}}]
    r = summarise(items, ["j1", "j2"])
    assert r["errors"]["hallucination"] == {"confirmed": 1, "disputed": 1, "of": 2}
    assert r["agreement"]["hallucination"] == 0.5


def test_gemini_judge_uses_its_own_endpoint_and_key(monkeypatch):
    from app.config import Settings
    from evals.judge_eval import GEMINI_URL, judge_client, pick_judges
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    s = Settings(groq_api_key="q-test")
    g, q = judge_client("gemini:gemini-3.8-flash", s), judge_client("openai/gpt-oss-120b", s)
    assert (g.s.groq_url, g.s.groq_api_key, g.s.primary_model) == (GEMINI_URL, "g-test", "gemini-3.8-flash")
    assert (q.s.groq_api_key, q.s.primary_model) == ("q-test", "openai/gpt-oss-120b")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert pick_judges("openai/gpt-oss-120b,gemini:gemini-3.8-flash") == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
