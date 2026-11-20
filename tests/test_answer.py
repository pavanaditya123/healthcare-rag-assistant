from app.answer import Assistant, citations_valid
from app.config import Settings
from app.llm import LLMError
from tests.fakes import FakeLLM, FakeRetriever

S = Settings(min_rerank_score=0.0, top_k=5)


def test_citation_validation():
    assert citations_valid("Glaucoma harms the optic nerve. [1]", 1)
    assert citations_valid("Fact one [1].\n\nFact two [1][2].\n\nPlease see your doctor.", 2)
    assert not citations_valid("No citations here.", 2)
    assert not citations_valid("Made-up source [7].", 2)
    assert not citations_valid("Cited [1].\n\nUncited claim.", 2)


def test_normal_question_is_answered_with_sources():
    a = Assistant(FakeRetriever(), FakeLLM(), S).answer("What is glaucoma?")
    assert a.route == "answered" and "[1]" in a.answer
    assert a.sources[0]["url"] == "https://nei.nih.gov/glaucoma"


def test_emergency_never_reaches_retrieval_or_llm():
    retriever, llm = FakeRetriever(), FakeLLM()
    a = Assistant(retriever, llm, S).answer("I want to kill myself")
    assert a.route == "emergency" and "988" in a.answer
    assert retriever.queries == [] and llm.calls == 0


def test_dose_question_is_refused_without_llm():
    llm = FakeLLM()
    a = Assistant(FakeRetriever(), llm, S).answer("how many mg of tylenol should I take")
    assert a.route == "diagnosis_dose" and llm.calls == 0


def test_low_retrieval_score_means_no_info():
    llm = FakeLLM()
    a = Assistant(FakeRetriever(score=-3.0), llm, S).answer("how do I bake sourdough bread")
    assert a.route == "no_info" and llm.calls == 0


def test_llm_outage_returns_sources_only():
    a = Assistant(FakeRetriever(), FakeLLM(error=LLMError("down")), S).answer("What is glaucoma?")
    assert a.route == "degraded" and "optic nerve" in a.answer and a.sources


def test_uncited_llm_answer_is_rejected():
    a = Assistant(FakeRetriever(), FakeLLM(text="Glaucoma is bad."), S).answer("What is glaucoma?")
    assert a.route == "degraded"


def test_context_cache_skips_retrieval_but_still_calls_llm():
    from app.cache import AnswerCache, MemoryStore
    from tests.fakes import FakeLLM, FakeRetriever
    retriever, llm = FakeRetriever(), FakeLLM()
    a = Assistant(retriever, llm, Settings(), AnswerCache(MemoryStore(), 60, layer="context"))
    first, second = a.answer("What is glaucoma?"), a.answer("what is GLAUCOMA?")
    assert len(retriever.queries) == 1 and llm.calls == 2   # retrieval cached, LLM not
    assert first.sources == second.sources and second.route == "answered"
