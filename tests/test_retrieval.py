from qdrant_client import QdrantClient

from app.config import Settings
from app.data import Chunk
from app.retrieval import Hit, HybridRetriever, rrf
from tests.fakes import FakeEncoders


def test_rrf_rewards_agreement():
    a = [Hit("x", 9, {}), Hit("y", 8, {}), Hit("z", 7, {})]
    b = [Hit("y", 0.9, {}), Hit("z", 0.8, {}), Hit("x", 0.1, {})]
    fused = rrf([a, b])
    assert fused[0].id == "y"                      # ranked 2nd and 1st beats 1st and 3rd
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-9


def chunk(i, focus, answer):
    return Chunk(f"GHR_{i:07d}_Sec1", "GHR", f"https://example.nih.gov/{i}", focus, "", "information", answer)


def test_hybrid_search_end_to_end_with_fake_models():
    s = Settings(collection="test", candidates_per_retriever=3)
    r = HybridRetriever(QdrantClient(":memory:"), FakeEncoders(), s)
    r.create_collection()
    r.index([
        chunk(1, "glaucoma", "Glaucoma is a group of eye diseases that damage the optic nerve."),
        chunk(2, "asthma", "Asthma is a chronic disease of the airways in the lungs."),
        chunk(3, "gout", "Gout is a form of arthritis caused by uric acid crystals."),
    ])

    res = r.search("what damages the optic nerve in glaucoma")

    assert r.is_ready()
    assert res.sparse[0].id == res.dense[0].id == res.fused[0].id == res.reranked[0].id == "GHR_0000001_Sec1"
    assert res.reranked[0].payload["url"] == "https://example.nih.gov/1"
