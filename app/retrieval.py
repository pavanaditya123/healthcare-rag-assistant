"""Hybrid retrieval: BM25 (sparse) + bge-small (dense) in Qdrant, fused with
Reciprocal Rank Fusion, then re-scored by a cross-encoder.

Each stage's output is returned separately so the evals can measure what
every stage contributes.
"""
import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.metrics import STAGE_LATENCY

DENSE, SPARSE = "dense", "bm25"


@dataclass
class Hit:
    id: str
    score: float
    payload: dict


@dataclass
class SearchResult:
    sparse: list[Hit] = field(default_factory=list)
    dense: list[Hit] = field(default_factory=list)
    fused: list[Hit] = field(default_factory=list)
    reranked: list[Hit] = field(default_factory=list)


class Encoders:
    """fastembed models (ONNX on CPU). Loaded once per process."""

    def __init__(self, s: Settings):
        from fastembed import SparseTextEmbedding, TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.dense = TextEmbedding(s.dense_model)
        self.sparse = SparseTextEmbedding(s.sparse_model)
        self.reranker = TextCrossEncoder(s.rerank_model)
        self.dense_dim = len(next(iter(self.dense.embed(["dim probe"]))))


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank).

    Uses only ranks, so BM25 scores and cosine similarities never need to be
    put on the same scale. k=60 is the value from the original RRF paper.
    """
    scores, payloads = {}, {}
    for hits in rankings:
        for rank, h in enumerate(hits, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (k + rank)
            payloads[h.id] = h.payload
    order = sorted(scores, key=scores.get, reverse=True)
    return [Hit(i, scores[i], payloads[i]) for i in order]


def point_id(chunk_id: str) -> str:
    # Qdrant needs int/UUID ids; uuid5 is deterministic so re-ingesting overwrites.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class HybridRetriever:
    def __init__(self, client: QdrantClient, encoders, settings: Settings):
        self.client, self.enc, self.s = client, encoders, settings

    # ---------- indexing ----------
    def create_collection(self):
        if self.client.collection_exists(self.s.collection):
            self.client.delete_collection(self.s.collection)
        self.client.create_collection(
            self.s.collection,
            vectors_config={DENSE: models.VectorParams(size=self.enc.dense_dim, distance=models.Distance.COSINE)},
            # IDF modifier: Qdrant computes IDF over the collection, which turns
            # the term-frequency vectors from Qdrant/bm25 into real BM25 scores.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

    def index(self, chunks, batch_size: int = 256):
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c.text for c in batch]
            dense = list(self.enc.dense.passage_embed(texts))
            sparse = list(self.enc.sparse.passage_embed(texts))
            self.client.upsert(self.s.collection, points=[
                models.PointStruct(
                    id=point_id(c.id),
                    vector={DENSE: list(map(float, d)),
                            SPARSE: models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())},
                    payload={"chunk_id": c.id, "source": c.source, "url": c.url, "focus": c.focus, "text": c.text},
                )
                for c, d, sp in zip(batch, dense, sparse)
            ])

    def is_ready(self) -> bool:
        return self.client.collection_exists(self.s.collection)

    # ---------- search ----------
    def _to_hits(self, points) -> list[Hit]:
        return [Hit(p.payload["chunk_id"], p.score, p.payload) for p in points]

    def sparse_search(self, query: str, k: int) -> list[Hit]:
        with STAGE_LATENCY.labels("sparse").time():
            q = next(iter(self.enc.sparse.query_embed(query)))
            res = self.client.query_points(
                self.s.collection, using=SPARSE, limit=k, with_payload=True,
                query=models.SparseVector(indices=q.indices.tolist(), values=q.values.tolist()))
        return self._to_hits(res.points)

    def dense_search(self, query: str, k: int) -> list[Hit]:
        with STAGE_LATENCY.labels("dense").time():
            q = next(iter(self.enc.dense.query_embed(query)))
            res = self.client.query_points(
                self.s.collection, using=DENSE, limit=k, with_payload=True, query=list(map(float, q)))
        return self._to_hits(res.points)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        with STAGE_LATENCY.labels("rerank").time():
            scores = list(self.enc.reranker.rerank(query, [h.payload["text"] for h in hits]))
        rescored = [Hit(h.id, float(s), h.payload) for h, s in zip(hits, scores)]
        return sorted(rescored, key=lambda h: h.score, reverse=True)

    def search(self, query: str) -> SearchResult:
        k = self.s.candidates_per_retriever
        r = SearchResult(sparse=self.sparse_search(query, k), dense=self.dense_search(query, k))
        with STAGE_LATENCY.labels("fusion").time():
            r.fused = rrf([r.sparse, r.dense])[:k]
        r.reranked = self.rerank(query, r.fused)
        return r


def make_client(s: Settings) -> QdrantClient:
    return QdrantClient(url=s.qdrant_url, timeout=10) if s.qdrant_url else QdrantClient(path=s.qdrant_path)
