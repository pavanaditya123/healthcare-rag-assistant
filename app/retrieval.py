"""Hybrid retrieval: BM25 (sparse) + bge-small (dense) in Qdrant, fused with
Reciprocal Rank Fusion, then re-scored by a cross-encoder.

Each stage's output is returned separately so the evals can measure what
every stage contributes.
"""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.metrics import INFLIGHT, SHED, STAGE_LATENCY

DENSE, SPARSE = "dense", "bm25"


class Overloaded(Exception):
    """All inference slots busy for longer than QUEUE_TIMEOUT_S."""


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
    timings_ms: dict = field(default_factory=dict)  # per-stage latency of this query


class Encoders:
    """fastembed models (ONNX on CPU). Loaded once per process."""

    def __init__(self, s: Settings):
        from fastembed import SparseTextEmbedding, TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        threads = s.onnx_threads or None
        self.dense = TextEmbedding(s.dense_model, threads=threads)
        self.sparse = SparseTextEmbedding(s.sparse_model, threads=threads)
        self.reranker = TextCrossEncoder(s.rerank_model, threads=threads)
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
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.slots = threading.BoundedSemaphore(settings.max_inflight)

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
        # Payload indexes make metadata filters fast (no full scan).
        for key in ("source", "qtype"):
            self.client.create_payload_index(self.s.collection, key, models.PayloadSchemaType.KEYWORD)

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
                    payload={"chunk_id": c.id, "source": c.source, "url": c.url, "focus": c.focus,
                             "qtype": c.qtype, "text": c.text},
                )
                for c, d, sp in zip(batch, dense, sparse)
            ])

    def is_ready(self) -> bool:
        return self.client.collection_exists(self.s.collection)

    # ---------- search ----------
    def _to_hits(self, points) -> list[Hit]:
        return [Hit(p.payload["chunk_id"], p.score, p.payload) for p in points]

    def sparse_search(self, query: str, k: int, flt=None) -> list[Hit]:
        q = next(iter(self.enc.sparse.query_embed(query)))
        res = self.client.query_points(
            self.s.collection, using=SPARSE, limit=k, with_payload=True, query_filter=flt,
            query=models.SparseVector(indices=q.indices.tolist(), values=q.values.tolist()))
        return self._to_hits(res.points)

    def dense_search(self, query: str, k: int, flt=None) -> list[Hit]:
        q = next(iter(self.enc.dense.query_embed(query)))
        res = self.client.query_points(
            self.s.collection, using=DENSE, limit=k, with_payload=True, query_filter=flt,
            query=list(map(float, q)))
        return self._to_hits(res.points)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        # Cross-encoder cost grows with passage length, so only the start of each
        # passage is scored (the topic line + opening sentences carry most signal).
        cap = self.s.rerank_max_chars
        scores = list(self.enc.reranker.rerank(query, [h.payload["text"][:cap] for h in hits]))
        rescored = [Hit(h.id, float(s), h.payload) for h, s in zip(hits, scores)]
        return sorted(rescored, key=lambda h: h.score, reverse=True)

    def search(self, query: str, sources: list[str] | None = None, qtypes: list[str] | None = None) -> SearchResult:
        """Optional metadata filters, applied inside Qdrant to BOTH retrievers
        (pre-filtering, so we still get k candidates after filtering):
          sources: e.g. ["GHR", "GARD"] - only genetic / rare-disease pages
          qtypes:  e.g. ["treatment"]   - only "How is X treated?" sections
        """
        if not self.slots.acquire(timeout=self.s.queue_timeout_s):
            SHED.inc()
            raise Overloaded()
        INFLIGHT.inc()
        try:
            return self._search(query, sources, qtypes)
        finally:
            INFLIGHT.dec()
            self.slots.release()

    def _search(self, query, sources, qtypes) -> SearchResult:
        k, r = self.s.candidates_per_retriever, SearchResult()
        flt = make_filter(sources, qtypes)

        def timed(stage, fn):
            t0 = time.perf_counter()
            out = fn()
            elapsed = time.perf_counter() - t0
            STAGE_LATENCY.labels(stage).observe(elapsed)
            r.timings_ms[stage] = round(elapsed * 1000, 2)
            return out

        # BM25 and dense search are independent, so run them at the same time.
        # (ONNX inference and Qdrant network calls release the GIL.)
        sparse = self.pool.submit(timed, "sparse", lambda: self.sparse_search(query, k, flt))
        dense = self.pool.submit(timed, "dense", lambda: self.dense_search(query, k, flt))
        r.sparse, r.dense = sparse.result(), dense.result()
        r.fused = timed("fusion", lambda: rrf([r.sparse, r.dense])[:k])
        r.reranked = timed("rerank", lambda: self.rerank(query, r.fused))
        return r


def make_filter(sources=None, qtypes=None) -> models.Filter | None:
    must = []
    if sources:
        must.append(models.FieldCondition(key="source", match=models.MatchAny(any=list(sources))))
    if qtypes:
        must.append(models.FieldCondition(key="qtype", match=models.MatchAny(any=list(qtypes))))
    return models.Filter(must=must) if must else None


def make_client(s: Settings) -> QdrantClient:
    return QdrantClient(url=s.qdrant_url, timeout=10) if s.qdrant_url else QdrantClient(path=s.qdrant_path)
