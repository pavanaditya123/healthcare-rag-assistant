"""Tiny fake models so tests run in seconds with no downloads."""
import re
import zlib
from types import SimpleNamespace

import numpy as np

from app.retrieval import Hit, SearchResult


def words(text):
    return re.findall(r"[a-z]+", text.lower())


class FakeDense:
    dim = 64

    def _vec(self, text):
        v = np.zeros(self.dim)
        for w in words(text):
            v[zlib.crc32(w.encode()) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed(self, texts):
        return (self._vec(t) for t in texts)

    passage_embed = embed

    def query_embed(self, text):
        return iter([self._vec(text)])


class FakeSparse:
    def _vec(self, text):
        ids = sorted({zlib.crc32(w.encode()) % 100_000 for w in words(text)})
        return SimpleNamespace(indices=np.array(ids), values=np.ones(len(ids)))

    def passage_embed(self, texts):
        return (self._vec(t) for t in texts)

    def query_embed(self, text):
        return iter([self._vec(text)])


class FakeReranker:
    def rerank(self, query, docs):
        q = set(words(query))
        return (float(len(q & set(words(d)))) for d in docs)


class FakeEncoders:
    def __init__(self):
        self.dense, self.sparse, self.reranker = FakeDense(), FakeSparse(), FakeReranker()
        self.dense_dim = FakeDense.dim


class FakeRetriever:
    def __init__(self, score=5.0, ready=True, overloaded=False):
        self.score, self.ready, self.overloaded, self.queries = score, ready, overloaded, []

    def search(self, q, sources=None, qtypes=None):
        if self.overloaded:
            from app.retrieval import Overloaded
            raise Overloaded()
        self.queries.append(q)
        self.filters = (sources, qtypes)
        payload = {"focus": "Glaucoma", "url": "https://nei.nih.gov/glaucoma", "text": "Glaucoma\nGlaucoma damages the optic nerve."}
        return SearchResult(reranked=[Hit("NEI_0000001_Sec1", self.score, payload)])

    def is_ready(self):
        return self.ready


class FakeLLM:
    def __init__(self, text="Glaucoma damages the optic nerve. [1]", error=None):
        self.text, self.error, self.calls = text, error, 0

    def complete(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return self.text, "fake-model"

    def stream(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        for word in self.text.split(" "):
            yield word + " ", "fake-model"
