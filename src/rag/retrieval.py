"""Retrieval strategies, shared by the pipeline versions and the retrieval-only eval.

- ``vector``: cosine similarity on bge-m3 embeddings (V0).
- ``lexical``: Postgres full-text search (Portuguese stemming), good at exact terms, numbers and
  article references that embeddings blur.
- ``hybrid``: both lists fused with Reciprocal Rank Fusion, which needs no score calibration
  between the two very different scales.
- optional ``reranker``: a cross-encoder rescoring a wider candidate pool, keeping the best k.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from rag.store import Hit, Store


class Embedder(Protocol):
    def embed_one(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def rrf_fuse(rankings: list[list[Hit]], k: int = 60, weights: list[float] | None = None) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of w / (k + rank). The fused score
    replaces the hit's original score; ties keep the first list's order."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    first: dict[str, Hit] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, 1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + w / (k + rank)
            first.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: -scores[cid])
    return [replace(first[cid], score=scores[cid]) for cid in ordered]


@dataclass
class Retriever:
    store: Store
    embedder: Embedder
    mode: str = "hybrid"  # vector | lexical | hybrid
    top_k: int = 6
    pool: int = 30  # candidates per list before fusion
    reranker: Reranker | None = None
    rerank_top: int = 30  # fused candidates sent to the reranker (its cost grows linearly with this)
    lexical_weight: float = 1.0  # weight of the BM25 list in the fusion (the vector list weighs 1.0)
    rrf_k: int = 60

    def candidates(self, query: str) -> list[Hit]:
        if self.mode == "vector":
            return self.store.vector_search(self.embedder.embed_one(query), self.pool)
        if self.mode == "lexical":
            return self.store.lexical_search(query, self.pool)
        if self.mode == "hybrid":
            return rrf_fuse([self.store.vector_search(self.embedder.embed_one(query), self.pool),
                             self.store.lexical_search(query, self.pool)],
                            k=self.rrf_k, weights=[1.0, self.lexical_weight])
        raise ValueError(f"unknown retrieval mode: {self.mode}")

    def retrieve(self, query: str) -> list[Hit]:
        hits = self.candidates(query)
        if self.reranker is None or not hits:
            return hits[: self.top_k]
        hits = hits[: self.rerank_top]
        scores = self.reranker.score(query, [h.text for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda p: -p[1])
        return [replace(h, score=float(s)) for h, s in ranked[: self.top_k]]
