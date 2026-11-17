"""Cross-encoder reranker (multilingual), run locally.

A cross-encoder reads the question and one candidate passage together and scores how well the
passage answers it: slower than comparing embeddings, much more precise. It rescoring a pool of
~30 candidates from the hybrid search and keeping the best few is the classic second stage.
"""

from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_RERANKER = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


@lru_cache(maxsize=2)
def _load(model_name: str, device: str | None):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=512, device=device)


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None, batch_size: int = 16):
        self.model_name = model_name
        self.model = _load(model_name, device)
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]
