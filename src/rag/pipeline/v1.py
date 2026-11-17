"""V1: same generation as V0, better retrieval.

Hybrid search (bge-m3 vectors + BM25 over Portuguese-stemmed terms, fused with Reciprocal Rank
Fusion) gathers candidates; a multilingual cross-encoder (bge-reranker-v2-m3) rescores the best
20 and keeps the best ``top_k``. Chosen with the retrieval-only eval (``rag.eval.retrieval_eval``):
hybrid fusion alone did not beat vector search on this corpus, the reranker is what moves recall.
Prompt, model and number of passages in the prompt are identical to V0, so any difference in the
eval comes from retrieval.
"""

from __future__ import annotations

from rag.config import get_settings
from rag.embeddings import OllamaEmbedder
from rag.llm import LLM
from rag.pipeline.v0 import V0Pipeline
from rag.retrieval import Retriever
from rag.store import Hit, Store


class V1Pipeline(V0Pipeline):
    version = "v1"

    def __init__(self, store: Store | None = None, embedder: OllamaEmbedder | None = None,
                 llm: LLM | None = None, top_k: int | None = None, pool: int = 30, rerank_top: int = 20,
                 reranker=None):
        super().__init__(store=store, embedder=embedder, llm=llm, top_k=top_k)
        if reranker is None:
            from rag.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker()
        self.retriever = Retriever(self.store, self.embedder, mode="hybrid", top_k=self.top_k or get_settings().top_k,
                                   pool=pool, reranker=reranker, rerank_top=rerank_top)

    def retrieve(self, question: str) -> list[Hit]:
        return self.retriever.retrieve(question)
