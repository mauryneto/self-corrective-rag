"""V4: V3, faster. Same passages, same prompt, same answers; the reranker moves to the GPU.

Where the time went in V1 to V3 (median per question): 8 s in the reranker on the CPU, 0.2 s in
everything else of retrieval, 26 s in the model writing the answer.

- **Reranker on the GPU, half precision.** 0.3 s instead of 8 s for the 20 candidates. It returns
  the same six passages for all 180 eval questions; in 4 of them two neighbours swap places.
- **Room for it:** V4 asks the generator for a context window of 8,192 tokens instead of 16,384,
  which frees about 1.2 GB of VRAM and changes no answer (V4's longest prompt in the eval set has
  5,600 tokens). The other versions keep the larger window: V2's prompts can be twice as long.

What was tried and left out: more of the generator's layers on the GPU. 28 layers instead of 16
give 10 tokens per second instead of 8, push the embedding model out of the GPU, and change the
answers, so the gain would not be comparable with the other rows.
"""

from __future__ import annotations

from rag.embeddings import OllamaEmbedder
from rag.llm import LLM
from rag.pipeline.v3 import MIN_TOP_SCORE, V3Pipeline
from rag.store import Store

NUM_CTX = 8192


class V4Pipeline(V3Pipeline):
    version = "v4"

    def __init__(self, store: Store | None = None, embedder: OllamaEmbedder | None = None,
                 llm: LLM | None = None, top_k: int | None = None, reranker=None,
                 min_top_score: float = MIN_TOP_SCORE):
        if reranker is None:
            from rag.rerank import fast_reranker

            reranker = fast_reranker()
        super().__init__(store=store, embedder=embedder, llm=llm, top_k=top_k, reranker=reranker,
                         min_top_score=min_top_score)
        if getattr(self.llm, "num_ctx", 0) > NUM_CTX:  # local model only
            self.llm.num_ctx = NUM_CTX
