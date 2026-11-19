"""V3: V1 plus a scope gate that declines before the model is called.

The cross-encoder already scores every passage against the question. When even the best passage
scores below ``min_top_score``, nothing in the corpus is about the question: the pipeline returns
a fixed refusal and makes no LLM call.

Why this signal and not V2's context check (``rag.eval.scope_gate_eval``, on the stored runs):

- The best reranker score separates out-of-scope questions cleanly: 18 of 20 score below 0.05,
  and no answerable question scores below 0.11.
- "The passages don't cover this" from V2's context check flags 58 of the 60 questions to
  decline and also 44 of the 120 answerable ones, 25 of them answered correctly. As a rule for
  declining it costs far more than it catches.

``MIN_TOP_SCORE`` is half the lowest best-passage score among the 120 answerable questions of the
eval set, rounded down. It was chosen on that same set; in a 5-fold check the rule picked on four
folds declined no answerable question of the fifth.

What the gate does not do: questions about health plans that the norms do not answer look like
answerable ones to the reranker (their passages are on topic). Those still reach the model, whose
prompt tells it to say when the passages are not enough.
"""

from __future__ import annotations

import time

from rag.embeddings import OllamaEmbedder
from rag.llm import LLM
from rag.pipeline.v0 import Answer
from rag.pipeline.v1 import V1Pipeline
from rag.store import Store

MIN_TOP_SCORE = 0.05
DECLINED = ("Não encontrei nas normas da ANS trechos que tratem dessa pergunta, por isso não tenho como "
            "respondê-la com base nelas.")


class V3Pipeline(V1Pipeline):
    version = "v3"

    def __init__(self, store: Store | None = None, embedder: OllamaEmbedder | None = None,
                 llm: LLM | None = None, top_k: int | None = None, reranker=None,
                 min_top_score: float = MIN_TOP_SCORE):
        super().__init__(store=store, embedder=embedder, llm=llm, top_k=top_k, reranker=reranker)
        self.min_top_score = min_top_score

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        top = max((h.score for h in hits), default=0.0)
        declined = top < self.min_top_score
        trace = {"scope_gate": {"top_score": round(top, 4), "min_top_score": self.min_top_score, "declined": declined}}
        if not declined:
            return self.answer_from(question, hits, t0, trace)
        elapsed = time.perf_counter() - t0
        # the passages stay in the record: they are what the retrieval metrics are computed on
        return Answer(question=question, answer=DECLINED, sources=hits, cited=[], version=self.version,
                      latency_s=elapsed, retrieval_latency_s=elapsed, trace=trace)
