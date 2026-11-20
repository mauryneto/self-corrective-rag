"""V0: naive RAG. Vector search, top-k chunks into the prompt, one generation call.

This is the baseline every later version is measured against.

``answer(question, on_event=f)`` reports what happens while it happens, for a client that shows
the answer being written: ``f("sources", {...})`` when the passages are chosen, then
``f("text", {"text": piece})`` for each piece of the answer. It is the same code path as without
``on_event``, and returns the same ``Answer``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rag.config import get_settings
from rag.costs import Usage
from rag.embeddings import OllamaEmbedder
from rag.llm import LLM, get_llm
from rag.pipeline.prompts import ANSWER_SYSTEM, answer_prompt
from rag.store import Hit, Store

CITATION_RE = re.compile(r"\[(\d+)\]")
OnEvent = Callable[[str, dict], None]


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Hit]
    cited: list[int]  # 1-based indexes of the sources the answer cites
    version: str
    latency_s: float
    usage: Usage = field(default_factory=Usage)
    retrieval_latency_s: float = 0.0
    trace: dict = field(default_factory=dict)  # what the self-corrective steps did (V2+)


def cited_indexes(text: str, n_sources: int) -> list[int]:
    found = {int(m) for m in CITATION_RE.findall(text)}
    return sorted(i for i in found if 1 <= i <= n_sources)


def streamed(on_event: OnEvent | None) -> dict:
    """Keyword arguments for ``llm.complete``: with a listener, the text goes to it piece by piece
    as "text" events; without one, nothing is added and the call is the one the eval measures."""
    return {"on_text": lambda piece: on_event("text", {"text": piece})} if on_event else {}


class V0Pipeline:
    version = "v0"
    answer_system = ANSWER_SYSTEM  # the eval runner can swap it (--answer-prompt)

    def __init__(self, store: Store | None = None, embedder: OllamaEmbedder | None = None,
                 llm: LLM | None = None, top_k: int | None = None):
        self.store = store or Store()
        self.embedder = embedder or OllamaEmbedder()
        self.llm = llm or get_llm()
        self.top_k = top_k or get_settings().top_k

    def retrieve(self, question: str) -> list[Hit]:
        return self.store.vector_search(self.embedder.embed_one(question), self.top_k)

    def answer(self, question: str, on_event: OnEvent | None = None) -> Answer:
        t0 = time.perf_counter()
        return self.answer_from(question, self.retrieve(question), t0, on_event=on_event)

    def answer_from(self, question: str, hits: list[Hit], t0: float, trace: dict | None = None,
                    on_event: OnEvent | None = None) -> Answer:
        """The generation step, given the passages. ``t0``: when the question arrived."""
        t_retrieval = time.perf_counter() - t0
        if on_event:
            on_event("sources", {"hits": hits, "retrieval_latency_s": t_retrieval})

        usage = Usage()
        text, record = self.llm.complete(self.answer_system, answer_prompt(question, [h.text for h in hits]),
                                         step="generate", **streamed(on_event))
        usage.add(record)
        return Answer(
            question=question,
            answer=text.strip(),
            sources=hits,
            cited=cited_indexes(text, len(hits)),
            version=self.version,
            latency_s=time.perf_counter() - t0,
            usage=usage,
            retrieval_latency_s=t_retrieval,
            trace=trace or {},
        )
