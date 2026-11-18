"""Reference rows for the results table: the answering model with no retrieval (floor) and with
perfect retrieval (ceiling).

The versions (V0, V1, ...) change the architecture around a fixed model. These two rows bound what
any architecture can do with that model, so a version's score can be read as a share of the
possible gain:

- ``floor``: the model answers from what it knows, no passages at all.
- ``ceiling``: the gold passages come first and V1's best other passages fill the prompt up to
  the same size as V1's. This is V1 with a retriever that never misses.
- ``ceiling_gold``: only the gold passages. Compared with ``ceiling``, it shows what the other
  passages in the prompt cost.

They only work on the eval set (the ceiling needs the gold sources) and are not served by the API.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rag.config import ROOT
from rag.costs import Usage
from rag.embeddings import OllamaEmbedder
from rag.eval.check_questions import normalize
from rag.eval.dataset import Question
from rag.llm import LLM, get_llm
from rag.pipeline.v0 import Answer, V0Pipeline
from rag.store import Hit, Store

# ANSWER_SYSTEM without the passages: same role, same "say so instead of guessing", same length.
CLOSED_BOOK_SYSTEM = """\
Você é um assistente que responde perguntas sobre a regulação da saúde suplementar no Brasil \
(normas da ANS). Nenhum trecho de norma foi fornecido: responda com o que você sabe.

Regras:
- Se você não souber a resposta com segurança, diga isso claramente em vez de supor.
- Responda em português, de forma direta, em até 3 parágrafos curtos."""


class FloorPipeline:
    """No retrieval: what the model knows about the norms on its own."""

    version = "floor"

    def __init__(self, llm: LLM | None = None):
        self.llm = llm or get_llm()

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        usage = Usage()
        text, record = self.llm.complete(CLOSED_BOOK_SYSTEM, f"Pergunta: {question}", step="generate")
        usage.add(record)
        return Answer(question=question, answer=text.strip(), sources=[], cited=[], version=self.version,
                      latency_s=time.perf_counter() - t0, usage=usage)


def gold_hits(store: Store, question: Question) -> list[Hit]:
    """The chunks a perfect retriever would return: for each gold article, the part that holds the
    evidence quote (every part of the article if the quote is not found in a single one)."""
    out: dict[str, Hit] = {}
    for g in question.gold:
        parts = store.unit_hits(g.doc, g.unit)
        quote = normalize(g.evidence)
        holders = [h for h in parts if quote and quote in normalize(h.text)]
        for h in holders or parts:
            out.setdefault(h.chunk_id, h)
    return list(out.values())


def load_replay(run: str = "v1_repro") -> dict[str, list[str]]:
    """Question id -> chunk ids a stored run retrieved, in rank order."""
    path = Path(ROOT / "eval" / "runs" / f"{run}.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: [s["chunk_id"] for s in r["sources"]] for r in rows}


class CeilingPipeline(V0Pipeline):
    """Perfect retrieval: gold passages first. With ``replay`` (a stored run's retrieval), its
    other passages fill the prompt to ``top_k``; a question without gold sources gets that run's
    passages unchanged. Without ``replay``, the prompt holds the gold passages only."""

    def __init__(self, questions: list[Question], replay: dict[str, list[str]] | None, store: Store | None = None,
                 embedder: OllamaEmbedder | None = None, llm: LLM | None = None, top_k: int | None = None):
        super().__init__(store=store, embedder=embedder, llm=llm, top_k=top_k)
        self.by_text = {q.question: q for q in questions}
        self.replay = replay
        self.version = "ceiling" if replay is not None else "ceiling_gold"

    def retrieve(self, question: str) -> list[Hit]:
        q = self.by_text.get(question)
        if q is None:
            raise KeyError("the ceiling needs gold sources: it only answers questions of the eval set")
        gold = gold_hits(self.store, q)
        if self.replay is None:
            return gold
        seen = {h.chunk_id for h in gold}
        fill = [h for h in self.store.get_hits(self.replay[q.id]) if h.chunk_id not in seen]
        return (gold + fill)[: max(self.top_k, len(gold))]
