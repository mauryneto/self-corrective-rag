"""V2: V1 plus a self-corrective loop, run by the same local model.

1. **Retrieve** as in V1 (hybrid search + reranker, top 6).
2. **Check the context.** The model lists what the question needs and whether the passages cover
   it. If something is missing, it writes up to 3 short search queries for just that part
   (decomposition), each retrieved like V1 with a smaller rerank pool; new passages are appended.
3. **Generate** with the same prompt as V0/V1.
4. **Check the answer.** The model compares each claim with the passages it cites. If it finds
   unsupported or contradicted claims, the answer is rewritten once with that list as feedback.

Both checks are structured outputs (JSON schema). If a check fails to parse, the pipeline carries
on as if it had passed: a broken check must never cost an answer. ``Answer.trace`` records what
each step did, including the first draft when it was rewritten, so the eval can measure each step.

With ``on_event`` (see ``rag.pipeline.v0``), each step is announced as a "step" event before it
starts. The draft is streamed as it is written; a "revise" step means the text sent so far is
discarded and the rewritten answer follows.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from rag.costs import Usage
from rag.embeddings import OllamaEmbedder
from rag.llm import LLM
from rag.pipeline.prompts import (
    PLAN_SYSTEM,
    VERIFY_SYSTEM,
    answer_prompt,
    plan_prompt,
    revise_prompt,
    verify_prompt,
)
from rag.pipeline.v0 import Answer, OnEvent, cited_indexes, streamed
from rag.pipeline.v1 import V1Pipeline
from rag.retrieval import Retriever
from rag.store import Hit, Store


class ContextCheck(BaseModel):
    sufficient: bool
    missing: str = ""
    queries: list[str] = Field(default_factory=list)


class AnswerCheck(BaseModel):
    supported: bool
    problems: list[str] = Field(default_factory=list)


class V2Pipeline(V1Pipeline):
    version = "v2"

    def __init__(self, store: Store | None = None, embedder: OllamaEmbedder | None = None,
                 llm: LLM | None = None, top_k: int | None = None, reranker=None,
                 max_queries: int = 3, per_query: int = 2, sub_rerank_top: int = 10, verify: bool = True):
        super().__init__(store=store, embedder=embedder, llm=llm, top_k=top_k, reranker=reranker)
        self.max_queries = max_queries
        self.per_query = per_query  # new passages kept per follow-up query
        self.verify = verify
        self.sub_retriever = Retriever(self.store, self.embedder, mode="hybrid", top_k=per_query + 2,
                                       pool=self.retriever.pool, reranker=self.retriever.reranker,
                                       rerank_top=sub_rerank_top)

    # -- step 2 ---------------------------------------------------------------------------------
    def check_context(self, question: str, hits: list[Hit], usage: Usage) -> ContextCheck | None:
        try:
            check, record = self.llm.complete_json(PLAN_SYSTEM, plan_prompt(question, [h.text for h in hits]),
                                                   ContextCheck, step="check_context")
        except Exception:  # noqa: BLE001 - see module docstring
            return None
        usage.add(record)
        return check

    def expand(self, hits: list[Hit], queries: list[str]) -> list[Hit]:
        """Append up to ``per_query`` passages per follow-up query that are not in the context yet."""
        out, seen = list(hits), {h.chunk_id for h in hits}
        for query in queries[: self.max_queries]:
            added = 0
            for h in self.sub_retriever.retrieve(query):
                if added == self.per_query:
                    break
                if h.chunk_id not in seen:
                    out.append(h)
                    seen.add(h.chunk_id)
                    added += 1
        return out

    def gather(self, question: str, usage: Usage) -> tuple[list[Hit], dict]:
        hits = self.retrieve(question)
        check = self.check_context(question, hits, usage)
        trace: dict = {"context_check": check.model_dump() if check else None}
        queries = [q.strip() for q in (check.queries if check and not check.sufficient else []) if q.strip()]
        n_first = len(hits)
        if queries:
            hits = self.expand(hits, queries)
        trace["added_passages"] = len(hits) - n_first
        return hits, trace

    # -- steps 3 and 4 --------------------------------------------------------------------------
    def answer(self, question: str, on_event: OnEvent | None = None) -> Answer:
        def step(name: str) -> None:
            if on_event:
                on_event("step", {"name": name})

        t0 = time.perf_counter()
        usage = Usage()
        step("check_context")
        hits, trace = self.gather(question, usage)
        t_retrieval = time.perf_counter() - t0
        texts = [h.text for h in hits]
        if on_event:
            on_event("sources", {"hits": hits, "retrieval_latency_s": t_retrieval})

        text, record = self.llm.complete(self.answer_system, answer_prompt(question, texts), step="generate",
                                         **streamed(on_event))
        usage.add(record)
        text = text.strip()

        trace["answer_check"] = None
        if self.verify and cited_indexes(text, len(hits)):  # nothing to check in a refusal
            step("check_answer")
            try:
                check, record = self.llm.complete_json(VERIFY_SYSTEM, verify_prompt(question, texts, text),
                                                       AnswerCheck, step="check_answer")
                usage.add(record)
            except Exception:  # noqa: BLE001 - see module docstring
                check = None
            trace["answer_check"] = check.model_dump() if check else None
            if check and not check.supported and check.problems:
                step("revise")
                revised, record = self.llm.complete(self.answer_system, revise_prompt(question, texts, text, check.problems),
                                                    step="revise", **streamed(on_event))
                usage.add(record)
                trace["draft"] = text
                trace["draft_cited"] = cited_indexes(text, len(hits))
                text = revised.strip()

        return Answer(
            question=question,
            answer=text,
            sources=hits,
            cited=cited_indexes(text, len(hits)),
            version=self.version,
            latency_s=time.perf_counter() - t0,
            usage=usage,
            retrieval_latency_s=t_retrieval,
            trace=trace,
        )
