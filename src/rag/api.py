"""HTTP API.

    uv run uvicorn rag.api:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.llm import LLMRefusal
from rag.pipeline.v0 import Answer, V0Pipeline

app = FastAPI(title="self-corrective-rag", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    version: Literal["v0", "v1"] = "v1"


class Source(BaseModel):
    index: int
    chunk_id: str
    norm: str
    unit: str
    url: str
    score: float
    cited: bool


class AskResponse(BaseModel):
    answer: str
    version: str
    sources: list[Source]
    latency_s: float
    retrieval_latency_s: float
    usage: dict


@lru_cache
def get_pipeline(version: str = "v1") -> V0Pipeline:
    if version == "v0":
        return V0Pipeline()
    from rag.pipeline.v1 import V1Pipeline

    return V1Pipeline()


def to_response(a: Answer) -> AskResponse:
    return AskResponse(
        answer=a.answer,
        version=a.version,
        sources=[
            Source(index=i, chunk_id=h.chunk_id, norm=h.doc_id.upper(), unit=h.unit, url=h.source_url,
                   score=round(h.score, 4), cited=i in a.cited)
            for i, h in enumerate(a.sources, 1)
        ],
        latency_s=round(a.latency_s, 3),
        retrieval_latency_s=round(a.retrieval_latency_s, 3),
        usage=a.usage.summary(),
    )


@app.get("/health")
def health() -> dict:
    n_chunks, n_docs = get_pipeline("v0").store.count()  # v0 loads no reranker
    return {"status": "ok", "chunks": n_chunks, "norms": n_docs}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        return to_response(get_pipeline(req.version).answer(req.question))
    except LLMRefusal as exc:
        raise HTTPException(status_code=422, detail=f"model declined to answer: {exc}") from exc
