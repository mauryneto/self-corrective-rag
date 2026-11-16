"""HTTP API.

    uv run uvicorn rag.api:app --reload
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.llm import LLMRefusal
from rag.pipeline.v0 import Answer, V0Pipeline

app = FastAPI(title="self-corrective-rag", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


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
def get_pipeline() -> V0Pipeline:
    return V0Pipeline()


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
    n_chunks, n_docs = get_pipeline().store.count()
    return {"status": "ok", "chunks": n_chunks, "norms": n_docs}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        return to_response(get_pipeline().answer(req.question))
    except LLMRefusal as exc:
        raise HTTPException(status_code=422, detail=f"model declined to answer: {exc}") from exc
