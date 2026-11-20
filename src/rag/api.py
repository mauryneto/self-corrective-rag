"""HTTP API.

    uv run uvicorn rag.api:app --reload

``POST /ask`` returns the answer when it is finished. ``POST /ask/stream`` takes the same body and
answers with Server-Sent Events while the pipeline works:

- ``sources``: the passages that go into the prompt, as soon as retrieval is done;
- ``step`` (V2 only): the check that is about to run; ``revise`` means the text sent so far is a
  draft being rewritten, and the client should discard it;
- ``text``: the next piece of the answer;
- ``done``: what ``/ask`` returns, plus ``first_token_s``, the time until the first piece of text;
- ``error``: the pipeline failed after the stream had started.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.llm import LLMRefusal
from rag.pipeline.v0 import Answer, V0Pipeline
from rag.sse import format_event
from rag.store import Hit

app = FastAPI(title="self-corrective-rag", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    version: Literal["v0", "v1", "v2", "v3", "v4"] = "v4"


class Source(BaseModel):
    index: int
    chunk_id: str
    norm: str
    unit: str
    url: str
    score: float
    cited: bool
    text: str


class AskResponse(BaseModel):
    answer: str
    version: str
    sources: list[Source]
    latency_s: float
    retrieval_latency_s: float
    usage: dict


@lru_cache
def get_pipeline(version: str = "v4") -> V0Pipeline:
    if version == "v0":
        return V0Pipeline()
    if version == "v2":
        from rag.pipeline.v2 import V2Pipeline

        return V2Pipeline()
    if version == "v3":
        from rag.pipeline.v3 import V3Pipeline

        return V3Pipeline()
    if version == "v4":
        from rag.pipeline.v4 import V4Pipeline

        return V4Pipeline()
    from rag.pipeline.v1 import V1Pipeline

    return V1Pipeline()


def to_sources(hits: list[Hit], cited: list[int]) -> list[Source]:
    return [Source(index=i, chunk_id=h.chunk_id, norm=h.doc_id.upper(), unit=h.unit, url=h.source_url,
                   score=round(h.score, 4), cited=i in cited, text=h.text)
            for i, h in enumerate(hits, 1)]


def to_response(a: Answer) -> AskResponse:
    return AskResponse(
        answer=a.answer,
        version=a.version,
        sources=to_sources(a.sources, a.cited),
        latency_s=round(a.latency_s, 3),
        retrieval_latency_s=round(a.retrieval_latency_s, 3),
        usage=a.usage.summary(),
    )


class _ClientGone(Exception):
    """Raised inside the pipeline's thread when nobody reads the stream any more: it stops the
    model instead of letting it write an answer to the end for no one."""


def stream_events(pipeline: V0Pipeline, question: str) -> Iterator[tuple[str, dict]]:
    """The pipeline's events as they happen, ready to serialize.

    The pipeline runs in its own thread and reports through a queue; this generator reads the
    queue. Closing the generator (the client went away) makes the pipeline's next report raise,
    which closes the model's stream.
    """
    events: queue.Queue[tuple[str, object]] = queue.Queue()
    gone = threading.Event()
    t0 = time.perf_counter()

    def on_event(kind: str, data: dict) -> None:
        if gone.is_set():
            raise _ClientGone
        events.put((kind, data))

    def work() -> None:
        try:
            events.put(("done", pipeline.answer(question, on_event=on_event)))
        except _ClientGone:
            pass
        except LLMRefusal as exc:
            events.put(("error", {"detail": f"model declined to answer: {exc}"}))
        except Exception as exc:  # noqa: BLE001 - the stream has started: report it there
            events.put(("error", {"detail": f"{type(exc).__name__}: {exc}"}))

    threading.Thread(target=work, daemon=True).start()
    first_token_s: float | None = None
    try:
        while True:
            kind, data = events.get()
            if kind == "sources":
                yield kind, {"sources": [s.model_dump(exclude={"cited"}) for s in to_sources(data["hits"], [])],
                             "retrieval_latency_s": round(data["retrieval_latency_s"], 3),
                             "declined": data.get("declined", False)}
            elif kind == "done":
                yield kind, to_response(data).model_dump() | {
                    "first_token_s": None if first_token_s is None else round(first_token_s, 3)}
                return
            else:
                if kind == "text":
                    first_token_s = first_token_s or time.perf_counter() - t0
                elif kind == "step" and data["name"] == "revise":
                    first_token_s = None  # the draft is discarded: the answer starts again
                yield kind, data
                if kind == "error":
                    return
    finally:
        gone.set()


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline("v0")  # v0 loads no reranker
    n_chunks, n_docs = pipeline.store.count()
    return {"status": "ok", "chunks": n_chunks, "norms": n_docs, "model": pipeline.llm.model}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        return to_response(get_pipeline(req.version).answer(req.question))
    except LLMRefusal as exc:
        raise HTTPException(status_code=422, detail=f"model declined to answer: {exc}") from exc


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    pipeline = get_pipeline(req.version)  # models load here, so a failure to load is still an HTTP error
    body = (format_event(kind, data) for kind, data in stream_events(pipeline, req.question))
    return StreamingResponse(body, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
