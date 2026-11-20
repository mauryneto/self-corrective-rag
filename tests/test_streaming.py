"""Streaming, from the model's HTTP stream to the API's events, with in-memory fakes."""

import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from rag import api
from rag.costs import CallRecord
from rag.llm import OllamaLLM, StreamInterrupted
from rag.pipeline.v0 import V0Pipeline
from rag.pipeline.v2 import AnswerCheck, ContextCheck, V2Pipeline
from rag.pipeline.v3 import DECLINED, V3Pipeline
from rag.sse import format_event, parse_events
from rag.store import Hit

HITS = [Hit(f"rn-566#{i}", "rn-566", f"Art. {i}", f"RN nº 566 — Art. {i}\ntexto {i}", f"u{i}", 0.9 - i / 10)
        for i in range(1, 4)]
PIECES = ["O prazo é ", "de 7 dias ", "úteis [1]."]


class FakeStore:
    def vector_search(self, embedding, k):
        return HITS[:k]

    def lexical_search(self, query, k):
        return []


class FakeEmbedder:
    def embed_one(self, text):
        return [0.0]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, texts):
        return self.scores[: len(texts)]


class StreamingLLM:
    """Writes each answer in the pieces given, like a model that streams."""

    model = "local"

    def __init__(self, *answers, context=None, answer_check=None):
        self.answers = list(answers) or [PIECES]
        self.context, self.answer_check = context, answer_check
        self.calls = 0

    def complete(self, system, prompt, step, max_tokens=2048, on_text=None):
        self.calls += 1
        pieces = self.answers.pop(0)
        for piece in pieces if on_text else []:
            on_text(piece)
        return "".join(pieces), CallRecord(step, self.model, 100, 10, 0.1)

    def complete_json(self, system, prompt, schema, step, max_tokens=1024):
        return (self.context if schema is ContextCheck else self.answer_check), CallRecord(step, self.model, 100, 10, 0.1)


def recorder():
    seen = []
    return seen, lambda kind, data: seen.append((kind, data))


def v0(llm=None):
    return V0Pipeline(store=FakeStore(), embedder=FakeEmbedder(), llm=llm or StreamingLLM(), top_k=3)


# -- the wire format --------------------------------------------------------------------------------

def test_events_survive_the_round_trip_with_accents_and_line_breaks():
    events = [("sources", {"sources": [{"norm": "RN-566", "text": "Art. 3\nprazo de atenção"}]}),
              ("text", {"text": "O prazo é\n"}), ("done", {"answer": "O prazo é\n7 dias [1]."})]
    wire = "".join(format_event(kind, data) for kind, data in events)
    assert list(parse_events(wire.splitlines(keepends=True))) == events


# -- the model's stream ------------------------------------------------------------------------------

def ollama(lines: list[dict]):
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        if not body["stream"]:
            return httpx.Response(200, json={"message": {"content": "ok"}})
        return httpx.Response(200, content="\n".join(json.dumps(line) for line in lines) + "\n")

    llm = OllamaLLM(model="qwen2.5:14b", url="http://ollama")
    llm.client = httpx.Client(transport=httpx.MockTransport(handler))
    llm.reproducible = True
    return llm, sent


def test_a_streamed_call_hands_over_each_piece_and_returns_what_the_plain_call_returns():
    llm, sent = ollama([{"message": {"content": p}} for p in PIECES]
                       + [{"message": {"content": ""}, "done": True, "prompt_eval_count": 900, "eval_count": 12}])
    got = []
    text, record = llm.complete("sistema", "pergunta", step="generate", on_text=got.append)
    assert got == PIECES and text == "".join(PIECES)
    assert (record.input_tokens, record.output_tokens, record.step) == (900, 12, "generate")
    reset, real = sent  # the prompt cache is reset before a streamed call too
    assert not reset["stream"] and real["stream"]
    assert real["options"]["temperature"] == 0 and real["options"]["seed"] == 0


def test_a_stream_that_breaks_after_some_text_is_not_tried_again():
    llm, sent = ollama([{"message": {"content": "O prazo é "}}])  # no final message
    got = []
    with pytest.raises(StreamInterrupted):
        llm.complete("sistema", "pergunta", step="generate", on_text=got.append)
    assert got == ["O prazo é "] and len(sent) == 2  # one reset, one attempt


# -- the pipelines' events ---------------------------------------------------------------------------

def test_v0_reports_the_sources_then_the_text_and_returns_the_same_answer():
    seen, on_event = recorder()
    streamed, plain = v0().answer("Qual o prazo?", on_event=on_event), v0().answer("Qual o prazo?")
    assert [kind for kind, _ in seen] == ["sources", "text", "text", "text"]
    assert seen[0][1]["hits"] == HITS
    assert "".join(data["text"] for kind, data in seen if kind == "text") == streamed.answer == plain.answer
    assert streamed.cited == plain.cited == [1]


def test_a_question_the_gate_declines_is_reported_without_calling_the_model():
    llm = StreamingLLM()
    p = V3Pipeline(store=FakeStore(), embedder=FakeEmbedder(), llm=llm, top_k=3, reranker=FakeReranker([0.01, 0.02, 0.0]))
    seen, on_event = recorder()
    a = p.answer("Como declaro o plano no Imposto de Renda?", on_event=on_event)
    assert [kind for kind, _ in seen] == ["sources", "text"]
    assert seen[0][1]["declined"] and seen[1][1]["text"] == a.answer == DECLINED and llm.calls == 0


def test_v2_announces_each_check_and_a_rewrite_before_the_new_text():
    llm = StreamingLLM(["Rascunho ", "[1]."], ["Resposta ", "revisada [2]."], context=ContextCheck(sufficient=True),
                       answer_check=AnswerCheck(supported=False, problems=["o prazo não está no trecho [1]"]))
    p = V2Pipeline(store=FakeStore(), embedder=FakeEmbedder(), llm=llm, top_k=3, reranker=FakeReranker([0.9, 0.8, 0.7]))
    seen, on_event = recorder()
    a = p.answer("pergunta", on_event=on_event)
    assert [data.get("name", kind) for kind, data in seen] == [
        "check_context", "sources", "text", "text", "check_answer", "revise", "text", "text"]
    assert a.answer == "Resposta revisada [2]." and a.trace["draft"] == "Rascunho [1]."


# -- the API -----------------------------------------------------------------------------------------

def ask_stream(monkeypatch, pipeline, **body):
    monkeypatch.setattr(api, "get_pipeline", lambda version="v4": pipeline)
    r = TestClient(api.app).post("/ask/stream", json={"question": "Qual o prazo?"} | body)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    return list(parse_events(r.text.splitlines(keepends=True)))


def test_ask_stream_sends_sources_text_and_the_full_response_at_the_end(monkeypatch):
    events = ask_stream(monkeypatch, v0())
    assert [kind for kind, _ in events] == ["sources", "text", "text", "text", "done"]
    sources, done = events[0][1], events[-1][1]
    assert [s["chunk_id"] for s in sources["sources"]] == [h.chunk_id for h in HITS]
    assert "cited" not in sources["sources"][0] and not sources["declined"]  # not known before the answer exists
    assert done["answer"] == "".join(PIECES) and [s["cited"] for s in done["sources"]] == [True, False, False]
    assert done["sources"][0]["text"] == HITS[0].text
    assert 0 <= done["first_token_s"] <= done["latency_s"] + 0.01 and done["usage"]["llm_calls"] == 1


def test_ask_stream_and_ask_give_the_same_answer(monkeypatch):
    done = ask_stream(monkeypatch, v0(StreamingLLM(PIECES, PIECES)))[-1][1]
    plain = TestClient(api.app).post("/ask", json={"question": "Qual o prazo?"}).json()
    for key in ("answer", "version", "sources"):
        assert done[key] == plain[key]


def test_a_failure_after_the_stream_started_is_an_error_event(monkeypatch):
    class Broken(StreamingLLM):
        def complete(self, system, prompt, step, max_tokens=2048, on_text=None):
            on_text("O prazo ")
            raise StreamInterrupted("connection lost")

    events = ask_stream(monkeypatch, v0(Broken()))
    assert [kind for kind, _ in events] == ["sources", "text", "error"]
    assert "connection lost" in events[-1][1]["detail"]


def test_when_the_client_stops_reading_the_pipeline_stops_writing():
    stopped = threading.Event()

    class Endless:
        def answer(self, question, on_event=None):
            try:
                while True:
                    on_event("text", {"text": "x"})
                    time.sleep(0.001)
            finally:
                stopped.set()

    events = api.stream_events(Endless(), "Qual o prazo?")
    assert next(events) == ("text", {"text": "x"})
    events.close()
    assert stopped.wait(timeout=2)


# -- the measurement ---------------------------------------------------------------------------------

def test_stream_check_times_only_the_questions_the_model_answered():
    from rag.eval.stream_check import summarize, timed_answer

    row = timed_answer(v0(), "Qual o prazo?")
    assert row["answer"] == "".join(PIECES) and row["llm_calls"] == 1
    assert 0 <= row["retrieval_s"] <= row["first_token_s"] <= row["total_s"]

    def fake(first, total, calls=1, same=True):
        return {"llm_calls": calls, "same_passages": same, "same_answer": same, "retrieval_s": 0.5,
                "first_token_s": first, "total_s": total, "output_tokens": 40 * calls}

    s = summarize([fake(4.0, 20.0), fake(6.0, 30.0), fake(5.0, 25.0, same=False), fake(0.5, 0.5, calls=0)])
    assert (s["questions"], s["answered_by_the_model"], s["declined_by_the_gate"]) == (4, 3, 1)
    assert (s["same_passages_as_stored"], s["identical_answer_of_those"]) == (3, 3)
    assert s["first_token_s"]["median"] == 5.0 and s["total_s"]["median"] == 25.0
    assert s["gate_declines_total_s_median"] == 0.5
