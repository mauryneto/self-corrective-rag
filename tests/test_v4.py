"""V4 and the eval runner's reuse of stored answers, with in-memory fakes: no database, no Ollama, no GPU."""

import json

import pytest

from rag.costs import CallRecord
from rag.eval import run
from rag.eval.dataset import Question
from rag.pipeline.v3 import V3Pipeline
from rag.pipeline.v4 import NUM_CTX, V4Pipeline
from rag.store import Hit

HITS = [Hit(f"rn-566#{i}", "rn-566", f"Art. {i}", f"RN nº 566 — Art. {i}\ntexto {i}", "u", 0.0) for i in range(1, 4)]
QUESTION = Question(id="q1", type="direct", question="Qual o prazo?", reference_answer="Sete dias úteis.",
                    gold=[{"doc": "rn-566", "unit": "Art. 2"}])


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


class FakeLLM:
    model = "local"
    num_ctx = 16384

    def complete(self, system, prompt, step, max_tokens=2048):
        return "O prazo é de 7 dias úteis [1].", CallRecord(step, self.model, 100, 10, 0.1)


def pipeline(cls=V4Pipeline, scores=(0.2, 0.9, 0.5)):
    return cls(store=FakeStore(), embedder=FakeEmbedder(), llm=FakeLLM(), top_k=3, reranker=FakeReranker(list(scores)))


def stored(chunk_ids=("rn-566#2", "rn-566#3", "rn-566#1"), declined=False):
    return {"id": "q1", "answer": "Sete dias úteis [1].", "cited": [1], "verdict": "correct", "refused": False,
            "citations_supported": True, "judge_reasoning": "ok", "latency_s": 40.0, "retrieval_latency_s": 14.0,
            "usage": {"llm_calls": 1, "input_tokens": 2000, "output_tokens": 100, "cost_usd": 0.0, "llm_latency_s": 26.0},
            "sources": [{"chunk_id": c} for c in chunk_ids], "trace": {"scope_gate": {"declined": declined}}}


def test_v4_is_v3_with_a_smaller_context_window():
    p = pipeline()
    assert isinstance(p, V3Pipeline) and p.version == "v4"
    assert p.llm.num_ctx == NUM_CTX
    a = p.answer("Qual o prazo?")
    assert a.cited == [1] and a.trace["scope_gate"]["declined"] is False


def test_an_answer_is_reused_only_when_the_prompt_would_be_the_same():
    p = pipeline()
    hits = p.retrieve("Qual o prazo?")
    assert run.same_prompt(p, hits, stored())
    assert not run.same_prompt(p, hits, stored(("rn-566#3", "rn-566#2", "rn-566#1")))  # same passages, other order
    assert not run.same_prompt(p, hits, stored(declined=True))  # the stored answer is the gate's fixed refusal
    low = pipeline(scores=(0.01, 0.04, 0.02))
    assert not run.same_prompt(low, low.retrieve("Qual o prazo?"), stored())  # this version declines it


def test_a_reused_record_keeps_the_answer_and_takes_this_runs_retrieval():
    p = pipeline()
    rec = run.reused_record(QUESTION, p, p.retrieve("Qual o prazo?"), stored(), "v3_repro", retrieval_s=0.4)
    assert (rec["answer"], rec["verdict"], rec["cited"], rec["reused_from"]) == ("Sete dias úteis [1].", "correct", [1],
                                                                                 "v3_repro")
    assert rec["retrieval_latency_s"] == 0.4 and rec["latency_s"] == 26.4  # this retrieval + the stored generation
    assert rec["usage"]["llm_latency_s"] == 26.0 and rec["usage"]["input_tokens"] == 2000
    assert [s["chunk_id"] for s in rec["sources"]] == ["rn-566#2", "rn-566#3", "rn-566#1"]
    assert rec["sources"][0]["score"] == 0.9 and rec["recall_at_5"] == 1.0
    assert rec["trace"]["scope_gate"]["declined"] is False


def write_run(tmp_path, monkeypatch, version):
    monkeypatch.setattr(run, "RUNS_DIR", tmp_path)
    rows = [stored(), stored() | {"id": "q2", "verdict": None}]
    (tmp_path / "old.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "old.summary.json").write_text(json.dumps({"version": version}), encoding="utf-8")


def test_only_graded_answers_of_a_compatible_run_can_be_reused(tmp_path, monkeypatch):
    write_run(tmp_path, monkeypatch, "v3")
    assert list(run.load_reusable("old", "default")) == ["q1"]  # q2 has no verdict
    with pytest.raises(SystemExit):
        run.load_reusable("old", "complete")  # another answering prompt
    write_run(tmp_path, monkeypatch, "v2")
    with pytest.raises(SystemExit):
        run.load_reusable("old", "default")  # V2 rewrites answers and adds passages
