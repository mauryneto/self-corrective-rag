"""V3's scope gate with in-memory fakes: no database, no Ollama."""

from rag.costs import CallRecord
from rag.eval.scope_gate_eval import apply_rule, cross_validate, top_score
from rag.pipeline.v1 import V1Pipeline
from rag.pipeline.v3 import DECLINED, V3Pipeline
from rag.store import Hit

HITS = [Hit(f"rn-566#{i}", "rn-566", f"Art. {i}", f"RN nº 566 — Art. {i}\ntexto {i}", "u", 0.0) for i in range(1, 4)]


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

    def __init__(self):
        self.calls = []

    def complete(self, system, prompt, step, max_tokens=2048):
        self.calls.append((system, prompt))
        return "O prazo é de 7 dias úteis [1].", CallRecord(step, self.model, 100, 10, 0.1)


def pipeline(cls, scores, **kw):
    return cls(store=FakeStore(), embedder=FakeEmbedder(), llm=FakeLLM(), top_k=3, reranker=FakeReranker(scores), **kw)


def test_declines_without_calling_the_model_when_no_passage_is_about_the_question():
    p = pipeline(V3Pipeline, [0.001, 0.04, 0.002])
    a = p.answer("Como declaro o plano no Imposto de Renda?")
    assert a.answer == DECLINED and a.cited == []
    assert p.llm.calls == [] and a.usage.summary()["llm_calls"] == 0
    assert [h.chunk_id for h in a.sources] == ["rn-566#2", "rn-566#3", "rn-566#1"]  # kept for the retrieval metrics
    assert a.trace["scope_gate"] == {"top_score": 0.04, "min_top_score": 0.05, "declined": True}
    assert a.latency_s == a.retrieval_latency_s


def test_answers_exactly_like_v1_when_a_passage_passes_the_gate():
    scores = [0.2, 0.9, 0.01]
    v3, v1 = pipeline(V3Pipeline, scores), pipeline(V1Pipeline, scores)
    a3, a1 = v3.answer("Qual o prazo?"), v1.answer("Qual o prazo?")
    assert v3.llm.calls == v1.llm.calls and len(v3.llm.calls) == 1  # same system prompt, same passages, same order
    assert (a3.answer, a3.cited, a3.sources) == (a1.answer, a1.cited, a1.sources)
    assert a3.version == "v3" and a3.trace["scope_gate"]["declined"] is False


def test_the_threshold_is_a_parameter():
    assert pipeline(V3Pipeline, [0.2, 0.3, 0.1], min_top_score=0.5).answer("Qual o prazo?").answer == DECLINED


def row(kind, score, verdict="correct", refused=False):
    return {"type": kind, "verdict": verdict, "refused": refused, "sources": [{"score": score}, {"score": score / 2}]}


ROWS = {
    "a": row("direct", 0.9),
    "b": row("multi_hop", 0.2),
    "c": row("direct", 0.3, verdict="incorrect"),
    "d": row("out_of_scope", 0.01, refused=True),
    "e": row("out_of_scope", 0.25, verdict="incorrect"),
    "f": row("unanswerable", 0.6, refused=True),
}


def test_apply_rule_counts_what_a_gate_would_change():
    assert top_score(ROWS["a"]) == 0.9
    no_gate = apply_rule(ROWS, lambda i: False)
    assert (no_gate["declined"], no_gate["correct"]) == (2, 2)
    gate = apply_rule(ROWS, lambda i: top_score(ROWS[i]) < 0.28)
    assert (gate["gated_out_of_scope"], gate["gated_unanswerable"], gate["gated_answerable"]) == (2, 0, 1)
    assert gate["gated_answerable_correct"] == 1
    assert (gate["declined"], gate["correct"]) == (3, 1)  # catches "e", loses "b"


def test_cross_validation_picks_the_threshold_without_the_held_out_fold():
    folds = cross_validate(ROWS, folds=3)
    assert sum(f["to_decline"] for f in folds) == 3
    for f in folds:
        assert f["threshold"] in (0.1, 0.15, 0.45)  # half of 0.2; of 0.3 or 0.9 when "b" (and "c") are held out
