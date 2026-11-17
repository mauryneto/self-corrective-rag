"""V0 pipeline and API with in-memory fakes: no database, no Ollama, no API key needed."""

from fastapi.testclient import TestClient

from rag import api
from rag.costs import CallRecord, cost_usd
from rag.pipeline.v0 import V0Pipeline, cited_indexes
from rag.store import Hit

HITS = [
    Hit("rn-566#3", "rn-566", "Art. 3", "RN nº 566 — Art. 3\nConsulta básica: até 7 dias úteis.", "u1", 0.82),
    Hit("rn-566#4", "rn-566", "Art. 4", "RN nº 566 — Art. 4\nUrgência: imediato.", "u2", 0.75),
]


class FakeStore:
    def vector_search(self, embedding, k):
        return HITS[:k]

    def count(self):
        return (2, 1)


class FakeEmbedder:
    def embed_one(self, text):
        return [0.0] * 4


class FakeLLM:
    model = "claude-haiku-5-5"

    def __init__(self):
        self.prompts = []

    def complete(self, system, prompt, step, max_tokens=2048):
        self.prompts.append(prompt)
        return "O prazo é de até 7 dias úteis [1].", CallRecord(step, self.model, 1000, 50, 0.4)


def make_pipeline():
    return V0Pipeline(store=FakeStore(), embedder=FakeEmbedder(), llm=FakeLLM(), top_k=2)


def test_cited_indexes_ignores_out_of_range():
    assert cited_indexes("a [1] b [3][2] c [9]", 3) == [1, 2, 3]


def test_v0_puts_numbered_sources_in_the_prompt_and_tracks_cost():
    p = make_pipeline()
    a = p.answer("Qual o prazo para consulta básica?")
    prompt = p.llm.prompts[0]
    assert "[1] RN nº 566 — Art. 3" in prompt and "[2] RN nº 566 — Art. 4" in prompt
    assert a.cited == [1]
    assert a.usage.cost_usd == cost_usd("claude-haiku-5-5", 1000, 50) > 0


def test_ask_endpoint(monkeypatch):
    monkeypatch.setattr(api, "get_pipeline", lambda version="v1": make_pipeline())
    client = TestClient(api.app)
    r = client.post("/ask", json={"question": "Qual o prazo para consulta básica?"})
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "v0"
    assert [s["cited"] for s in body["sources"]] == [True, False]
    assert body["usage"]["llm_calls"] == 1


def test_ask_rejects_empty_question(monkeypatch):
    monkeypatch.setattr(api, "get_pipeline", lambda version="v1": make_pipeline())
    assert TestClient(api.app).post("/ask", json={"question": ""}).status_code == 422
