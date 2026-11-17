"""V2's self-corrective steps with in-memory fakes: no database, no Ollama."""

from rag.costs import CallRecord
from rag.pipeline.v2 import AnswerCheck, ContextCheck, V2Pipeline
from rag.store import Hit


def hit(i: int, doc: str = "rn-566") -> Hit:
    return Hit(f"{doc}#{i}", doc, f"Art. {i}", f"{doc} — Art. {i}\ntexto {i}", "u", 0.0)


FIRST = [hit(i) for i in range(1, 7)]
EXTRA = [hit(1, "rn-623"), hit(2, "rn-623"), hit(3, "rn-623")]


class FakeStore:
    def vector_search(self, embedding, k):
        return (EXTRA if embedding == [1.0] else FIRST)[:k]

    def lexical_search(self, query, k):
        return []


class FakeEmbedder:
    def embed_one(self, text):
        return [1.0] if text.startswith("busca") else [0.0]


class FakeReranker:
    def score(self, query, texts):
        return [-i for i in range(len(texts))]  # keep the incoming order


class FakeLLM:
    model = "local"

    def __init__(self, context: ContextCheck | Exception, answer_check: AnswerCheck | None = None,
                 answers=("Resposta [1].", "Resposta revisada [7].")):
        self.context, self.answer_check, self.answers = context, answer_check, list(answers)
        self.steps: list[str] = []

    def complete(self, system, prompt, step, max_tokens=2048):
        self.steps.append(step)
        return self.answers.pop(0), CallRecord(step, self.model, 100, 10, 0.1)

    def complete_json(self, system, prompt, schema, step, max_tokens=1024):
        self.steps.append(step)
        result = self.context if schema is ContextCheck else self.answer_check
        if isinstance(result, Exception):
            raise result
        return result, CallRecord(step, self.model, 100, 10, 0.1)


def pipeline(llm: FakeLLM, **kw) -> V2Pipeline:
    return V2Pipeline(store=FakeStore(), embedder=FakeEmbedder(), llm=llm, top_k=6, reranker=FakeReranker(), **kw)


def test_sufficient_context_and_supported_answer_take_the_short_path():
    llm = FakeLLM(ContextCheck(sufficient=True), AnswerCheck(supported=True))
    a = pipeline(llm).answer("pergunta")
    assert llm.steps == ["check_context", "generate", "check_answer"]
    assert len(a.sources) == 6 and a.trace["added_passages"] == 0 and "draft" not in a.trace
    assert a.answer == "Resposta [1]." and a.usage.summary()["llm_calls"] == 3


def test_missing_information_adds_passages_from_follow_up_queries():
    llm = FakeLLM(ContextCheck(sufficient=False, missing="prazo", queries=["busca prazo"]),
                  AnswerCheck(supported=True))
    a = pipeline(llm, per_query=2).answer("pergunta")
    assert [h.chunk_id for h in a.sources[6:]] == ["rn-623#1", "rn-623#2"]
    assert a.trace["added_passages"] == 2


def test_follow_up_query_skips_passages_already_in_context():
    p = pipeline(FakeLLM(ContextCheck(sufficient=True)), per_query=2)
    out = p.expand(FIRST + [EXTRA[0]], ["busca prazo"])
    assert [h.chunk_id for h in out[7:]] == ["rn-623#2", "rn-623#3"]


def test_unsupported_answer_is_rewritten_once_and_the_draft_kept():
    llm = FakeLLM(ContextCheck(sufficient=False, queries=["busca prazo"]),
                  AnswerCheck(supported=False, problems=["o prazo não está no trecho [1]"]))
    a = pipeline(llm).answer("pergunta")
    assert llm.steps == ["check_context", "generate", "check_answer", "revise"]
    assert a.answer == "Resposta revisada [7]." and a.cited == [7]
    assert a.trace["draft"] == "Resposta [1]." and a.trace["draft_cited"] == [1]


def test_a_failed_check_never_costs_the_answer():
    llm = FakeLLM(ValueError("bad json"), ValueError("bad json"))
    a = pipeline(llm).answer("pergunta")
    assert a.answer == "Resposta [1]." and len(a.sources) == 6
    assert a.trace["context_check"] is None and a.trace["answer_check"] is None


def test_a_refusal_is_not_sent_to_the_answer_check():
    llm = FakeLLM(ContextCheck(sufficient=True), AnswerCheck(supported=True),
                  answers=("Os trechos não são suficientes para responder.",))
    pipeline(llm).answer("pergunta")
    assert "check_answer" not in llm.steps
