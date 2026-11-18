"""Floor and ceiling reference pipelines with in-memory fakes: no database, no Ollama."""

import pytest

from rag.costs import CallRecord
from rag.eval.dataset import GoldSource, Question
from rag.eval.reference import CLOSED_BOOK_SYSTEM, CeilingPipeline, FloorPipeline, gold_hits
from rag.store import Hit


def hit(n: int, unit: str, body: str, doc: str = "rn-566") -> Hit:
    return Hit(f"{doc}#{n}", doc, unit, f"{doc} — {unit}\n{body}", "u", 0.0)


CHUNKS = [
    hit(1, "Art. 3", "O prazo é de sete dias úteis."),
    hit(2, "Art. 4 (parte 1)", "A operadora deve garantir o atendimento."),
    hit(3, "Art. 4 (parte 2)", "O descumprimento sujeita a operadora a multa."),
    hit(4, "Art. 5", "Disposições finais."),
    hit(5, "Art. 6", "Outro assunto."),
    hit(6, "Art. 7", "Mais um assunto."),
    hit(7, "Art. 8", "Assunto sete."),
    hit(8, "Art. 9", "Assunto oito."),
]


class FakeStore:
    def unit_hits(self, doc_id, unit):
        return [h for h in CHUNKS if h.doc_id == doc_id and (h.unit == unit or h.unit.startswith(unit + " (parte "))]

    def get_hits(self, chunk_ids):
        by_id = {h.chunk_id: h for h in CHUNKS}
        return [by_id[c] for c in chunk_ids if c in by_id]


class FakeLLM:
    model = "local"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, prompt, step, max_tokens=2048):
        self.calls.append((system, prompt))
        return "Resposta [1].", CallRecord(step, self.model, 100, 10, 0.1)


MULTI = Question(id="q1", type="multi_hop", question="Qual o prazo e a sanção?", reference_answer="Sete dias; multa.",
                 gold=[GoldSource(doc="rn-566", unit="Art. 4", evidence="sujeita a operadora a  multa"),
                       GoldSource(doc="rn-566", unit="Art. 3", evidence="sete dias úteis")])
NO_GOLD = Question(id="q2", type="unanswerable", question="Qual a cor do boleto?", reference_answer="Não consta.")
V1_RETRIEVAL = {"q1": ["rn-566#5", "rn-566#1", "rn-566#2", "rn-566#6", "rn-566#7", "rn-566#8"],
                "q2": ["rn-566#4", "rn-566#5", "rn-566#6"]}


def ceiling(replay, top_k=4) -> tuple[CeilingPipeline, FakeLLM]:
    llm = FakeLLM()
    return CeilingPipeline([MULTI, NO_GOLD], replay, store=FakeStore(), embedder=object(), llm=llm, top_k=top_k), llm


def test_floor_answers_without_passages_or_citations():
    llm = FakeLLM()
    a = FloorPipeline(llm=llm).answer("Qual o prazo?")
    assert a.sources == [] and a.cited == [] and a.version == "floor"
    assert llm.calls == [(CLOSED_BOOK_SYSTEM, "Pergunta: Qual o prazo?")]


def test_gold_hits_keeps_only_the_part_of_the_article_that_holds_the_evidence():
    # Art. 4 has two parts; the quote (spacing aside) is in part 2 only
    assert [h.chunk_id for h in gold_hits(FakeStore(), MULTI)] == ["rn-566#3", "rn-566#1"]


def test_gold_hits_falls_back_to_every_part_when_the_quote_is_in_none():
    q = MULTI.model_copy(update={"gold": [GoldSource(doc="rn-566", unit="Art. 4", evidence="texto que não existe"),
                                          GoldSource(doc="rn-566", unit="Art. 3", evidence="sete dias úteis")]})
    assert [h.chunk_id for h in gold_hits(FakeStore(), q)] == ["rn-566#2", "rn-566#3", "rn-566#1"]


def test_ceiling_puts_gold_first_and_fills_to_top_k_with_the_replayed_retrieval():
    p, llm = ceiling(V1_RETRIEVAL)
    a = p.answer(MULTI.question)
    # gold first; then V1's passages in their order, skipping the one that is already there
    assert [h.chunk_id for h in a.sources] == ["rn-566#3", "rn-566#1", "rn-566#5", "rn-566#2"]
    assert a.version == "ceiling" and a.cited == [1]
    assert "[1] rn-566 — Art. 4 (parte 2)" in llm.calls[0][1]


def test_ceiling_replays_the_stored_retrieval_unchanged_when_there_is_no_gold():
    p, _ = ceiling(V1_RETRIEVAL)
    assert [h.chunk_id for h in p.answer(NO_GOLD.question).sources] == V1_RETRIEVAL["q2"]


def test_ceiling_gold_gives_only_the_gold_passages():
    p, _ = ceiling(None)
    a = p.answer(MULTI.question)
    assert [h.chunk_id for h in a.sources] == ["rn-566#3", "rn-566#1"] and a.version == "ceiling_gold"


def test_ceiling_refuses_questions_outside_the_eval_set():
    p, _ = ceiling(V1_RETRIEVAL)
    with pytest.raises(KeyError):
        p.answer("uma pergunta qualquer")


def test_the_answering_prompt_can_be_swapped_for_an_experiment():
    from rag.pipeline.prompts import ANSWER_SYSTEM, ANSWER_SYSTEM_COMPLETE, COMPLETE_RULE

    # the experiment is the official prompt plus one rule, placed before the last one
    assert ANSWER_SYSTEM_COMPLETE.replace(COMPLETE_RULE, "") == ANSWER_SYSTEM
    assert ANSWER_SYSTEM_COMPLETE.splitlines()[-1] == ANSWER_SYSTEM.splitlines()[-1]

    p, llm = ceiling(None)
    p.answer(MULTI.question)
    p.answer_system = ANSWER_SYSTEM_COMPLETE
    p.answer(MULTI.question)
    assert [system for system, _ in llm.calls] == [ANSWER_SYSTEM, ANSWER_SYSTEM_COMPLETE]
