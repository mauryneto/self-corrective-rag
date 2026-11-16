from rag.eval.check_questions import check, normalize
from rag.eval.dataset import GoldSource, Question

DOC = {
    "kind": "RN", "number": 566, "doc_id": "rn-566", "dou_date": "2023-01-02", "updated": "2026-03-02",
    "ementa": "Dispõe sobre a garantia de atendimento.", "source_url": "u",
    "paragraphs": [
        "A Diretoria Colegiada adota a seguinte Resolução Normativa.",
        "Art. 3º A operadora deverá garantir o atendimento nos seguintes prazos:",
        "V – consulta/sessão com psicólogo: em até dez dias úteis;",
        "Art. 4º Na hipótese de indisponibilidade de prestador, a operadora deverá garantir o transporte.",
    ],
}


def q(**kw):
    base = {"id": "x1", "type": "direct", "question": "Prazo do psicólogo?", "reference_answer": "10 dias úteis."}
    return Question(**(base | kw))


def test_normalize_ignores_case_accents_quotes_and_ordinals():
    assert normalize("Art. 3º  “Consulta” — PSICÓLOGO") == normalize('art. 3o "consulta" - psicologo')


def test_verbatim_evidence_passes():
    ok = q(gold=[GoldSource(doc="rn-566", unit="Art. 3", evidence="consulta/sessão com psicólogo: em até dez dias úteis")])
    assert check([ok], [DOC]) == ([], [])


def test_paraphrased_evidence_fails():
    bad = q(gold=[GoldSource(doc="rn-566", unit="Art. 3", evidence="psicólogo em no máximo dez dias úteis")])
    errors, _ = check([bad], [DOC])
    assert errors and "not found word for word" in errors[0]


def test_evidence_from_another_article_fails():
    wrong_unit = q(gold=[GoldSource(doc="rn-566", unit="Art. 4", evidence="consulta/sessão com psicólogo")])
    assert check([wrong_unit], [DOC])[0]


def test_missing_norm_or_unit_fails():
    assert "not in the corpus" in check([q(gold=[GoldSource(doc="rn-1", unit="Art. 1", evidence="x")])], [DOC])[0][0]
    assert "has no unit" in check([q(gold=[GoldSource(doc="rn-566", unit="Art. 9", evidence="x")])], [DOC])[0][0]


def test_duplicate_questions_fail():
    gold = [GoldSource(doc="rn-566", unit="Art. 3", evidence="em até dez dias úteis")]
    errors, _ = check([q(id="a", gold=gold), q(id="b", question="prazo do PSICÓLOGO?", gold=gold)], [DOC])
    assert any("same question" in e for e in errors)


def test_unanswerable_terms_found_in_corpus_warn():
    un = Question(id="u1", type="unanswerable", question="Q?", reference_answer="Não há.",
                  absence_terms=["transporte", "teletransporte quântico"])
    errors, warnings = check([un], [DOC])
    assert errors == [] and len(warnings) == 1 and "'transporte'" in warnings[0]
