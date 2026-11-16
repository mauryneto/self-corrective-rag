import pytest
from pydantic import ValidationError

from rag.eval.dataset import GoldSource, Question, load_questions


def test_the_committed_question_set_is_valid():
    questions = load_questions()
    assert len(questions) >= 10
    assert {q.type for q in questions} == {"direct", "multi_hop", "unanswerable", "out_of_scope"}


def test_gold_source_matches_split_articles():
    gold = GoldSource(doc="rn-566", unit="Art. 3")
    assert gold.matches("rn-566", "Art. 3")
    assert gold.matches("rn-566", "Art. 3 (parte 2)")
    assert not gold.matches("rn-566", "Art. 30")
    assert not gold.matches("rn-623", "Art. 3")


def test_answerable_questions_need_gold_and_unanswerable_must_not_have_it():
    with pytest.raises(ValidationError):
        Question(id="x", type="direct", question="q", reference_answer="r")
    with pytest.raises(ValidationError):
        Question(id="x", type="unanswerable", question="q", reference_answer="r",
                 gold=[GoldSource(doc="rn-1", unit="Art. 1")])


def test_multi_hop_needs_two_different_articles():
    with pytest.raises(ValidationError):
        Question(id="x", type="multi_hop", question="q", reference_answer="r",
                 gold=[GoldSource(doc="rn-1", unit="Art. 1")])
    ok = Question(id="x", type="multi_hop", question="q", reference_answer="r",
                  gold=[GoldSource(doc="rn-1", unit="Art. 1"), GoldSource(doc="rn-1", unit="Art. 2")])
    assert ok.answerable
