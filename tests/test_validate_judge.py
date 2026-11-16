import pytest

from rag.eval.validate_judge import agreement_report, cohen_kappa


def test_kappa_perfect_and_chance():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # Agreement no better than chance -> kappa 0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_kappa_known_value():
    # Classic 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_rejects_mismatched_lists():
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


def test_report_has_binary_view_and_confusion():
    pairs = [("correct", "correct"), ("partial", "incorrect"), ("incorrect", "incorrect"), ("partial", "partial")]
    r = agreement_report(pairs)
    assert r["agreement"] == 0.75
    assert r["binary_agreement"] == 1.0  # partial vs incorrect are both "not correct"
    assert r["confusion_human_rows_judge_cols"]["partial"]["incorrect"] == 1
