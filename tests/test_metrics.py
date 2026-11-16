from rag.eval.metrics import percentile, retrieval_scores, summarize


def src(doc, unit):
    return {"doc_id": doc, "unit": unit}


def test_retrieval_recall_and_reciprocal_rank():
    gold = [{"doc": "rn-566", "unit": "Art. 3"}, {"doc": "rn-623", "unit": "Art. 12"}]
    sources = [src("rn-1", "Art. 1"), src("rn-566", "Art. 3 (parte 2)"), src("rn-9", "Art. 9"),
               src("rn-9", "Art. 8"), src("rn-9", "Art. 7"), src("rn-623", "Art. 12")]
    recall, rr = retrieval_scores(gold, sources, k=5)
    assert recall == 0.5  # rn-623 Art. 12 is 6th: outside top-5
    assert rr == 0.5      # first hit at position 2


def test_repeated_gold_pairs_count_once():
    gold = [{"doc": "rn-1", "unit": "Art. 1"}, {"doc": "rn-1", "unit": "Art. 1"}]
    assert retrieval_scores(gold, [src("rn-1", "Art. 1")]) == (1.0, 1.0)


def test_percentile():
    assert percentile([1, 2, 3, 4, 100], 50) == 3
    assert percentile([1, 2, 3, 4, 100], 95) == 100


def row(type_, verdict, refused, cited=None, recall=None, rr=None):
    return {"type": type_, "verdict": verdict, "refused": refused, "citations_supported": cited,
            "recall_at_5": recall, "rr": rr, "latency_s": 1.0, "cost_usd": 0.001}


def test_summary_metrics():
    results = [
        row("direct", "correct", False, True, 1.0, 1.0),
        row("direct", "partial", False, True, 1.0, 0.5),
        row("multi_hop", "incorrect", True, None, 0.5, 0.0),   # wrongful refusal
        row("unanswerable", "correct", True),                  # right refusal
        row("out_of_scope", "incorrect", False),               # answered when it should decline
    ]
    s = summarize(results)
    assert s["answer_correct"] == round(1 / 3, 4)
    assert s["answer_correct_or_partial"] == round(2 / 3, 4)
    assert s["refusal_recall"] == 0.5
    assert s["refusal_precision"] == 0.5
    assert s["wrongful_refusal"] == round(1 / 3, 4)
    assert s["citations_supported"] == 1.0
    assert s["recall_at_5"] == round((1 + 1 + 0.5) / 3, 4)
