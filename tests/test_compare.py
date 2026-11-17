from rag.eval.compare import compare, mcnemar_exact


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert abs(mcnemar_exact(0, 6) - 2 / 64) < 1e-12  # 2 * 0.5^6
    assert mcnemar_exact(10, 0) < 0.01


def test_compare_counts_discordant_questions():
    def row(t, v):
        return {"type": t, "verdict": v, "refused": v == "incorrect"}

    a = {"q1": row("direct", "correct"), "q2": row("direct", "incorrect"), "q3": row("multi_hop", "correct"),
         "q4": row("unanswerable", "correct")}
    b = {"q1": row("direct", "incorrect"), "q2": row("direct", "correct"), "q3": row("multi_hop", "correct"),
         "q4": row("unanswerable", "correct")}
    r = compare(a, b, "correct", ("direct", "multi_hop"))
    assert (r["n"], r["a"], r["b"], r["gained"], r["lost"]) == (3, 2, 2, ["q2"], ["q1"])
