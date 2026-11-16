"""Turn per-question results into the numbers reported in the README."""

from __future__ import annotations

import statistics


def retrieval_scores(gold: list[dict], sources: list[dict], k: int = 5) -> tuple[float, float]:
    """(recall@k, reciprocal rank) for one question.

    recall@k: share of the gold (norm, article) pairs that appear among the first k sources.
    reciprocal rank: 1 / position of the first source that matches any gold pair (0 if none).
    """

    def hit(g: dict, s: dict) -> bool:
        return s["doc_id"] == g["doc"] and (s["unit"] == g["unit"] or s["unit"].startswith(g["unit"] + " (parte"))

    pairs = {(g["doc"], g["unit"]): g for g in gold}.values()
    found = sum(any(hit(g, s) for s in sources[:k]) for g in pairs)
    rr = next((1 / i for i, s in enumerate(sources, 1) if any(hit(g, s) for g in pairs)), 0.0)
    return found / len(pairs), rr


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def summarize(results: list[dict]) -> dict:
    """Aggregate metrics. Each result has: type, verdict, refused, citations_supported, recall_at_5,
    rr, latency_s, cost_usd."""
    answerable = [r for r in results if r["type"] in ("direct", "multi_hop")]
    declinable = [r for r in results if r["type"] in ("unanswerable", "out_of_scope")]

    def rate(rows, pred):
        return round(sum(1 for r in rows if pred(r)) / len(rows), 4) if rows else None

    refused_all = [r for r in results if r["refused"]]
    correct_refusals = [r for r in refused_all if r["type"] in ("unanswerable", "out_of_scope")]
    with_citations = [r for r in answerable if r["citations_supported"] is not None]

    out = {
        "n": len(results),
        "answer_correct": rate(answerable, lambda r: r["verdict"] == "correct"),
        "answer_correct_or_partial": rate(answerable, lambda r: r["verdict"] in ("correct", "partial")),
        "direct_correct": rate([r for r in answerable if r["type"] == "direct"], lambda r: r["verdict"] == "correct"),
        "multi_hop_correct": rate([r for r in answerable if r["type"] == "multi_hop"],
                                  lambda r: r["verdict"] == "correct"),
        "recall_at_5": round(statistics.mean(r["recall_at_5"] for r in answerable), 4) if answerable else None,
        "mrr": round(statistics.mean(r["rr"] for r in answerable), 4) if answerable else None,
        "refusal_recall": rate(declinable, lambda r: r["refused"]),  # declined what it should
        "refusal_precision": round(len(correct_refusals) / len(refused_all), 4) if refused_all else None,
        "wrongful_refusal": rate(answerable, lambda r: r["refused"]),  # declined what it could answer
        "citations_supported": rate(with_citations, lambda r: r["citations_supported"]),
        "latency_p50_s": round(percentile([r["latency_s"] for r in results], 50), 2),
        "latency_p95_s": round(percentile([r["latency_s"] for r in results], 95), 2),
        "cost_per_question_usd": round(statistics.mean(r["cost_usd"] for r in results), 6) if results else None,
    }
    return out
