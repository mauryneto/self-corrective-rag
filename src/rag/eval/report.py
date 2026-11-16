"""Print the README results table from the run summaries in eval/runs/.

    uv run python -m rag.eval.report [run_name ...]
"""

from __future__ import annotations

import json
import sys

from rag.eval.run import RUNS_DIR

COLUMNS = [
    ("Version", lambda s: s["version"].upper()),
    ("Correct", lambda s: _pct(s["answer_correct"])),
    ("Correct + partial", lambda s: _pct(s["answer_correct_or_partial"])),
    ("Multi-hop correct", lambda s: _pct(s["multi_hop_correct"])),
    ("Recall@5", lambda s: _pct(s["recall_at_5"])),
    ("MRR", lambda s: f"{s['mrr']:.2f}"),
    ("Declines when it should", lambda s: _pct(s["refusal_recall"])),
    ("Refusal precision", lambda s: _pct(s["refusal_precision"])),
    ("Citations supported", lambda s: _pct(s["citations_supported"])),
    ("p50 / p95 latency", lambda s: f"{s['latency_p50_s']:.0f}s / {s['latency_p95_s']:.0f}s"),
    ("Cost / question", lambda s: f"US$ {s['cost_per_question_usd']:.4f}"),
]


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def table(summaries: list[dict]) -> str:
    head = "| " + " | ".join(c for c, _ in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    rows = ["| " + " | ".join(f(s) for _, f in COLUMNS) + " |" for s in summaries]
    return "\n".join([head, sep, *rows])


def main() -> None:
    names = sys.argv[1:]
    paths = [RUNS_DIR / f"{n}.summary.json" for n in names] if names else sorted(RUNS_DIR.glob("*.summary.json"))
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    print(table(summaries))


if __name__ == "__main__":
    main()
