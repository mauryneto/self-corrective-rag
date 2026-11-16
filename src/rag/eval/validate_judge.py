"""How well does the judge agree with a person?

    uv run python -m rag.eval.validate_judge [--run v0_baseline]

Reads the human labels in ``eval/judge_labels.json`` ({question_id: "correct"|"partial"|"incorrect"})
and the judge's verdicts for the same questions in ``eval/runs/<run>.jsonl``, then reports:
- raw agreement and Cohen's kappa on the three classes;
- the same on a binary view (correct vs. not correct), which is what the headline metric uses;
- the confusion matrix, and every disagreement with the judge's reasoning, for inspection.

Reading kappa: < 0.4 weak, 0.4-0.6 moderate, 0.6-0.8 substantial, > 0.8 almost perfect.
The sample is stratified by the judge's own verdicts (so partial and incorrect answers are well
represented); agreement is measured on that sample, not re-weighted to the full run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from rag.config import ROOT
from rag.eval.run import RUNS_DIR

LABELS_PATH = ROOT / "eval" / "judge_labels.json"
CLASSES = ["correct", "partial", "incorrect"]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over the same items."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def confusion(human: list[str], judge: list[str]) -> dict[str, dict[str, int]]:
    m = {h: {j: 0 for j in CLASSES} for h in CLASSES}
    for h, j in zip(human, judge, strict=True):
        m[h][j] += 1
    return m


def agreement_report(pairs: list[tuple[str, str]]) -> dict:
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    to_bin = lambda xs: ["correct" if x == "correct" else "not_correct" for x in xs]  # noqa: E731
    hb, jb = to_bin(human), to_bin(judge)
    return {
        "n": len(pairs),
        "agreement": round(sum(h == j for h, j in pairs) / len(pairs), 4),
        "kappa": round(cohen_kappa(human, judge), 4),
        "binary_agreement": round(sum(h == j for h, j in zip(hb, jb, strict=True)) / len(pairs), 4),
        "binary_kappa": round(cohen_kappa(hb, jb), 4),
        "confusion_human_rows_judge_cols": confusion(human, judge),
        "human_counts": dict(Counter(human)),
        "judge_counts": dict(Counter(judge)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="v0_baseline")
    args = parser.parse_args()

    labels: dict[str, str] = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    results = {}
    for line in (RUNS_DIR / f"{args.run}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            results[r["id"]] = r
    ids = [i for i in labels if i in results and results[i].get("verdict")]
    pairs = [(labels[i], results[i]["verdict"]) for i in ids]
    report = agreement_report(pairs) | {"run": args.run}
    out = ROOT / "eval" / "judge_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "confusion_human_rows_judge_cols"}, indent=1))
    print("\nconfusion (rows = human, cols = judge):")
    print(f"{'':>10} " + " ".join(f"{c:>10}" for c in CLASSES))
    for h in CLASSES:
        print(f"{h:>10} " + " ".join(f"{report['confusion_human_rows_judge_cols'][h][j]:>10}" for j in CLASSES))
    print("\ndisagreements:")
    for i in ids:
        if labels[i] != results[i]["verdict"]:
            print(f"- {i} ({results[i]['type']}): human={labels[i]} judge={results[i]['verdict']} | "
                  f"{results[i].get('judge_reasoning', '')[:200]}")


if __name__ == "__main__":
    main()
