"""Offline check of the signals a scope guardrail could use (V3), on stored runs. No model call.

    uv run python -m rag.eval.scope_gate_eval [--run v1_repro] [--checks-run v2_repro_drafts]

Two candidate rules for declining before the model answers:

- **Best reranker score below a threshold.** The score of the best passage is already computed by
  V1's retrieval. For each threshold: how many questions of each type the gate would decline, and
  what the run's totals would become.
- **V2's context check** ("the passages don't cover this"), alone and combined with the score,
  applied to the V2 run graded before the answer check.

It also cross-validates the way the threshold is chosen (half the lowest best score among the
answerable questions): picked on four folds, applied to the fifth. Results go to
``eval/runs/scope_gate.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable

from rag.eval.compare import load
from rag.eval.run import RUNS_DIR

ANSWERABLE = ("direct", "multi_hop")
THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)


def top_score(row: dict) -> float:
    """Best score among the passages of the first retrieval (V2 appends follow-up passages after them)."""
    return max((s["score"] for s in row["sources"][:6]), default=0.0)


def apply_rule(rows: dict[str, dict], declines: Callable[[str], bool]) -> dict:
    """What a run's totals become if every question the rule picks is declined before answering."""
    gated = [i for i in rows if declines(i)]
    answerable = [i for i in gated if rows[i]["type"] in ANSWERABLE]
    return {
        "gated": len(gated),
        "gated_out_of_scope": sum(rows[i]["type"] == "out_of_scope" for i in gated),
        "gated_unanswerable": sum(rows[i]["type"] == "unanswerable" for i in gated),
        "gated_answerable": len(answerable),
        "gated_answerable_correct": sum(rows[i]["verdict"] == "correct" for i in answerable),
        "declined": sum(1 for i, r in rows.items() if r["type"] not in ANSWERABLE and (i in gated or r["refused"])),
        "correct": sum(1 for i, r in rows.items()
                       if r["type"] in ANSWERABLE and i not in gated and r["verdict"] == "correct"),
    }


def half_lowest_answerable(rows: dict[str, dict], ids: list[str]) -> float:
    return min(top_score(rows[i]) for i in ids if rows[i]["type"] in ANSWERABLE) / 2


def cross_validate(rows: dict[str, dict], folds: int = 5, seed: int = 0) -> list[dict]:
    ids = sorted(rows)
    random.Random(seed).shuffle(ids)
    out = []
    for k in range(folds):
        test = ids[k::folds]
        held = set(test)
        tau = half_lowest_answerable(rows, [i for i in ids if i not in held])
        below = [i for i in test if top_score(rows[i]) < tau]
        out.append({"fold": k, "threshold": round(tau, 4),
                    "answerable_declined": sum(rows[i]["type"] in ANSWERABLE for i in below),
                    "to_decline_declined": sum(rows[i]["type"] not in ANSWERABLE for i in below),
                    "to_decline": sum(rows[i]["type"] not in ANSWERABLE for i in test)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="v1_repro", help="run whose retrieval scores and verdicts are used")
    parser.add_argument("--checks-run", default="v2_repro_drafts", help="run with V2's context check in the trace")
    args = parser.parse_args()

    rows = load(args.run)
    base = apply_rule(rows, lambda i: False)
    print(f"{args.run}: declines {base['declined']}/60, correct {base['correct']}/120 without a gate\n")
    print("gate: decline when the best reranker score is below the threshold")
    print("threshold | out-of-scope  unanswerable  answerable (correct) | declines  correct")
    curve = []
    for t in THRESHOLDS:
        r = apply_rule(rows, lambda i, t=t: top_score(rows[i]) < t)
        curve.append({"threshold": t} | r)
        print(f"   {t:5.2f}  | {r['gated_out_of_scope']:8d}/20 {r['gated_unanswerable']:10d}/40 "
              f"{r['gated_answerable']:8d} ({r['gated_answerable_correct']}) {r['declined']:13d}/60 {r['correct']:5d}/120")

    cv = cross_validate(rows)
    print("\nthreshold = half the lowest best score of the answerable questions, picked on 4 folds, applied to the 5th")
    for f in cv:
        print(f"   fold {f['fold']}: threshold {f['threshold']:.3f} | answerable declined {f['answerable_declined']}, "
              f"to-decline declined {f['to_decline_declined']} of {f['to_decline']}")

    checks = load(args.checks_run)
    insufficient = {i for i, r in checks.items() if (c := r["trace"]["context_check"]) and not c["sufficient"]}
    rules = {
        "no rule": lambda i: False,
        "context check says insufficient": lambda i: i in insufficient,
        "insufficient and best score < 0.5": lambda i: i in insufficient and top_score(checks[i]) < 0.5,
        "insufficient and best score < 0.05": lambda i: i in insufficient and top_score(checks[i]) < 0.05,
    }
    print(f"\nrules on {args.checks_run} (V2 before the answer check)")
    by_rule = {}
    for name, rule in rules.items():
        r = by_rule[name] = apply_rule(checks, rule)
        print(f"   {name:36s} gates {r['gated'] - r['gated_answerable']:2d} to decline, "
              f"{r['gated_answerable']:2d} answerable ({r['gated_answerable_correct']} correct) "
              f"-> declines {r['declined']}/60, correct {r['correct']}/120")

    out = {"run": args.run, "checks_run": args.checks_run, "curve": curve, "cross_validation": cv, "context_check": by_rule}
    (RUNS_DIR / "scope_gate.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
