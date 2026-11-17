"""Paired comparison of two eval runs on the same questions.

    uv run python -m rag.eval.compare v1 v2 [--metric correct|declined]

Both runs answer the same 180 questions, so the honest comparison is per question: how many went
from wrong to right (``gained``) and from right to wrong (``lost``). McNemar's exact test asks
whether ``gained`` vs ``lost`` is more lopsided than a coin flip would make it; with a few dozen
discordant questions, differences of 2–3 points are usually noise. Generation on a local GPU is
not bit-for-bit reproducible even at temperature 0, so re-running the *same* version is the
baseline every difference should be read against.
"""

from __future__ import annotations

import argparse
import json
from math import comb

from rag.eval.run import RUNS_DIR

TYPES = {"correct": ("direct", "multi_hop"), "declined": ("unanswerable", "out_of_scope")}


def mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of ``gained`` vs ``lost`` with p = 0.5."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def outcome(row: dict, metric: str) -> bool:
    return row["verdict"] == "correct" if metric == "correct" else bool(row["refused"])


def compare(a: dict[str, dict], b: dict[str, dict], metric: str, types: tuple[str, ...]) -> dict:
    ids = [i for i in a if i in b and a[i]["type"] in types and a[i].get("verdict") and b[i].get("verdict")]
    ok_a = {i: outcome(a[i], metric) for i in ids}
    ok_b = {i: outcome(b[i], metric) for i in ids}
    gained = sorted(i for i in ids if ok_b[i] and not ok_a[i])
    lost = sorted(i for i in ids if ok_a[i] and not ok_b[i])
    return {"n": len(ids), "a": sum(ok_a.values()), "b": sum(ok_b.values()), "gained": gained, "lost": lost,
            "p_value": round(mcnemar_exact(len(gained), len(lost)), 4)}


def load(name: str) -> dict[str, dict]:
    rows = (json.loads(line) for line in (RUNS_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line)
    return {r["id"]: r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    args = parser.parse_args()
    a, b = load(args.run_a), load(args.run_b)
    print(f"{args.run_a} -> {args.run_b}")
    groups = [("correct", ("direct",)), ("correct", ("multi_hop",)), ("correct", TYPES["correct"]),
              ("declined", TYPES["declined"])]
    for metric, types in groups:
        r = compare(a, b, metric, types)
        print(f"  {metric:8s} {'+'.join(types):28s} {r['a']:3d} -> {r['b']:3d} of {r['n']:3d} | "
              f"gained {len(r['gained']):2d}, lost {len(r['lost']):2d} | McNemar p = {r['p_value']}")


if __name__ == "__main__":
    main()
