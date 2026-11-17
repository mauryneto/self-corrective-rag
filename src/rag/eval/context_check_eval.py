"""Retrieval-only eval of V2's context check (step 2) on the 120 answerable questions.

    uv run python -m rag.eval.context_check_eval [--from-run v1] [--per-query 2,3]

Replays V1's retrieval from a stored run (same chunks, no re-ranking), asks the local model whether
the passages cover the question and, if not, which searches to run, then measures how much of the
gold the final context holds. The model's checks are cached in ``eval/runs/v2_context_checks.jsonl``,
so trying other expansion settings afterwards only costs the follow-up searches.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from rag.costs import Usage
from rag.eval.dataset import load_questions
from rag.eval.metrics import retrieval_scores
from rag.eval.run import RUNS_DIR
from rag.pipeline.v2 import ContextCheck, V2Pipeline

CACHE = RUNS_DIR / "v2_context_checks.jsonl"


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, CACHE.read_text(encoding="utf-8").splitlines()) if r}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-run", default="v1")
    parser.add_argument("--per-query", default="2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sub-rerank-top", type=int, default=10)
    args = parser.parse_args()

    stored = {r["id"]: r for r in map(json.loads, (RUNS_DIR / f"{args.from_run}.jsonl").read_text(
        encoding="utf-8").splitlines()) if r}
    questions = [q for q in load_questions() if q.answerable][: args.limit]
    pipe = V2Pipeline(sub_rerank_top=args.sub_rerank_top)
    cache = load_cache()

    # 1. the context checks (one LLM call each), cached
    t0 = time.perf_counter()
    with CACHE.open("a", encoding="utf-8") as fh:
        for i, q in enumerate(questions, 1):
            if q.id in cache:
                continue
            hits = pipe.store.get_hits([s["chunk_id"] for s in stored[q.id]["sources"]])
            t = time.perf_counter()
            check = pipe.check_context(q.question, hits, Usage())
            rec = {"id": q.id, "check": check.model_dump() if check else None,
                   "latency_s": round(time.perf_counter() - t, 2)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            cache[q.id] = rec
            print(f"[{i}/{len(questions)}] {q.id} {q.type:9s} sufficient={check.sufficient if check else None} "
                  f"queries={check.queries if check else None} ({rec['latency_s']}s)", flush=True)
    print(f"checks done in {(time.perf_counter() - t0) / 60:.1f} min")

    # 2. expansion settings
    results = {}
    for per_query in map(int, args.per_query.split(",")):
        pipe.per_query = per_query
        pipe.sub_retriever.top_k = per_query + 2
        per_type: dict[str, list] = {"direct": [], "multi_hop": []}
        t0 = time.perf_counter()
        for q in questions:
            hits = pipe.store.get_hits([s["chunk_id"] for s in stored[q.id]["sources"]])
            check = ContextCheck(**cache[q.id]["check"]) if cache[q.id]["check"] else None
            first = retrieval_scores([g.model_dump() for g in q.gold], [{"doc_id": h.doc_id, "unit": h.unit}
                                                                        for h in hits], k=len(hits))[0]
            if check and not check.sufficient and check.queries:
                hits = pipe.expand(hits, check.queries)
            src = [{"doc_id": h.doc_id, "unit": h.unit} for h in hits]
            per_type[q.type].append({"before": first, "after": retrieval_scores(
                [g.model_dump() for g in q.gold], src, k=len(src))[0], "n": len(hits),
                "triggered": bool(check and not check.sufficient and check.queries)})
        out = {"seconds_per_question": round((time.perf_counter() - t0) / len(questions), 2)}
        for t, rows in [("all", per_type["direct"] + per_type["multi_hop"]), *per_type.items()]:
            out[t] = {"recall_before": round(statistics.mean(r["before"] for r in rows), 4),
                      "recall_after": round(statistics.mean(r["after"] for r in rows), 4),
                      "complete_before": sum(r["before"] == 1.0 for r in rows),
                      "complete_after": sum(r["after"] == 1.0 for r in rows),
                      "triggered": sum(r["triggered"] for r in rows),
                      "mean_passages": round(statistics.mean(r["n"] for r in rows), 2), "n": len(rows)}
        results[f"per_query={per_query},sub_rerank_top={args.sub_rerank_top}"] = out
        print(f"per_query={per_query}: " + json.dumps(out, ensure_ascii=False), flush=True)
    out_path = RUNS_DIR / "v2_context_check_eval.json"
    previous = json.loads(out_path.read_text(encoding="utf-8-sig")) if out_path.exists() else {}
    out_path.write_text(json.dumps(previous | results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
