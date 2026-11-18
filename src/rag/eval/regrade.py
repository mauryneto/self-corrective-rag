"""Re-grade the stored answers of a run with the current judge, without re-generating them.

    uv run python -m rag.eval.regrade --run v0_baseline [--ids-from eval/judge_labels.json] [--out NAME]

Used after changing the judge rubric: the answers stay identical, only the verdicts change, so
the effect of the rubric can be measured on its own. With ``--drafts`` (V2+), it grades the drafts
the answer check rewrote instead, which measures that step on its own. Writes ``eval/runs/<out>.jsonl`` and its
summary (the default name is ``<run>_rubric-<version>``).

With ``--missing`` it grades only the answers the judge failed on during the run and updates the
run in place, so a judge hiccup does not cost a re-generation.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.eval.dataset import load_questions
from rag.eval.judge import DEFAULT_JUDGE_MODEL, RUBRIC_VERSION, JudgeCache, get_judge, grade
from rag.eval.metrics import summarize
from rag.eval.run import JUDGE_CACHE, RUNS_DIR
from rag.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--ids-from", type=Path, default=None, help="JSON object or list whose keys/items are question ids")
    parser.add_argument("--out", default=None)
    parser.add_argument("--judge-backend", default="claude-code", choices=["claude-code", "anthropic"])
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drafts", action="store_true",
                        help="V2+: grade the first draft of every answer the answer check rewrote (the rest are kept "
                             "as graded), i.e. the same run without the answer-check step")
    parser.add_argument("--missing", action="store_true",
                        help="grade only the answers left without a verdict (judge failures) and update the run in place")
    args = parser.parse_args()
    if args.missing and (args.drafts or args.out):
        parser.error("--missing updates the run in place; it cannot be combined with --drafts or --out")

    rows = [json.loads(line) for line in (RUNS_DIR / f"{args.run}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    kept: list[dict] = []
    if args.missing:
        kept = [r for r in rows if r.get("verdict")]
        rows = [r for r in rows if not r.get("verdict")]
    if args.drafts:
        kept = [r for r in rows if "draft" not in r.get("trace", {})]
        rows = [r | {"answer": r["trace"]["draft"], "cited": r["trace"]["draft_cited"]}
                for r in rows if "draft" in r.get("trace", {})]
    if args.ids_from:
        wanted = set(json.loads(args.ids_from.read_text(encoding="utf-8")))
        rows = [r for r in rows if r["id"] in wanted]
    questions = {q.id: q for q in load_questions()}
    store = Store()
    judge = get_judge(args.judge_backend, args.judge_model)
    cache = JudgeCache(JUDGE_CACHE)

    # Read every cited chunk once, up front: psycopg connections are not thread-safe.
    ids = sorted({s["chunk_id"] for r in rows for s in r["sources"]})
    texts = dict(store.conn.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (ids,)).fetchall())

    def regrade(r: dict) -> dict:
        q = questions[r["id"]]
        by_index = {s["index"]: s["chunk_id"] for s in r["sources"]}
        cited = {n: texts.get(by_index[n], "(trecho não encontrado no índice atual)") for n in r["cited"] if n in by_index}
        j = grade(judge, q.question, q.reference_answer, q.answerable, r["answer"], cited, cache=cache)
        r = {k: v for k, v in r.items() if k != "judge_error"}
        return r | {"verdict": j.verdict.verdict, "refused": j.verdict.refused,
                    "citations_supported": j.verdict.citations_supported, "judge_reasoning": j.verdict.reasoning,
                    "judge_rubric": RUBRIC_VERSION, "judge_cached": j.cached}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        out_rows = list(pool.map(regrade, rows))
    if kept:
        order = {q: i for i, q in enumerate(questions)}
        out_rows = sorted(out_rows + kept, key=lambda r: order[r["id"]])

    if args.missing:
        name = args.run
    else:
        name = args.out or (f"{args.run}_drafts" if args.drafts else f"{args.run}_rubric-{RUBRIC_VERSION}")
    (RUNS_DIR / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
                                            encoding="utf-8")
    source_summary = RUNS_DIR / f"{args.run}.summary.json"
    inherited = json.loads(source_summary.read_text(encoding="utf-8")) if source_summary.exists() else {}
    if args.missing:  # same run, now complete: keep its summary fields and recompute the metrics
        summary = inherited | summarize(out_rows) | {"judge_failures": 0}
    else:
        keep = ("version", "llm_provider", "llm_model", "embed_model", "top_k")
        summary = {k: inherited[k] for k in keep if k in inherited} | summarize(out_rows) | {
            "run": name, "regraded_from": args.run, "judge_rubric": RUBRIC_VERSION,
            "judge": f"{args.judge_backend}:{args.judge_model}"}
    (RUNS_DIR / f"{name}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"re-graded {len(out_rows)} answers -> eval/runs/{name}.jsonl")


if __name__ == "__main__":
    main()
