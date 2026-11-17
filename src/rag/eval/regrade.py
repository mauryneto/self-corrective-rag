"""Re-grade the stored answers of a run with the current judge, without re-generating them.

    uv run python -m rag.eval.regrade --run v0_baseline [--ids-from eval/judge_labels.json] [--out NAME]

Used after changing the judge rubric: the answers stay identical, only the verdicts change, so
the effect of the rubric can be measured on its own. Writes ``eval/runs/<out>.jsonl`` and its
summary (the default name is ``<run>_rubric-<version>``).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.eval.dataset import load_questions
from rag.eval.judge import DEFAULT_JUDGE_MODEL, RUBRIC_VERSION, get_judge, grade
from rag.eval.metrics import summarize
from rag.eval.run import RUNS_DIR
from rag.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--ids-from", type=Path, default=None, help="JSON object or list whose keys/items are question ids")
    parser.add_argument("--out", default=None)
    parser.add_argument("--judge-backend", default="claude-code", choices=["claude-code", "anthropic"])
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (RUNS_DIR / f"{args.run}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.ids_from:
        wanted = set(json.loads(args.ids_from.read_text(encoding="utf-8")))
        rows = [r for r in rows if r["id"] in wanted]
    questions = {q.id: q for q in load_questions()}
    store = Store()
    judge = get_judge(args.judge_backend, args.judge_model)

    # Read every cited chunk once, up front: psycopg connections are not thread-safe.
    ids = sorted({s["chunk_id"] for r in rows for s in r["sources"]})
    texts = dict(store.conn.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (ids,)).fetchall())

    def regrade(r: dict) -> dict:
        q = questions[r["id"]]
        by_index = {s["index"]: s["chunk_id"] for s in r["sources"]}
        cited = {n: texts.get(by_index[n], "(trecho não encontrado no índice atual)") for n in r["cited"] if n in by_index}
        j = grade(judge, q.question, q.reference_answer, q.answerable, r["answer"], cited)
        return r | {"verdict": j.verdict.verdict, "refused": j.verdict.refused,
                    "citations_supported": j.verdict.citations_supported, "judge_reasoning": j.verdict.reasoning,
                    "judge_rubric": RUBRIC_VERSION}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        out_rows = list(pool.map(regrade, rows))

    name = args.out or f"{args.run}_rubric-{RUBRIC_VERSION}"
    (RUNS_DIR / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
                                            encoding="utf-8")
    source_summary = RUNS_DIR / f"{args.run}.summary.json"
    inherited = json.loads(source_summary.read_text(encoding="utf-8")) if source_summary.exists() else {}
    keep = ("version", "llm_provider", "llm_model", "embed_model", "top_k")
    summary = {k: inherited[k] for k in keep if k in inherited} | summarize(out_rows) | {
        "run": name, "regraded_from": args.run, "judge_rubric": RUBRIC_VERSION,
        "judge": f"{args.judge_backend}:{args.judge_model}"}
    (RUNS_DIR / f"{name}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"re-graded {len(out_rows)} answers -> eval/runs/{name}.jsonl")


if __name__ == "__main__":
    main()
