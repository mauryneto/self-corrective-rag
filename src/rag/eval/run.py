"""Run the eval set through one pipeline version and grade every answer.

    uv run python -m rag.eval.run --version v0 [--limit N] [--judge-backend claude-code|anthropic]

- Results go to ``eval/runs/<run_name>.jsonl`` (one line per question, written as soon as it is
  graded) and the summary to ``eval/runs/<run_name>.summary.json``. Re-running with the same
  ``--name`` resumes: questions already in the file are skipped.
- Answers are generated one at a time (the local model uses the GPU); grading runs in parallel
  threads, so the judge works while the next answer is being generated.
- If MLflow is installed, the run is logged to ``./mlruns`` (parameters + summary metrics).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

from rag.config import ROOT, get_settings
from rag.eval.dataset import Question, load_questions
from rag.eval.judge import DEFAULT_JUDGE_MODEL, get_judge, grade
from rag.eval.metrics import retrieval_scores, summarize

RUNS_DIR = ROOT / "eval" / "runs"


def get_pipeline(version: str):
    if version == "v0":
        from rag.pipeline.v0 import V0Pipeline

        return V0Pipeline()
    if version == "v1":
        from rag.pipeline.v1 import V1Pipeline

        return V1Pipeline()
    raise SystemExit(f"unknown pipeline version: {version}")


def answer_record(q: Question, answer) -> dict:
    sources = [{"index": i, "chunk_id": h.chunk_id, "doc_id": h.doc_id, "unit": h.unit, "score": round(h.score, 4)}
               for i, h in enumerate(answer.sources, 1)]
    recall, rr = retrieval_scores([g.model_dump() for g in q.gold], sources) if q.gold else (None, None)
    return {
        "id": q.id, "type": q.type, "question": q.question, "answer": answer.answer,
        "cited": answer.cited, "sources": sources, "recall_at_5": recall, "rr": rr,
        "latency_s": round(answer.latency_s, 3), "retrieval_latency_s": round(answer.retrieval_latency_s, 3),
        "cost_usd": answer.usage.cost_usd, "usage": answer.usage.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v0")
    parser.add_argument("--name", default=None, help="run name (default: <version>_<timestamp>); reuse it to resume")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    parser.add_argument("--judge-backend", default="claude-code", choices=["claude-code", "anthropic"])
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-workers", type=int, default=3)
    args = parser.parse_args()

    name = args.name or f"{args.version}_{datetime.now():%Y%m%d_%H%M}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{name}.jsonl"
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r

    questions = load_questions()[: args.limit] if args.limit else load_questions()
    todo = [q for q in questions if q.id not in done]
    print(f"run {name}: {len(questions)} questions, {len(done)} already done, {len(todo)} to go")

    pipeline = get_pipeline(args.version)
    judge = get_judge(args.judge_backend, args.judge_model)
    t0 = time.perf_counter()

    def judge_and_save(q: Question, rec: dict, cited: dict[int, str]) -> dict:
        try:
            j = grade(judge, q.question, q.reference_answer, q.answerable, rec["answer"], cited)
            rec |= {"verdict": j.verdict.verdict, "refused": j.verdict.refused,
                    "citations_supported": j.verdict.citations_supported, "judge_reasoning": j.verdict.reasoning,
                    "judge_latency_s": round(j.record.latency_s, 2)}
        except Exception as exc:  # noqa: BLE001 - keep the answer; it can be re-graded later
            rec |= {"verdict": None, "refused": None, "citations_supported": None, "judge_error": str(exc)[:300]}
        return rec

    with ThreadPoolExecutor(max_workers=args.judge_workers) as pool, out_path.open("a", encoding="utf-8") as fh:
        pending: list[Future] = []

        def flush(block: bool) -> None:
            for f in list(pending):
                if block or f.done():
                    rec = f.result()
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done[rec["id"]] = rec
                    pending.remove(f)

        for q in todo:
            answer = pipeline.answer(q.question)
            rec = answer_record(q, answer)
            cited = {n: answer.sources[n - 1].text for n in answer.cited}
            pending.append(pool.submit(judge_and_save, q, rec, cited))
            flush(block=False)
            elapsed = time.perf_counter() - t0
            print(f"[{len(done) + len(pending)}/{len(questions)}] {q.id} {q.type:12s} "
                  f"answer {answer.latency_s:5.1f}s | elapsed {elapsed / 60:5.1f} min", flush=True)
        flush(block=True)

    results = [done[q.id] for q in questions if q.id in done]
    graded = [r for r in results if r.get("verdict")]
    failed = len(results) - len(graded)
    summary = summarize(graded) | {
        "run": name, "version": args.version, "judge": f"{args.judge_backend}:{args.judge_model}",
        "llm_provider": get_settings().llm_provider,
        "llm_model": getattr(pipeline.llm, "model", None), "embed_model": get_settings().embed_model,
        "top_k": get_settings().top_k, "judge_failures": failed, "date": datetime.now().isoformat(timespec="seconds"),
    }
    (RUNS_DIR / f"{name}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    _log_mlflow(summary)


def _log_mlflow(summary: dict) -> None:
    try:
        import mlflow
    except ImportError:
        return
    try:
        # SQLite backend (the plain-folder store is deprecated). Browse with:
        #   uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("self-corrective-rag")
        params = {k: summary[k] for k in ("version", "judge", "llm_provider", "llm_model", "embed_model", "top_k")}
        metrics = {k: v for k, v in summary.items() if isinstance(v, int | float) and not isinstance(v, bool)}
        with mlflow.start_run(run_name=summary["run"]):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception as exc:  # noqa: BLE001 - tracking must never lose a finished eval
        print(f"MLflow logging skipped: {exc}")


if __name__ == "__main__":
    main()
