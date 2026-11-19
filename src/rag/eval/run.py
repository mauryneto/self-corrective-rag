"""Run the eval set through one pipeline version and grade every answer.

    uv run python -m rag.eval.run --version v0 [--limit N] [--judge-backend claude-code|anthropic]

``--version`` also takes the reference rows ``floor``, ``ceiling`` and ``ceiling_gold``
(``rag.eval.reference``). ``--answer-prompt complete`` swaps the answering prompt for an experiment.

- Results go to ``eval/runs/<run_name>.jsonl`` (one line per question, written as soon as it is
  graded) and the summary to ``eval/runs/<run_name>.summary.json``. Re-running with the same
  ``--name`` resumes: questions already in the file are skipped.
- Answers are generated one at a time (the local model uses the GPU); grading runs in parallel
  threads, so the judge works while the next answer is being generated.
- Verdicts are cached in ``eval/judge_cache.jsonl`` by everything the judge sees: an answer that
  comes out identical in a later run keeps its verdict (``--no-judge-cache`` to grade afresh).
- ``--reuse-from v1_repro`` (V3 only) runs just the questions V3 treats differently from V1: the
  ones its scope gate declines. The others keep the stored V1 answer and verdict.
- If MLflow is installed, the run is logged to ``mlflow.db`` (parameters + summary metrics).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

from rag.config import ROOT, get_settings
from rag.eval.dataset import Question, load_questions
from rag.eval.judge import DEFAULT_JUDGE_MODEL, RUBRIC_VERSION, JudgeCache, get_judge, grade
from rag.eval.metrics import retrieval_scores, summarize
from rag.pipeline.prompts import ANSWER_SYSTEMS

RUNS_DIR = ROOT / "eval" / "runs"
JUDGE_CACHE = ROOT / "eval" / "judge_cache.jsonl"


def get_pipeline(version: str):
    if version == "v0":
        from rag.pipeline.v0 import V0Pipeline

        return V0Pipeline()
    if version == "v1":
        from rag.pipeline.v1 import V1Pipeline

        return V1Pipeline()
    if version == "v2":
        from rag.pipeline.v2 import V2Pipeline

        return V2Pipeline()
    if version == "v3":
        from rag.pipeline.v3 import V3Pipeline

        return V3Pipeline()
    if version in ("floor", "ceiling", "ceiling_gold"):  # reference rows, see rag.eval.reference
        from rag.eval.reference import CeilingPipeline, FloorPipeline, load_replay

        if version == "floor":
            return FloorPipeline()
        return CeilingPipeline(load_questions(), load_replay() if version == "ceiling" else None)
    raise SystemExit(f"unknown pipeline version: {version}")


def reuse_past_gate(run: str, min_top_score: float) -> dict[str, dict]:
    """Stored V1 records of the questions whose best passage passes V3's scope gate.

    Past the gate V3 does exactly what V1 does, and generation is deterministic (README,
    "Reproducibility"), so the stored answer and verdict are V3's too. Generating them again would
    only repeat the V1 run. Each record is marked with ``reused_from``; its latency is the one
    measured in that run.
    """
    summary = json.loads((RUNS_DIR / f"{run}.summary.json").read_text(encoding="utf-8"))
    if summary.get("version") != "v1" or summary.get("answer_prompt", "default") != "default":
        raise SystemExit(f"--reuse-from needs a V1 run made with the official prompt; {run} is not one")
    out = {}
    for line in (RUNS_DIR / f"{run}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        top = max((s["score"] for s in r["sources"]), default=0.0)
        if top >= min_top_score:
            gate = {"top_score": top, "min_top_score": min_top_score, "declined": False}
            out[r["id"]] = r | {"trace": {"scope_gate": gate}, "reused_from": run}
    return out


def answer_record(q: Question, answer) -> dict:
    sources = [{"index": i, "chunk_id": h.chunk_id, "doc_id": h.doc_id, "unit": h.unit, "score": round(h.score, 4)}
               for i, h in enumerate(answer.sources, 1)]
    gold = [g.model_dump() for g in q.gold]
    recall, rr = retrieval_scores(gold, sources) if gold else (None, None)
    # share of the gold articles anywhere in the prompt (V2 can add passages after the first 6)
    context_recall = retrieval_scores(gold, sources, k=len(sources))[0] if gold else None
    return {
        "id": q.id, "type": q.type, "question": q.question, "answer": answer.answer,
        "cited": answer.cited, "sources": sources, "recall_at_5": recall, "rr": rr, "context_recall": context_recall,
        "latency_s": round(answer.latency_s, 3), "retrieval_latency_s": round(answer.retrieval_latency_s, 3),
        "cost_usd": answer.usage.cost_usd, "usage": answer.usage.summary(),
        **({"trace": answer.trace} if answer.trace else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v0")
    parser.add_argument("--name", default=None, help="run name (default: <version>_<timestamp>); reuse it to resume")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    parser.add_argument("--answerable-only", action="store_true",
                        help="only the questions the corpus answers (for ceiling_gold, which has nothing to show "
                             "for the others)")
    parser.add_argument("--answer-prompt", default="default", choices=sorted(ANSWER_SYSTEMS),
                        help="system prompt of the answering step (rag.pipeline.prompts.ANSWER_SYSTEMS); "
                             "anything but the default is an experiment and needs its own --name")
    parser.add_argument("--reuse-from", default=None, metavar="RUN",
                        help="v3 only: questions that pass the scope gate keep the answer and verdict of this stored "
                             "V1 run instead of being generated again")
    parser.add_argument("--judge-backend", default="claude-code", choices=["claude-code", "anthropic"])
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-workers", type=int, default=3)
    parser.add_argument("--no-judge-cache", action="store_true",
                        help="grade every answer afresh instead of reusing verdicts for identical answers")
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

    questions = [q for q in load_questions() if q.answerable or not args.answerable_only]
    questions = questions[: args.limit] if args.limit else questions
    todo = [q for q in questions if q.id not in done]
    print(f"run {name}: {len(questions)} questions, {len(done)} already done, {len(todo)} to go")

    pipeline = get_pipeline(args.version)
    if args.answer_prompt != "default":
        if not hasattr(pipeline, "answer_system"):
            raise SystemExit(f"--answer-prompt does not apply to {args.version}: it has its own prompt")
        pipeline.answer_system = ANSWER_SYSTEMS[args.answer_prompt]
    if args.reuse_from:
        if args.version != "v3":
            raise SystemExit("--reuse-from only applies to v3: it is V1 plus a gate before the model")
        kept = reuse_past_gate(args.reuse_from, pipeline.min_top_score)
        with out_path.open("a", encoding="utf-8") as fh:
            for q in todo:
                if q.id in kept:
                    fh.write(json.dumps(kept[q.id], ensure_ascii=False) + "\n")
                    done[q.id] = kept[q.id]
        todo = [q for q in todo if q.id not in done]
        print(f"kept from {args.reuse_from}: {sum('reused_from' in r for r in done.values())}, {len(todo)} to run")
    judge = get_judge(args.judge_backend, args.judge_model)
    cache = None if args.no_judge_cache else JudgeCache(JUDGE_CACHE)
    t0 = time.perf_counter()

    def judge_and_save(q: Question, rec: dict, cited: dict[int, str]) -> dict:
        try:
            j = grade(judge, q.question, q.reference_answer, q.answerable, rec["answer"], cited, cache=cache)
            rec |= {"verdict": j.verdict.verdict, "refused": j.verdict.refused,
                    "citations_supported": j.verdict.citations_supported, "judge_reasoning": j.verdict.reasoning,
                    "judge_latency_s": round(j.record.latency_s, 2), "judge_cached": j.cached}
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
        "judge_rubric": RUBRIC_VERSION, "answer_prompt": args.answer_prompt,
        "reused_from": args.reuse_from, "reused": sum("reused_from" in r for r in results),
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
