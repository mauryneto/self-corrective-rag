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
- ``--reuse-from RUN`` generates only what the version changes. Retrieval runs for every
  question; when the passages are the ones the stored run put in the prompt, in the same order,
  the stored answer and verdict are kept (generation is deterministic, so generating again would
  give the same text). ``--reuse-check N`` generates N of those anyway and records whether they
  match.
- ``--replay-retrieval RUN`` gives each question the passages a stored run retrieved instead of
  searching again. Retrieval does not depend on the model that answers, so this is how another
  answering model is measured on exactly the same passages (set it with ``OLLAMA_LLM_MODEL``).
  The latency of such a run is the model's time only.
- If MLflow is installed, the run is logged to ``mlflow.db`` (parameters + summary metrics).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

from rag.config import ROOT, get_settings
from rag.costs import CallRecord, Usage
from rag.eval.dataset import Question, load_questions
from rag.eval.judge import DEFAULT_JUDGE_MODEL, RUBRIC_VERSION, JudgeCache, get_judge, grade
from rag.eval.metrics import retrieval_scores, summarize
from rag.pipeline.prompts import ANSWER_SYSTEMS
from rag.pipeline.v0 import Answer

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
    if version == "v4":
        from rag.pipeline.v4 import V4Pipeline

        return V4Pipeline()
    if version in ("floor", "ceiling", "ceiling_gold"):  # reference rows, see rag.eval.reference
        from rag.eval.reference import CeilingPipeline, FloorPipeline, load_replay

        if version == "floor":
            return FloorPipeline()
        return CeilingPipeline(load_questions(), load_replay() if version == "ceiling" else None)
    raise SystemExit(f"unknown pipeline version: {version}")


REUSABLE = ("v0", "v1", "v3", "v4")  # one generation call with the same prompt over the retrieved passages


def load_reusable(run: str, answer_prompt: str, llm_model: str | None) -> dict[str, dict]:
    """Records of a stored run whose answers another version may keep (``--reuse-from``)."""
    summary = json.loads((RUNS_DIR / f"{run}.summary.json").read_text(encoding="utf-8"))
    same_way = (summary.get("version") in REUSABLE and summary.get("answer_prompt", "default") == answer_prompt
                and summary.get("llm_model") == llm_model)
    if not same_way:
        raise SystemExit(f"--reuse-from: {run} was not generated the way this run would generate")
    rows = (json.loads(line) for line in (RUNS_DIR / f"{run}.jsonl").read_text(encoding="utf-8").splitlines() if line)
    return {r["id"]: r for r in rows if r.get("verdict")}


def replay_retrieval(pipeline, run: str, questions: list[Question]) -> None:
    """Make ``pipeline.retrieve`` return what a stored run retrieved: same passages, order and scores."""
    rows = (json.loads(line) for line in (RUNS_DIR / f"{run}.jsonl").read_text(encoding="utf-8").splitlines() if line)
    sources = {r["id"]: r["sources"] for r in rows}
    by_text = {q.question: sources[q.id] for q in questions}

    def retrieve(question: str):
        srcs = by_text[question]
        hits = pipeline.store.get_hits([s["chunk_id"] for s in srcs])
        if len(hits) != len(srcs):
            raise SystemExit(f"--replay-retrieval: a passage of {run} is no longer in the index")
        return [replace(h, score=s["score"]) for h, s in zip(hits, srcs, strict=True)]

    pipeline.retrieve = retrieve


def same_prompt(pipeline, hits, stored: dict) -> bool:
    """Would this pipeline send the model exactly the prompt the stored answer came from?"""
    if stored.get("trace", {}).get("scope_gate", {}).get("declined"):
        return False  # that answer is a fixed refusal, the model never wrote it
    if hasattr(pipeline, "gate") and pipeline.gate(hits)["declined"]:
        return False
    return [h.chunk_id for h in hits] == [s["chunk_id"] for s in stored["sources"]]


def reused_record(q: Question, pipeline, hits, stored: dict, run: str, retrieval_s: float) -> dict:
    """The stored answer and verdict as this version's record. Retrieval is this run's (passages,
    scores, time); the model's time is the stored one, since the same call would be made."""
    usage = Usage()
    u = stored["usage"]
    usage.add(CallRecord("generate", getattr(pipeline.llm, "model", ""), u["input_tokens"], u["output_tokens"],
                         u["llm_latency_s"]))
    trace = {"scope_gate": pipeline.gate(hits)} if hasattr(pipeline, "gate") else {}
    answer = Answer(question=q.question, answer=stored["answer"], sources=hits, cited=stored["cited"],
                    version=pipeline.version, latency_s=retrieval_s + u["llm_latency_s"], usage=usage,
                    retrieval_latency_s=retrieval_s, trace=trace)
    verdict = {k: stored[k] for k in ("verdict", "refused", "citations_supported", "judge_reasoning")}
    return answer_record(q, answer) | verdict | {"reused_from": run}


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
                        help="keep this stored run's answer and verdict for every question whose retrieved passages "
                             "are the same, in the same order; only the others are generated")
    parser.add_argument("--reuse-check", type=int, default=0, metavar="N",
                        help="with --reuse-from: generate N of the reusable questions anyway and record whether the "
                             "answer matches the stored one")
    parser.add_argument("--replay-retrieval", default=None, metavar="RUN",
                        help="answer from the passages this stored run retrieved instead of searching again "
                             "(to measure another answering model on the same passages)")
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
    stored = {}
    if args.reuse_from:
        if args.version not in REUSABLE:
            raise SystemExit(f"--reuse-from does not apply to {args.version}")
        stored = load_reusable(args.reuse_from, args.answer_prompt, getattr(pipeline.llm, "model", None))
    if args.replay_retrieval:
        if args.version not in REUSABLE:
            raise SystemExit(f"--replay-retrieval does not apply to {args.version}")
        replay_retrieval(pipeline, args.replay_retrieval, questions)
    check_every = max(1, len(todo) // args.reuse_check) if args.reuse_check else 0
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

        for pos, q in enumerate(todo):
            check = False
            if q.id in stored:
                t_q = time.perf_counter()
                hits = pipeline.retrieve(q.question)
                reusable = same_prompt(pipeline, hits, stored[q.id])
                check = bool(reusable and check_every and pos % check_every == 0)
                if reusable and not check:
                    rec = reused_record(q, pipeline, hits, stored[q.id], args.reuse_from, time.perf_counter() - t_q)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done[rec["id"]] = rec
                    continue
                answer = pipeline.answer_from(q.question, hits, t_q)
            else:
                answer = pipeline.answer(q.question)
            rec = answer_record(q, answer)
            if check:
                rec["reuse_check"] = answer.answer == stored[q.id]["answer"]
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
        "retrieval_replayed_from": args.replay_retrieval,
        "reused_from": args.reuse_from, "reused": sum("reused_from" in r for r in results),
        "reuse_checked": sum("reuse_check" in r for r in results),
        "reuse_check_identical": sum(bool(r.get("reuse_check")) for r in results),
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
