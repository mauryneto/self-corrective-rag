"""Streaming, measured: when the first words arrive, and whether streaming changes the answer.

    uv run python -m rag.eval.stream_check --run v4_repro --every 5 --name v4_stream

A sample of the eval questions goes through the real pipeline with a listener, the way the API's
``POST /ask/stream`` runs it. For each question it records when the passages were chosen, when
the first piece of text arrived and when the answer was complete, and compares the answer with
the one the stored run ``--run`` has for the same question. Nothing is judged: an identical
answer keeps its verdict.

One question is answered before the sample and thrown away, so no row pays for loading a model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from rag.config import ROOT, get_settings
from rag.eval.dataset import load_questions
from rag.eval.run import get_pipeline

RUNS = ROOT / "eval" / "runs"


def timed_answer(pipeline, question: str) -> dict:
    t0 = time.perf_counter()
    first: list[float] = []

    def on_event(kind: str, data: dict) -> None:
        if kind == "text" and not first:
            first.append(time.perf_counter() - t0)

    a = pipeline.answer(question, on_event=on_event)
    return {"answer": a.answer, "chunk_ids": [h.chunk_id for h in a.sources],
            "retrieval_s": round(a.retrieval_latency_s, 3), "first_token_s": round(first[0], 3),
            "total_s": round(a.latency_s, 3), "llm_calls": len(a.usage.calls),
            "input_tokens": a.usage.input_tokens, "output_tokens": a.usage.output_tokens}


def summarize(rows: list[dict]) -> dict:
    """Timings over the questions the model answered; a question the gate declines has no text to wait for."""
    answered = [r for r in rows if r["llm_calls"]]
    same_passages = [r for r in rows if r["same_passages"]]

    def stats(key: str) -> dict:
        values = sorted(r[key] for r in answered)
        return {"median": round(statistics.median(values), 2), "p95": round(values[int(0.95 * (len(values) - 1))], 2)}

    return {
        "questions": len(rows),
        "answered_by_the_model": len(answered),
        "declined_by_the_gate": len(rows) - len(answered),
        "same_passages_as_stored": len(same_passages),
        "identical_answer_of_those": sum(r["same_answer"] for r in same_passages),
        "retrieval_s": stats("retrieval_s"),
        "first_token_s": stats("first_token_s"),
        "total_s": stats("total_s"),
        "output_tokens_median": statistics.median(r["output_tokens"] for r in answered),
        "gate_declines_total_s_median": (round(statistics.median(r["total_s"] for r in rows if not r["llm_calls"]), 2)
                                         if len(answered) < len(rows) else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v4")
    parser.add_argument("--run", default="v4_repro", help="stored run whose answers the streamed ones are compared with")
    parser.add_argument("--every", type=int, default=5, help="take one question in N, in the order of the eval set")
    parser.add_argument("--name", default="v4_stream")
    args = parser.parse_args()

    stored = {r["id"]: r for r in map(json.loads, (RUNS / f"{args.run}.jsonl").read_text(encoding="utf-8").splitlines())}
    questions = load_questions()
    sample = questions[:: args.every]
    pipeline = get_pipeline(args.version)
    timed_answer(pipeline, questions[1].question)  # loads the models; not in the sample

    rows = []
    for n, q in enumerate(sample, 1):
        r = timed_answer(pipeline, q.question)
        old = stored[q.id]
        r = {"id": q.id, "type": q.type,
             "same_passages": r.pop("chunk_ids") == [s["chunk_id"] for s in old["sources"]],
             "same_answer": r.pop("answer") == old["answer"]} | r
        rows.append(r)
        print(f"[{n}/{len(sample)}] {q.id} {q.type:12s} retrieval {r['retrieval_s']:5.2f}s first {r['first_token_s']:6.2f}s "
              f"total {r['total_s']:6.2f}s | same passages {r['same_passages']} same answer {r['same_answer']}", flush=True)

    s = get_settings()
    out = {"name": args.name, "version": args.version, "compared_with": args.run, "llm_model": pipeline.llm.model,
           "ollama_num_gpu": s.ollama_num_gpu, "embed_num_gpu": s.embed_num_gpu, "summary": summarize(rows), "rows": rows}
    path = RUNS / f"{args.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=1))
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
