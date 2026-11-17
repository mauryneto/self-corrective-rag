"""Retrieval-only evaluation: Recall@k and MRR of each retrieval setup on the 120 answerable questions.

    uv run python -m rag.eval.retrieval_eval [--configs vector,lexical,hybrid,hybrid+rerank] [--k 5]

No answer is generated and no judge is called, so a setup is measured in seconds (minutes with a
reranker on CPU). Used to choose V1's retrieval before paying for a full eval run. Results are
written to ``eval/runs/retrieval_<date>.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime

from rag.embeddings import OllamaEmbedder
from rag.eval.dataset import load_questions
from rag.eval.metrics import retrieval_scores
from rag.eval.run import RUNS_DIR
from rag.retrieval import Retriever
from rag.store import Store


def build(config: str, store: Store, embedder: OllamaEmbedder, pool: int, top_k: int) -> Retriever:
    mode, _, extra = config.partition("+")
    reranker = None
    if extra == "rerank":
        from rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    return Retriever(store, embedder, mode=mode, top_k=top_k, pool=pool, reranker=reranker)


def evaluate(retriever: Retriever, questions, k: int) -> dict:
    per_type: dict[str, list[tuple[float, float]]] = {"direct": [], "multi_hop": []}
    t0 = time.perf_counter()
    for q in questions:
        hits = retriever.retrieve(q.question)
        sources = [{"doc_id": h.doc_id, "unit": h.unit} for h in hits]
        per_type[q.type].append(retrieval_scores([g.model_dump() for g in q.gold], sources, k=k))
    elapsed = time.perf_counter() - t0
    out = {"seconds_per_question": round(elapsed / len(questions), 3)}
    for t, scores in [("all", per_type["direct"] + per_type["multi_hop"]), *per_type.items()]:
        out[t] = {f"recall@{k}": round(statistics.mean(s[0] for s in scores), 4),
                  "all_gold_found": sum(1 for s in scores if s[0] == 1.0),
                  "mrr": round(statistics.mean(s[1] for s in scores), 4), "n": len(scores)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default="vector,lexical,hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=30)
    args = parser.parse_args()

    questions = [q for q in load_questions() if q.answerable]
    store, embedder = Store(), OllamaEmbedder()
    results = {}
    for config in args.configs.split(","):
        results[config] = evaluate(build(config, store, embedder, args.pool, top_k=max(args.k, 6)), questions, args.k)
        r = results[config]
        recall = f"recall@{args.k}"
        print(f"{config:16s} {recall} {r['all'][recall]:.3f} (direct {r['direct'][recall]:.3f}, "
              f"multi-hop {r['multi_hop'][f'recall@{args.k}']:.3f}, "
              f"multi-hop complete {r['multi_hop']['all_gold_found']}/40) "
              f"| MRR {r['all']['mrr']:.3f} | {r['seconds_per_question']}s/q", flush=True)
    out = RUNS_DIR / f"retrieval_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"k": args.k, "pool": args.pool, "results": results}, indent=1), encoding="utf-8")
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
