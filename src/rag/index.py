"""Chunk the corpus, embed it and load it into Postgres.

Usage:
    uv run python -m rag.index            # (re)index every norm in data/corpus
    uv run python -m rag.index --reset    # drop the table first
    uv run python -m rag.index --changed  # only what the last build_corpus run changed or removed
"""

from __future__ import annotations

import argparse
import json
import time

from rag.chunking import chunk_document
from rag.config import get_settings
from rag.embeddings import OllamaEmbedder
from rag.ingest.build_corpus import load_corpus
from rag.store import Store


def index_docs(docs: list[dict], store: Store, embedder: OllamaEmbedder, max_chars: int = 1800) -> int:
    t0, total = time.perf_counter(), 0
    for i, doc in enumerate(docs, 1):
        chunks = chunk_document(doc, max_chars=max_chars)
        vectors = embedder.embed([c.text for c in chunks])
        store.delete_doc(doc["doc_id"])  # re-indexing a norm replaces all of its chunks
        store.upsert(chunks, vectors)
        total += len(chunks)
        if i % 20 == 0 or i == len(docs):
            print(f"[{i}/{len(docs)}] {total} chunks ({time.perf_counter() - t0:.0f}s)")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="drop and rebuild the chunks table")
    parser.add_argument("--changed", action="store_true", help="only apply data/corpus/_changes.json")
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()

    docs = load_corpus()
    if not docs:
        raise SystemExit("data/corpus is empty: run `python -m rag.ingest.build_corpus` first")

    store = Store()
    store.init_schema(reset=args.reset)

    if args.changed:
        changes = json.loads((get_settings().corpus_dir / "_changes.json").read_text(encoding="utf-8"))
        for doc_id in changes["removed"]:
            store.delete_doc(doc_id)
        wanted = set(changes["changed"])
        docs = [d for d in docs if d["doc_id"] in wanted]
        print(f"applying changes from {changes['date']}: {len(docs)} to index, {len(changes['removed'])} removed")

    if docs:
        index_docs(docs, store, OllamaEmbedder(), args.max_chars)
    n_chunks, n_docs = store.count()
    print(f"index ready: {n_chunks} chunks from {n_docs} norms")


if __name__ == "__main__":
    main()
