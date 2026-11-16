"""Keep the corpus and the index in sync with the ANS portal: one command, meant to run on a schedule.

    uv run python -m rag.ingest.update

1. ``build_corpus``: fetch the in-force list, rewrite the norms whose "Atualizado em" changed,
   add new ones, drop revoked ones;
2. ``index --changed``: re-embed only those norms and delete the revoked ones from the index.
"""

from __future__ import annotations

import sys

from rag import index
from rag.ingest import build_corpus


def main() -> None:
    sys.argv = [sys.argv[0]]
    build_corpus.main()
    sys.argv = [sys.argv[0], "--changed"]
    index.main()


if __name__ == "__main__":
    main()
