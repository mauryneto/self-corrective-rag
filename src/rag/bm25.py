"""BM25 over the chunks, using Postgres' Portuguese stemmer for tokens.

Postgres full-text ranking (``ts_rank``) has no notion of how rare a term is across the corpus,
so in a regulatory corpus the words every norm uses ("operadora", "plano", "beneficiário")
outweigh the distinctive ones. BM25 weighs each term by its inverse document frequency. The
corpus is small (a few thousand chunks), so the index is built in memory from the stored
``tsvector``s in well under a second: tokens come from the same stemmer as the database.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# A tsvector renders as: 'lexeme':1,5 'other':3A ...  (positions, optionally weighted)
_TSV_RE = re.compile(r"'((?:[^']|'')+)':([\d,A-D]+)")


def parse_tsvector(tsv: str) -> Counter[str]:
    """Term frequencies from a tsvector's text form: the number of positions of each lexeme."""
    return Counter({m.group(1).replace("''", "'"): len(m.group(2).split(",")) for m in _TSV_RE.finditer(tsv)})


@dataclass
class BM25Index:
    ids: list[str]
    tfs: list[Counter[str]]
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(self.tfs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def search(self, terms: list[str], k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = {}
        for t in set(terms):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tfs[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        best = sorted(scores, key=lambda i: -scores[i])[:k]
        return [(self.ids[i], scores[i]) for i in best]
