"""Postgres storage: one table of chunks with a pgvector embedding and a full-text index.

Both retrieval legs live in the same database:
- vector search: cosine distance on ``embedding`` (HNSW index);
- lexical search: Postgres full-text search with the Portuguese configuration (GIN index),
  which handles stemming ("coberturas" ~ "cobertura") and exact codes and numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from rag.chunking import Chunk
from rag.config import get_settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    text PRIMARY KEY,
    doc_id      text NOT NULL,
    unit        text NOT NULL,
    text        text NOT NULL,
    source_url  text NOT NULL,
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('portuguese', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    unit: str
    text: str
    source_url: str
    score: float


class Store:
    def __init__(self, url: str | None = None):
        s = get_settings()
        self.dim = s.embed_dim
        self.conn = psycopg.connect(url or s.database_url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def init_schema(self, reset: bool = False) -> None:
        if reset:
            self.conn.execute("DROP TABLE IF EXISTS chunks")
        self.conn.execute(SCHEMA.format(dim=self.dim))

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        rows = [
            (c.chunk_id, c.doc_id, c.unit, c.text, c.source_url, np.asarray(e, dtype=np.float32))
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (chunk_id, doc_id, unit, text, source_url, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO UPDATE SET
                     doc_id = EXCLUDED.doc_id, unit = EXCLUDED.unit, text = EXCLUDED.text,
                     source_url = EXCLUDED.source_url, embedding = EXCLUDED.embedding""",
                rows,
            )

    def delete_doc(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

    def count(self) -> tuple[int, int]:
        n_chunks, n_docs = self.conn.execute("SELECT count(*), count(DISTINCT doc_id) FROM chunks").fetchone()
        return n_chunks, n_docs

    def vector_search(self, embedding: list[float], k: int) -> list[Hit]:
        rows = self.conn.execute(
            """SELECT chunk_id, doc_id, unit, text, source_url, 1 - (embedding <=> %s) AS score
               FROM chunks ORDER BY embedding <=> %s LIMIT %s""",
            (np.asarray(embedding, dtype=np.float32),) * 2 + (k,),
        ).fetchall()
        return [Hit(*r) for r in rows]

    def lexical_search(self, query: str, k: int) -> list[Hit]:
        rows = self.conn.execute(
            """SELECT chunk_id, doc_id, unit, text, source_url,
                      ts_rank_cd(tsv, q) AS score
               FROM chunks, websearch_to_tsquery('portuguese', %s) q
               WHERE tsv @@ q ORDER BY score DESC LIMIT %s""",
            (query, k),
        ).fetchall()
        return [Hit(*r) for r in rows]
