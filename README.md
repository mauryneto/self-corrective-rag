# self-corrective-rag

A self-corrective RAG over Brazilian health-insurance regulation (normative acts of ANS, the
national regulator). An LLM judge checks every answer; failed answers trigger query rewriting and
decomposition; questions the corpus can't answer are declined. Every design choice is backed by an
ablation on a fixed eval set, reporting accuracy, refusal precision, latency and cost per query.

> **Status: work in progress (week 1 of 4).** The corpus pipeline (with incremental updates),
> the index and the baseline (V0) are in place. The eval set, the self-corrective loop, the guardrail and the results table
> come next. No numbers are reported until they are measured.

## Roadmap

| Version | What it adds | Status |
|---|---|---|
| V0 | Naive RAG: vector search only | ✅ |
| V1 | Hybrid search (vector + full-text) and a reranker | ⏳ |
| V2 | Self-corrective loop: LLM judge, query rewriting and decomposition | ⏳ |
| V3 | Scope guardrail: decline what the corpus can't answer | ⏳ |
| V4 | Optimized loop: cheaper judge, early exit, caching | ⏳ |

## Corpus

- **Source of truth:** the ANS legislation portal
  ([componentes-portal.ans.gov.br/link/legislacoes](https://componentes-portal.ans.gov.br/link/legislacoes)).
  It lists the Resoluções Normativas (RNs) in force and shows each one **consolidated**: passages
  changed or revoked by later norms are struck through next to the new wording, and the page says
  when the text was last updated. The pipeline drops the struck passages, so the corpus holds the
  text in force today, not the text as first published.
- **Fallback:** when the portal has no text for a norm, the original is fetched from the official
  gazette, the *Diário Oficial da União* ([in.gov.br](https://www.in.gov.br)), and marked
  `source: "dou"` (not consolidated).
- **How it is read:** the portal is a JavaScript app, so a headless browser (Playwright) uses it
  the way a person would: the public list of norms and each norm's public page, one at a time.
- **Licensing:** official acts are not protected by copyright in Brazil (Lei 9.610/98, art. 8º, IV).
- **Known limitation:** annexes published as separate files (e.g. the procedure tables of RN 465)
  are not parsed yet; the articles that reference them are.

### Keeping it up to date

```bash
uv run python -m rag.ingest.update
```

One incremental run: re-reads the in-force list, rewrites only the norms whose "Atualizado em"
date changed, adds new norms, removes revoked ones, then re-embeds only what changed. Removal has
guard rails: nothing is deleted unless the whole list was read, and never more than 10 norms or
20% of the corpus in one run. Each run writes what it did to `data/corpus/_changes.json`.
Schedule it (cron, Windows Task Scheduler) to run weekly.

## Quickstart

Requirements: Docker, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) with
`bge-m3` (embeddings) and a chat model pulled.

```bash
cp .env.example .env
docker compose up -d                       # Postgres 17 + pgvector on 127.0.0.1:5433
uv sync
uv run playwright install chromium        # headless browser for the ANS portal
uv run python -m rag.ingest.build_corpus   # ANS portal (consolidated texts) -> data/corpus/
uv run python -m rag.index                 # chunk, embed, load into Postgres
uv run uvicorn rag.api:app --reload        # POST /ask
```

```bash
curl -s localhost:8000/ask -H "content-type: application/json" \
  -d '{"question": "Qual o prazo máximo para atendimento de consulta básica?"}'
```

To answer with Claude instead of a local model, set `LLM_PROVIDER=anthropic` and
`ANTHROPIC_API_KEY` in `.env`.

## Design decisions

- **Structural chunking.** Chunks follow the legal structure (one article, split at paragraph
  boundaries only when too long) and carry a header with the norm and article number, so a
  retrieved chunk is citable on its own.
- **Postgres for both search legs.** pgvector for embeddings and Postgres full-text search
  (Portuguese configuration) for the lexical leg, so hybrid search is one database, not two.
- **Local embeddings.** `bge-m3` is multilingual and runs locally through Ollama: re-indexing
  costs nothing and the eval's cost numbers only reflect the LLM calls.
- **Provider-agnostic LLM layer.** Every call records tokens, latency and estimated cost, so the
  versions can be compared on price as well as quality.

## License

MIT
