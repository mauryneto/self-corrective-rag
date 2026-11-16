# self-corrective-rag

A self-corrective RAG over Brazilian health-insurance regulation (normative acts of ANS, the
national regulator). An LLM judge checks every answer; failed answers trigger query rewriting and
decomposition; questions the corpus can't answer are declined. Every design choice is backed by an
ablation on a fixed eval set, reporting accuracy, refusal precision, latency and cost per query.

> **Status: work in progress (week 2 of 4).** Corpus pipeline with incremental updates, index,
> baseline (V0), a 180-question eval set and its first measured results. The self-corrective loop
> and the guardrail come next.

## Results

Measured on the 180-question eval set (`eval/runs/`). Each new version gets a row.

| Version | Correct | Correct + partial | Multi-hop correct | Recall@5 | MRR | Declines when it should | Refusal precision | Citations supported | p50 / p95 latency | Cost / question |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 43% | 77% | 5% | 73% | 0.61 | 97% | 94% | 69% | 28s / 44s | US$ 0 (local) |

"Correct", "Multi-hop correct", Recall@5 and MRR are over the 120 answerable questions; the two
refusal columns over the 60 it should decline. V0 answers with `qwen2.5:14b` on a laptop GPU
(RTX 5070, 8 GB), hence the latency and zero cost. Judge: Claude Haiku 5.5.

**What V0 shows:**
- **Multi-hop is the weak spot: 5% correct.** Retrieval finds every needed article for only 8 of
  the 40 multi-hop questions (Recall@5 50%, against 85% on direct questions). One vector search
  for a question that needs two norms brings back one of them. This is what V1 (hybrid search and
  reranking) and V2 (query decomposition and a second retrieval pass) are for.
- **Declining works already: 97%.** The prompt's "say when the excerpts are not enough" holds on
  unanswerable and out-of-scope questions, with 3% of answerable questions wrongly declined.
- **Citations: 69% supported.** The local model sometimes fills gaps from memory and attributes
  rules to revoked norms (e.g. RN 259 instead of RN 566).

> The judge's grades are not yet validated against human labels; that comparison (50 labelled
> answers, agreement reported as Cohen's kappa) is the next step and may change these numbers.

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
- **Coverage (Nov 2026):** 240 RNs in force, 4,124 chunks: 234 consolidated texts from the
  portal and 6 originals from the gazette. 14 norms have no text in either source (mostly from
  2002–2011); they are listed in `data/corpus/_changes.json` under `failed`. Norms the portal
  marks "Não vigente" (e.g. published but not yet in effect) are left out.
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

## Eval set

`eval/questions.jsonl`: 180 questions in Portuguese, fixed and versioned.

| Type | Count | What it tests |
|---|---|---|
| direct | 80 | one article answers it |
| multi_hop | 40 | the answer combines two or more articles, often from different norms |
| unanswerable | 40 | sounds like ANS regulation, but the corpus does not answer it: the system must say so |
| out_of_scope | 20 | not ANS regulation at all: the system must decline |

- **How it was written:** drafted by Claude from the corpus text, following
  [`eval/question_brief.md`](eval/question_brief.md). Every reference answer carries a verbatim
  quote of the article that supports it.
- **How it is checked:** `python -m rag.eval.check_questions` verifies, mechanically, that every
  gold norm and article exists and every quote is word for word in that article; for unanswerable
  questions it searches the corpus for the key terms. `python -m rag.eval.review_sheet` writes
  `eval/review.md`, with each question next to its quote, for human review.
- **Gold sources are (norm, article) pairs**, not chunk ids, so they survive changes to chunking.

## Judge

`rag.eval.judge` grades each answer as correct / partial / incorrect, plus whether the system
declined and whether its citations support it. Same prompt and JSON schema on two backends:
Claude Code headless (`claude -p`, no tools, no MCP servers) or the Anthropic API. The default
model is Claude Haiku 5.5. The judge is separate from anything inside the pipeline, and is
validated against human labels before its grades are trusted.

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
