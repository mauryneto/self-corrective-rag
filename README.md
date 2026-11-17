# self-corrective-rag

A self-corrective RAG over Brazilian health-insurance regulation (normative acts of ANS, the
national regulator). An LLM judge checks every answer; failed answers trigger query rewriting and
decomposition; questions the corpus can't answer are declined. Every design choice is backed by an
ablation on a fixed eval set, reporting accuracy, refusal precision, latency and cost per query.

> **Status: work in progress (week 3 of 4).** Corpus pipeline with incremental updates, a
> 180-question eval set, a judge validated against human labels, and V0 (baseline), V1 (hybrid
> search + reranker) and V2 (self-corrective loop) measured, together with the run-to-run noise
> that every comparison has to clear. The guardrail comes next.

## Results

Measured on the 180-question eval set (`eval/runs/`). Each new version gets a row.

| Version | Correct | Correct + partial | Multi-hop correct | Recall@5 | MRR | Declines when it should | Refusal precision | Citations supported | p50 / p95 latency | Cost / question |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 56% | 82% | 22% | 73% | 0.61 | 98% | 92% | 66% | 28s / 44s | US$ 0 (local) |
| V1 | 63% | 82% | 32% | 82% | 0.77 | 93% | 95% | 68% | 36s / 58s | US$ 0 (local) |
| V1, run again | 60% | 82% | 40% | 82% | 0.77 | 95% | 97% | 66% | 51s / 73s | US$ 0 (local) |
| V2 | 62% | 84% | 25% | 82% | 0.77 | 97% | 97% | 67% | 88s / 192s | US$ 0 (local) |

"Correct", "Multi-hop correct", Recall@5 and MRR are over the 120 answerable questions; the two
refusal columns over the 60 it should decline. Every version answers with the same prompt and
`qwen2.5:14b` on a laptop GPU (RTX 5070, 8 GB), hence the latency and zero cost; only what the
row's version changes differs. Judge: Claude Haiku 5.5, rubric v2 (validated below).

**Read the rows against the noise.** "V1, run again" is the same code, data and settings run a
second time. Local generation is not bit-for-bit reproducible even at temperature 0 (see below),
so 141 of 180 answers come out worded differently and some flip between right and wrong: direct
79% → 70%, multi-hop 32% → 40%. (The judge adds little of this: on the 39 identical answers it
changed its verdict twice.) With 120 answerable questions, a single run cannot tell apart
versions a few points apart. `python -m rag.eval.compare RUN_A RUN_B` does the paired comparison
(questions gained and lost, McNemar's exact test):

| Comparison | Correct (of 120) | Gained / lost | p |
|---|---|---|---|
| V1 vs. V1 run again (pure noise) | 76 → 72 | 7 / 11 | 0.48 |
| V0 vs. V1 | 67 → 76 | 16 / 7 | 0.09 |
| V1 vs. V2 | 76 → 75 | 12 / 13 | 1.00 |

Retrieval metrics (Recall@5, MRR) are deterministic, so V1's retrieval gain is solid; its gain in
correct answers is in the right direction but not significant on one run.

**What V0 shows:**
- **Multi-hop is the weak spot: 22% correct, against 72% on direct questions.** Retrieval finds
  every needed article for only 8 of the 40 multi-hop questions (Recall@5 50%, against 85% on
  direct ones): one vector search for a question that needs two norms brings back one of them.
  This is what V1 (hybrid search and reranking) and V2 (query decomposition and a second
  retrieval pass) are for.
- **Declining already works: 98%** of unanswerable and out-of-scope questions, with 4% of
  answerable questions wrongly declined.
- **Citations: 66% supported.** The local model sometimes fills gaps from memory and attributes
  rules to revoked norms (e.g. RN 259 instead of RN 566).

**What V1 shows** (hybrid search + cross-encoder reranker; same prompt and model as V0):
- **Retrieval clearly improves: Recall@5 73% → 82%, MRR 0.61 → 0.77.** Correct answers go
  56% → 63% (multi-hop 22% → 32%), in the expected direction but, as the noise check above shows,
  not significant on a single run (p = 0.09).
- **The reranker did the work, not the hybrid fusion.** In the retrieval-only ablation below,
  fusing BM25 with the vectors did not beat vector search on this corpus (bge-m3 is already
  strong and BM25 adds noise); the 568M multilingual cross-encoder is what moved recall.
- **A trade-off appeared in declining: 98% → 93%.** With more relevant-looking passages in the
  prompt, the model answers some questions it should decline. That is the job of V3's guardrail.
- **It costs latency:** the reranker runs on CPU (~10 s per question), p50 28 s → 36 s. Cutting
  that is part of V4.
- **Multi-hop is still the weak spot** (32%): even with a better ranking, a single search for a
  question that needs two norms finds every needed article for only 16 of 40. That is V2's job:
  decompose the question and search again.

**What V2 shows** (V1 plus a self-corrective loop, every step run by the same 14B model: check
whether the passages cover the question and, if not, search for what is missing; check the
answer against the passages it cites and rewrite it once if a claim is unsupported):
- **No measurable gain in correct answers** (62% against 60–63% for V1's two runs), at 2.4× the
  latency (p50 88 s): three to five LLM calls per question instead of one.
- **Declining improves a little: 97%** of unanswerable and out-of-scope questions, against
  93–95%. The context check is the useful part here: it flagged 43 of those 60 as "the passages
  don't cover this", which is the signal V3's guardrail builds on.
- **Decomposition finds little of what is missing.** In the retrieval-only check below, the
  model flags 13 of the 22 multi-hop questions whose context is incomplete and writes sensible
  follow-up queries, but those searches complete only 2 more of them (context recall 83% → 87%).
- **Multi-hop is also a reasoning problem, not only a retrieval one.** Even with every needed
  article in the prompt, the 14B model answers only about half of the multi-hop questions
  correctly (V1: 9 of 18); with the extra passages V2 adds, 5 of 21. More context seems to
  distract the small model more than it helps.
- **The answer check helps a bit.** Grading the first drafts it rewrote (`regrade --drafts`):
  without it, 61% correct and multi-hop 20%; with it, 62% and 25%; citations supported 64% → 67%.
  Small numbers (30 rewritten answers), same caveat as above.

Retrieval-only check of the decomposition step (`python -m rag.eval.context_check_eval`, replays
V1's retrieval, 120 answerable questions; context recall = gold articles anywhere in the prompt):

| Follow-up retrieval | Context recall | Multi-hop recall | Multi-hop complete | Passages in prompt |
|---|---|---|---|---|
| none (V1) | 0.832 | 0.646 | 18/40 | 6 |
| 1 new passage per query | 0.860 | 0.704 | 20/40 | 6.6 |
| **2 new passages per query (V2)** | **0.867** | **0.725** | **20/40** | **7.0** |
| 3 new passages per query | 0.867 | 0.725 | 20/40 | 7.0 |
| 2 per query, rerank pool 20 | 0.867 | 0.725 | 21/40 | 7.0 |

### Retrieval ablation (V1 design)

Retrieval-only eval on the 120 answerable questions (`python -m rag.eval.retrieval_eval`; no
answer generated, so each setup takes seconds to minutes). "Complete" = multi-hop questions with
every needed article in the top 5.

| Retrieval | Recall@5 | MRR | Direct | Multi-hop | Complete |
|---|---|---|---|---|---|
| Vector, bge-m3 (V0) | 0.733 | 0.611 | 0.850 | 0.498 | 8/40 |
| Lexical, Postgres `ts_rank` | 0.083 | 0.046 | 0.087 | 0.075 | 1/40 |
| Lexical, BM25 | 0.597 | 0.502 | 0.675 | 0.442 | 8/40 |
| Hybrid (RRF) | 0.703 | 0.620 | 0.787 | 0.533 | 11/40 |
| Hybrid, BM25 weight 0.3 | 0.726 | 0.643 | 0.838 | 0.504 | 8/40 |
| Hybrid + mMiniLM reranker (118M) | 0.708 | 0.623 | 0.800 | 0.525 | 10/40 |
| **Hybrid + bge-reranker-v2-m3 (568M), top 20 (V1)** | **0.817** | **0.767** | **0.913** | **0.625** | **16/40** |

Postgres `ts_rank` has no notion of how rare a term is, so the words every norm uses
("operadora", "plano") dominated; BM25 over the same stemmed lexemes fixed that.

## Judge validation

The judge's grades only mean something if they match a person's. 50 V0 answers, stratified by
question type and by the judge's verdict, were labelled blind (judge verdict hidden) as correct /
partial / incorrect. Agreement is Cohen's kappa (< 0.4 weak, 0.4–0.6 moderate, 0.6–0.8 substantial).

| Judge rubric | Human labels | Kappa (3 classes) | Kappa (correct vs. not) |
|---|---|---|---|
| v1 | first pass | 0.25 | 0.35 |
| v2 | first pass | 0.38 | 0.49 |
| v1 | after second look | 0.46 | 0.46 |
| **v2** | **after second look** | **0.63** | **0.62** |

- **v1 → v2.** v1 mixed the content of an answer with the quality of its citations (a right
  answer citing the wrong norm was "incorrect") and treated every detail of the reference as
  essential. v2 grades the verdict on content only, keeps citations in their own metric, and calls
  an answer partial only when what is missing changes the practical answer.
- **Second look.** In 6 answers the system contradicted the reference on a concrete fact (a
  percentage, a year, a sanction, a rule stated as its opposite) and the first-pass label had
  been "correct" or "partial". The labeller re-checked them against the quoted norm and changed
  them to "incorrect", as the labelling guide defines it. Because that second look saw the judge's
  argument, the last row is not an independent measurement; the first-pass rows are.
- Files: `eval/judge_labels_initial.json`, `eval/judge_labels.json`, `eval/judge_validation*.json`;
  `python -m rag.eval.validate_judge` recomputes the numbers and lists every disagreement.

## Roadmap

| Version | What it adds | Status |
|---|---|---|
| V0 | Naive RAG: vector search only | ✅ |
| V1 | Hybrid search (vector + BM25) and a cross-encoder reranker | ✅ |
| V2 | Self-corrective loop: context check, query decomposition, answer check and rewrite | ✅ measured: better declining, no gain in correct answers, 2.4× latency |
| V3 | Scope guardrail: decline what the corpus can't answer | ⏳ |
| V4 | Faster and reproducible: embeddings off the GPU so the generator stays loaded, deterministic generation, early exit | ⏳ |

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
validated against human labels (see Judge validation).

## Design decisions

- **Structural chunking.** Chunks follow the legal structure (one article, split at paragraph
  boundaries only when too long) and carry a header with the norm and article number, so a
  retrieved chunk is citable on its own.
- **Postgres for both search legs, BM25 in memory.** pgvector for embeddings; for the lexical leg,
  Postgres' Portuguese stemmer produces the terms and BM25 ranks them in memory (the corpus is a
  few thousand chunks), because Postgres' own ranking has no IDF.
- **Measure retrieval on its own first.** A retrieval-only eval (Recall@5, MRR, seconds per
  setup) chose V1's design before paying for a full eval run with generation and judging.
- **Measure the noise before reading a difference.** The same version run twice moves direct
  questions by 9 points, so every comparison is paired per question and tested, and a re-run of
  an unchanged version sits in the results table. On this setup, local generation is reproducible
  only when the model starts from a fresh load: Ollama reuses the cached start of the previous
  prompt, which changes the floating-point rounding.
- **Negative results stay in the table.** V2 did not pay for its latency with this model; the
  table says so, and the eval shows which of its steps still earn their place.
- **A small local model on purpose.** Every version answers with the same 14B local model, so the
  table shows how much the architecture improves a small model, not how good a frontier model is.
- **Local embeddings.** `bge-m3` is multilingual and runs locally through Ollama: re-indexing
  costs nothing and the eval's cost numbers only reflect the LLM calls.
- **Provider-agnostic LLM layer.** Every call records tokens, latency and estimated cost, so the
  versions can be compared on price as well as quality.

## License

MIT
