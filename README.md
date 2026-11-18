# self-corrective-rag

A self-corrective RAG over Brazilian health-insurance regulation (normative acts of ANS, the
national regulator). An LLM judge checks every answer; failed answers trigger query rewriting and
decomposition; questions the corpus can't answer are declined. Every design choice is backed by an
ablation on a fixed eval set, reporting accuracy, refusal precision, latency and cost per query.

> **Status: work in progress (week 3 of 4).** Corpus pipeline with incremental updates, a
> 180-question eval set, a judge validated against human labels, and V0 (baseline), V1 (hybrid
> search + reranker) and V2 (self-corrective loop) measured in a reproducible setup: the same
> version run twice gives the same answers, so a difference between rows comes from the version.
> The guardrail comes next.

## Results

Measured on the 180-question eval set (`eval/runs/*_repro.*`). Each new version gets a row.

| Version | Correct | Correct + partial | Multi-hop correct | Recall@5 | MRR | Declines when it should | Refusal precision | Citations supported | p50 / p95 latency | Cost / question |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 58% | 82% | 23% | 73% | 0.61 | 98% | 92% | 69% | 20s / 38s | US$ 0 (local) |
| V1 | 65% | 83% | 38% | 82% | 0.77 | 93% | 93% | 68% | 42s / 68s | US$ 0 (local) |
| V2 | 66% | 78% | 35% | 82% | 0.77 | 97% | 88% | 68% | 55s / 140s | US$ 0 (local) |
| V2 without the answer check | 68% | 83% | 43% | 82% | 0.77 | 92% | 95% | 68% | n/a | US$ 0 (local) |

"Correct", "Multi-hop correct", Recall@5 and MRR are over the 120 answerable questions; the two
refusal columns over the 60 it should decline. Every version answers with the same prompt and
`qwen2.5:14b` on a laptop GPU (RTX 5070, 8 GB), hence the latency and zero cost; only what the
row's version changes differs. Judge: Claude Haiku 5.5, rubric v2 (validated below). The last row
is the V2 run graded on the drafts written before the answer check (`regrade --drafts`).

**Compare versions question by question.** With 120 answerable questions, versions a few points
apart can differ by chance in which questions they happen to get right.
`python -m rag.eval.compare RUN_A RUN_B` does the paired comparison (questions gained and lost,
McNemar's exact test):

| Comparison | Correct (of 120) | Gained / lost | p |
|---|---|---|---|
| V0 → V1 | 69 → 78 | 18 / 9 | 0.12 |
| V1 → V2 | 78 → 79 | 8 / 7 | 1.00 |
| V1 → V2 without the answer check | 78 → 82 | 8 / 4 | 0.39 |
| V0 → V2 without the answer check | 69 → 82 | 20 / 7 | 0.02 |
| Answer check: without → with | 82 → 79 | 1 / 4 | 0.38 |

Retrieval metrics (Recall@5, MRR) are exact, so V1's retrieval gain is solid. In correct answers,
each single step is in the right direction but not significant on its own; only the two steps
together (V0 → V2 without the answer check, +13 questions) clear the test, and that is one of
five comparisons, not a planned one.

### Reproducibility

The first version of this table was not comparable row to row, and said so wrongly. A second run
of V1, same code and settings, gave 39 identical answers out of 180 and 72 correct instead of 76;
it was reported here as "pure noise". Two causes, neither of them random:

- **Where the model ran.** Ollama decides how many layers go on the GPU from the VRAM free when
  the model loads. A crashed runner process was still holding VRAM, so the second V1 run and the
  first V2 run had 0 of 49 layers on the GPU, while V0 and V1 had about 21. Different hardware
  split, different floating-point rounding, different answers.
- **The prompt cache.** Ollama reuses the cached start of the previous prompt, and how much it
  reuses changes how the rest is batched, which again changes the rounding.

What the eval does now (`rag.llm.OllamaLLM`, `rag.eval.judge.JudgeCache`):

| Source of variation | Fix |
|---|---|
| GPU / CPU split chosen at load time | `OLLAMA_NUM_GPU` pins the layers on the GPU (16 keeps the generator and the embedding model loaded together in 8 GB) |
| Prompt cache carried over from the previous call | a one-token request before every call resets it, so every prompt starts from the same state |
| The judge grading the same answer differently | verdicts are cached by rubric, judge model and full prompt; an unchanged answer keeps its verdict |

Checks that it holds: two runs of the same version gave identical answers and traces (12 of 12
for V1, 6 of 6 for V2), and those answers came out identical again in the full runs hours later.
For the 90 questions where V2's context check added no passage, V2's draft is byte-for-byte V1's
answer, from a separate process. The earlier runs stay in `eval/runs/` (`v0_baseline`, `v1`,
`v1_rerun`, `v2`) but are not in the table: 67, 76, 72 and 75 correct of 120.

### What each version shows

**V0** (vector search only):
- **Multi-hop is the weak spot: 9 of 40 correct, against 60 of 80 on direct questions.**
  Retrieval finds every needed article for only 8 of the 40 multi-hop questions (Recall@5 50%,
  against 85% on direct ones): one vector search for a question that needs two norms brings back
  one of them. This is what V1 (hybrid search and reranking) and V2 (query decomposition and a
  second retrieval pass) are for.
- **Declining already works: 98%** of unanswerable and out-of-scope questions, with 4% of
  answerable questions wrongly declined.
- **Citations: 69% supported.** The local model sometimes fills gaps from memory and attributes
  rules to revoked norms (e.g. RN 259 instead of RN 566).

**V1** (hybrid search + cross-encoder reranker; same prompt and model as V0):
- **Retrieval clearly improves: Recall@5 73% → 82%, MRR 0.61 → 0.77.** Correct answers go
  69 → 78 of 120 (multi-hop 9 → 15 of 40), in the expected direction but not significant on
  their own (p = 0.12).
- **The reranker did the work, not the hybrid fusion.** In the retrieval-only ablation below,
  fusing BM25 with the vectors did not beat vector search on this corpus (bge-m3 is already
  strong and BM25 adds noise); the 568M multilingual cross-encoder is what moved recall.
- **A trade-off appeared in declining: 59 → 56 of 60.** With more relevant-looking passages in
  the prompt, the model answers some questions it should decline. That is the job of V3's
  guardrail.
- **It costs latency:** the reranker runs on CPU (median 14 s per question), p50 20 s → 42 s.
  Cutting that is part of V4.
- **Multi-hop is still the weak spot** (15 of 40): even with a better ranking, a single search
  for a question that needs two norms finds every needed article for only 16 of 40. That is V2's
  job: decompose the question and search again.

**V2** (V1 plus a self-corrective loop, every step run by the same 14B model: check whether the
passages cover the question and, if not, search for what is missing; check the answer against
the passages it cites and rewrite it once if a claim is unsupported):
- **No measurable gain in correct answers over V1** (79 against 78 of 120), at 1.3× the median
  latency and 2× the p95: two to four LLM calls per question instead of one.
- **The answer check trades correct answers for refusals.** It rewrote 31 answers. Three wrong
  answers to questions the corpus cannot answer became correct refusals (55 → 58 of 60 declined),
  but four correct answers were lost and one gained (82 → 79 of 120), and wrongful refusals went
  from 3 to 8 of 120. It makes the system more cautious, not more accurate, and an earlier run
  that suggested a small gain did not replicate.
- **The context check is a strong but noisy signal for declining.** It flags 58 of the 60
  questions the system should decline as "the passages don't cover this", and also 44 of the
  120 answerable ones. V3's guardrail starts from this signal and has to fix its precision.
- **Decomposition finds little of what is missing.** The model flags 13 of the 22 multi-hop
  questions whose context is incomplete and writes sensible follow-up queries, but those
  searches complete only 3 more of them (18 → 21 of 40; context recall 83% → 88%).
- **Multi-hop is also a reasoning problem, not only a retrieval one.** With every needed
  article in the prompt, the 14B model still answers only about half of the multi-hop questions
  correctly: 10 of 18 in V1, 11 of 21 in V2 before the answer check. Better retrieval cannot
  close that half; see [IDEAS.md](IDEAS.md) for the reference rows that would measure it.

Retrieval-only check of the decomposition step (`python -m rag.eval.context_check_eval`, replays
V1's retrieval, 120 answerable questions; context recall = gold articles anywhere in the prompt;
measured before the reproducible setup, so the follow-up queries differ slightly from the V2 row
above):

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
| V2 | Self-corrective loop: context check, query decomposition, answer check and rewrite | ✅ measured: no gain in correct answers over V1; the answer check trades accuracy for refusals |
| V3 | Scope guardrail: decline what the corpus can't answer, starting from V2's context check | ⏳ |
| V4 | Faster: where each model runs (generator, embeddings, reranker), skip the checks when they are not needed | ⏳ |

Ideas beyond the roadmap (floor and ceiling reference rows, fine-tuning) are in [IDEAS.md](IDEAS.md).

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

`OLLAMA_NUM_GPU` in `.env` pins how many layers of the local model run on the GPU. The default
in `.env.example` (16) fits 8 GB of VRAM; adjust it to your card, and keep it fixed between the
runs you want to compare.

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
- **Make the measurement repeatable before reading a difference.** The same version run twice
  used to disagree on most answers. The eval now pins the GPU split, resets the prompt cache
  before every call and caches judge verdicts (see Reproducibility), and every comparison is
  paired per question and tested.
- **Negative results and corrections stay in the README.** V2 did not pay for its latency with
  this model, and the first results table blamed on noise what was a hardware difference; both
  are written down, with the eval showing which of V2's steps still earn their place.
- **A small local model on purpose.** Every version answers with the same 14B local model, so the
  table shows how much the architecture improves a small model, not how good a frontier model is.
- **Local embeddings.** `bge-m3` is multilingual and runs locally through Ollama: re-indexing
  costs nothing and the eval's cost numbers only reflect the LLM calls.
- **Provider-agnostic LLM layer.** Every call records tokens, latency and estimated cost, so the
  versions can be compared on price as well as quality.

## License

MIT
