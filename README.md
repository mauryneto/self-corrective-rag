# self-corrective-rag

A self-corrective RAG over Brazilian health-insurance regulation (normative acts of ANS, the
national regulator). An LLM judge checks every answer; failed answers trigger query rewriting and
decomposition; questions the corpus can't answer are declined. Every design choice is backed by an
ablation on a fixed eval set, reporting accuracy, refusal precision, latency and cost per query.

> **Status: work in progress (week 3 of 4).** Corpus pipeline with incremental updates, a
> 180-question eval set, a judge validated against human labels, and V0 (baseline), V1 (hybrid
> search + reranker), V2 (self-corrective loop) and V3 (scope gate) measured in a reproducible setup: the same
> version run twice gives the same answers, so a difference between rows comes from the version.
> Two reference rows (no retrieval, perfect retrieval) bound what the architecture can add.
> Speed (V4) comes next.

## Results

Measured on the 180-question eval set (`eval/runs/*_repro.*`). Each new version gets a row.

| Version | Correct | Correct + partial | Multi-hop correct | Recall@5 | MRR | Declines when it should | Refusal precision | Citations supported | p50 / p95 latency | Cost / question |
|---|---|---|---|---|---|---|---|---|---|---|
| *Floor: no retrieval* | 12% | 32% | 0% | n/a | n/a | 43% | 84% | n/a | n/a | US$ 0 (local) |
| V0 | 58% | 82% | 23% | 73% | 0.61 | 98% | 92% | 69% | 20s / 38s | US$ 0 (local) |
| V1 | 65% | 83% | 38% | 82% | 0.77 | 93% | 93% | 68% | 42s / 68s | US$ 0 (local) |
| V2 | 66% | 78% | 35% | 82% | 0.77 | 97% | 88% | 68% | 55s / 140s | US$ 0 (local) |
| V2 without the answer check | 68% | 83% | 43% | 82% | 0.77 | 92% | 95% | 68% | n/a | US$ 0 (local) |
| V3 | 65% | 83% | 38% | 82% | 0.77 | 93% | 93% | 68% | 41s / 68s | US$ 0 (local) |
| *Ceiling: perfect retrieval* | 73% | 88% | 50% | 100% | 1.00 | 93% | 100% | 71% | n/a | US$ 0 (local) |
| *Ceiling, gold passages only* | 73% | 91% | 48% | 100% | 1.00 | n/a | n/a | 88% | n/a | US$ 0 (local) |

"Correct", "Multi-hop correct", Recall@5 and MRR are over the 120 answerable questions; the two
refusal columns over the 60 it should decline. Every version answers with the same prompt and
`qwen2.5:14b` on a laptop GPU (RTX 5070, 8 GB), hence the latency and zero cost; only what the
row's version changes differs. Judge: Claude Haiku 5.5, rubric v2 (validated below). "V2 without
the answer check" is the V2 run graded on the drafts written before the answer check
(`regrade --drafts`). V3 is V1 plus a gate before the model, so only the 24 questions the gate
declines needed a run; the others have V1's answers (`--reuse-from v1_repro`, see V3 below).

The rows in italics are references, not versions (`rag.eval.reference`): the same model with no
passages at all, and with the gold passages handed to it (first in the prompt, followed by V1's
other passages up to the same six; or alone). They bound what any retrieval can add.

**How much of the possible gain each version delivers.** From floor to ceiling there are 74
questions to gain (14 → 88 of 120):

| Version | Correct (of 120) | Share of the floor-to-ceiling gain |
|---|---|---|
| Floor | 14 | 0% |
| V0 | 69 | 74% |
| V1 | 78 | 86% |
| V2 | 79 | 88% |
| V2 without the answer check | 82 | 92% |
| Ceiling | 88 | 100% |

Plain vector search already delivers three quarters of what retrieval can give this model; the
reranker and the follow-up searches most of the rest. What is left above the ceiling, 32
questions of 120, no retrieval can reach: it belongs to the model.

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
| V1 → V3 | 78 → 78 | 0 / 0 | 1.00 |
| Floor → V0 | 14 → 69 | 58 / 3 | < 0.001 |
| V1 → ceiling | 78 → 88 | 15 / 5 | 0.04 |
| Ceiling → ceiling, gold passages only | 88 → 88 | 11 / 11 | 1.00 |

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

**Floor and ceiling** (the same model with no passages, and with the gold passages):
- **Without retrieval the model knows little: 14 of 120, and none of the 40 multi-hop
  questions.** It also answers 34 of the 60 questions it should decline. Almost everything in
  the table is the architecture's doing, not the model's memory.
- **Perfect retrieval gives 88 of 120, not 120.** Direct questions reach 68 of 80; multi-hop
  only 20 of 40. Half of the multi-hop questions fail with every needed article in the prompt.
- **Retrieval has 10 questions left to give** (V1 → ceiling, p = 0.04), 6 over V2 without the
  answer check. The model has 32. The larger remaining gain is in what the model does with the
  passages, not in finding them.
- **The other passages in the prompt do not lower the score, but they change the answers.**
  With only the gold passages the total is the same 88, yet 22 questions flip (11 each way):
  the model is right in both setups for 77 questions and in at least one for 99. Its answer
  depends on what surrounds the right passage. Citations are the clear difference: 88%
  supported with gold passages only, 71% with the others around.
- **Read one by one, about 22 of those 32 are the model's.** In 10 the practical answer is
  there and the reference or the judge is strict. Of the rest, 8 leave out a condition or
  exception that is in the passage, 6 answer from a neighbouring passage, and 8 misread the
  right one. The answering prompt asks for a direct, short answer and never for exceptions,
  which fits the largest pattern. The full reading is in
  [eval/ceiling_failures.md](eval/ceiling_failures.md); it is one reader's pass and changes no
  number here.
- **A check of the setup:** in the ceiling run, 122 questions got exactly V1's passages (the 60
  without gold sources, and 62 where V1 already had the gold passages first). All 122 answers
  are identical to V1's.

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
  the prompt, the model answers some questions it should decline. V3's gate does not recover
  them: their passages look relevant (see V3).
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
  120 answerable ones. V3 tested it as a rule for declining and dropped it (see V3).
- **Decomposition finds little of what is missing.** The model flags 13 of the 22 multi-hop
  questions whose context is incomplete and writes sensible follow-up queries, but those
  searches complete only 3 more of them (18 → 21 of 40; context recall 83% → 88%).
- **Multi-hop is also a reasoning problem, not only a retrieval one.** With every needed
  article in the prompt, the 14B model still answers only about half of the multi-hop questions
  correctly: 10 of 18 in V1, 11 of 21 in V2 before the answer check, 20 of 40 at the ceiling.
  Better retrieval cannot close that half.

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

**V3** (V1 plus a scope gate: when the best reranker score among the retrieved passages is below
0.05, the system declines with a fixed message and does not call the model):
- **No quality number changes: 78 of 120 correct, 56 of 60 declined, as in V1.** The gate
  declines 24 questions (18 of the 20 out-of-scope ones, 6 of the 40 unanswerable ones). The
  model already declined all 24 in V1, and no answerable question is below the gate.
- **What it buys is time and certainty.** Those 24 take a median 9.6 s (retrieval only)
  instead of 31 s, with no LLM call: 156 calls instead of 180 over the eval set, mean latency
  43.9 s → 40.7 s. The median barely moves (42 s → 41 s) because it sits among the answerable
  questions. And these refusals no longer depend on the model obeying the prompt: with no
  passages at all, the same model answers 34 of the 60 (the floor row).
- **V2's context check is not usable as the guardrail**, though the roadmap planned to start
  from it. "Decline when the check says the passages don't cover the question" declines 58 of
  60 and also 44 answerable questions, 25 of them answered correctly (82 → 57 correct).
  Restricted to questions whose best score is below 0.5, it reaches 58 of 60 at the price of 2
  correct answers and 5 more wrongful refusals: the trade V2's answer check already makes. So
  the gate sits on V1, and V2's loop is not part of V3.
- **The four questions that should be declined and are answered stay out of reach.** Two
  retrieve passages that look like the answer (best score above 0.95, e.g. a 300-day grace
  period stated for another situation) and pass V2's context check too. The other two score
  0.30, and a threshold that catches them also declines 7 answerable questions, 4 of them
  answered correctly. Unanswerable questions about health plans look like answerable ones to
  the reranker (12 of 40 score above 0.5); there the prompt does the declining, 37 of 40.
- **The threshold was chosen on the eval set.** 0.05 is half the lowest best score among the
  120 answerable questions (0.11), rounded down. A check of that rule: picked on four folds, it
  declined no answerable question of the fifth, in each of the five folds.
- **Only what V3 changes was run.** Past the gate V3 does exactly what V1 does and generation
  is deterministic, so the run took 4 minutes instead of 2 hours: the 24 gated questions
  ran, 2 more as a check, and the other 154 keep V1's stored answer, verdict and latency. The
  26 that ran retrieved exactly V1's passages, and 3 answers generated again past the gate
  came out identical to V1's.
- **Left for V4:** a declined question still pays for the reranker (about 10 s on CPU).

Threshold curve on the V1 run (`python -m rag.eval.scope_gate_eval`, no model call):

| Threshold | Out-of-scope declined (of 20) | Unanswerable declined (of 40) | Answerable declined (correct in V1) | Declines (of 60) | Correct (of 120) |
|---|---|---|---|---|---|
| 0.02 | 17 | 2 | 0 | 56 | 78 |
| **0.05 (V3)** | **18** | **6** | **0** | **56** | **78** |
| 0.10 | 19 | 12 | 0 | 56 | 78 |
| 0.20 | 19 | 15 | 3 (2) | 56 | 76 |
| 0.35 | 20 | 21 | 7 (4) | 58 | 74 |
| 0.50 | 20 | 28 | 12 (5) | 58 | 73 |

### Experiment: a completeness rule in the answering prompt

The reading of the ceiling failures pointed at answers that leave out a condition or exception
the passages bring. One general rule was added to the answering prompt ("give the whole rule:
the conditions, exceptions and deadlines the cited passages attach to it, and answer every part
of the question") and three rows were run again (`--answer-prompt complete`; runs `v1_complete`,
`ceiling_complete`, `ceiling_gold_complete`). Correct answers of 120:

| Row | Official prompt | With the rule | Gained / lost | McNemar p |
|---|---|---|---|---|
| V1 | 78 | 79 | 10 / 9 | 1.0 |
| Ceiling | 88 | 94 | 15 / 9 | 0.31 |
| Ceiling, gold passages only | 88 | 96 | 12 / 4 | 0.08 |

- **No gain where it counts.** V1 moves by one question, with 19 questions changing sides. The
  official prompt stays, and the rows above are not in the results table.
- **The rule helps only when the passages are there.** On the 92 questions where V1 retrieves
  every gold article, nothing changes (V1 72 → 73, ceiling 73 → 73). The whole gain at the
  ceiling comes from the 28 where V1 misses one (ceiling 15 → 21, V1 6 → 6): the model can
  only add a condition from an article it was given.
- **So retrieval has more left than the first ceiling said.** With the rule, V1 → ceiling is
  15 questions (19 gained, 4 lost, p = 0.003) instead of 10, and multi-hop at the ceiling goes
  from 20 to 26 of 40 while V1's goes from 15 to 14.
- **It costs length and, in V1, citations.** Answers grow from a median of 85 to 109 words;
  supported citations go from 68% to 65% in V1 (71% → 77% at the ceiling).
- **Of the 32 ceiling failures, 15 become correct** (6 of the 10 strict-reference ones, 4 of 8
  omissions, 4 of 8 misreadings, 1 of 6 distractions) **and 9 other questions are lost** (5
  direct, 4 multi-hop; 6 of them now incorrect, 3 partial).
- **Caveat:** the rule was written after reading failures on these same questions; there is no
  held-out set. A gain would have needed that caveat. The absence of one in V1 does not.
- **Same setup as the official runs:** 8 questions re-run with the official prompt came out
  identical to the stored answers, and the 62 answerable questions with the same passages in V1
  and in the ceiling got identical answers under the new prompt.

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
| V3 | Scope gate: decline before calling the model when no retrieved passage is about the question | ✅ measured: same answers as V1; out-of-scope questions are declined in 10 s instead of 31 s, with no LLM call |
| V4 | Faster: where each model runs (generator, embeddings, reranker), skip the checks when they are not needed | ⏳ |

Ideas beyond the roadmap (V2 with the completeness rule, fine-tuning the
answering model) are in [IDEAS.md](IDEAS.md).

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
- **Run only what a change can affect.** With deterministic generation, a version that differs
  from another on 24 questions is measured on those 24; the rest is reused and marked as such.
- **A small local model on purpose.** Every version answers with the same 14B local model, so the
  table shows how much the architecture improves a small model, not how good a frontier model is.
- **Bound the question before answering it.** A floor (no retrieval) and a ceiling (gold
  passages) turn "V1 is better than V0" into "V1 delivers 86% of what retrieval can give this
  model", and show when the next gain has to come from somewhere else.
- **Local embeddings.** `bge-m3` is multilingual and runs locally through Ollama: re-indexing
  costs nothing and the eval's cost numbers only reflect the LLM calls.
- **Provider-agnostic LLM layer.** Every call records tokens, latency and estimated cost, so the
  versions can be compared on price as well as quality.

## License

MIT
