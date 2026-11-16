# Brief: writing eval questions for the ANS RAG

Project: `C:\Users\maury\Desktop\maury\self-corrective-rag` — a RAG that answers questions about
Brazilian health-insurance regulation (Resoluções Normativas, "RNs", of the ANS). We are building
the eval set that will score every version of the pipeline. Quality matters more than speed: a
wrong reference answer poisons every future measurement.

## The corpus
- `data/corpus/<doc_id>.json` (e.g. `rn-566.json`): one norm each, with `ementa` (summary) and
  `paragraphs` (the consolidated text in force today, in order). Read these files directly.
- Articles start with "Art. 5º", "Art. 12.", "Art. 4º-A". The pipeline chunks by article, so a
  gold source is a (norm, unit) pair where unit is written exactly like `Art. 5` or `Art. 4-A`
  (no "º", no trailing dot). Annex text is unit `Anexo`.

## Output: one JSON object per line (JSONL), UTF-8
```json
{"id": "a001", "type": "direct", "question": "...", "reference_answer": "...",
 "gold": [{"doc": "rn-566", "unit": "Art. 3", "evidence": "<verbatim excerpt>"}],
 "absence_terms": [], "status": "draft", "notes": ""}
```
- `type`: `direct` (one article answers), `multi_hop` (needs 2+ articles, ideally from different
  norms; the answer must combine them), `unanswerable` (sounds like ANS regulation but the corpus
  does not answer it), `out_of_scope` (not ANS regulation at all).
- `question`: natural Brazilian Portuguese, phrased the way a beneficiary, a broker or an
  operator's analyst would ask — not "o que diz o art. 5º da RN 566". Vary the wording and topic.
- `reference_answer`: ONLY the essential points that answer the question, in Portuguese, ending
  with the source in parentheses, e.g. "Até 10 dias úteis (RN 566, art. 3º, V)." No extra
  context the question did not ask for — the judge penalizes answers that miss anything in the
  reference, so every point in it must be essential.
- `gold[].evidence`: copy a short excerpt (one sentence or clause, 8–40 words) **character for
  character** from that article's paragraphs. It must support the reference answer. A checker
  verifies it is literally in the article (it ignores case, accents, quotes and extra spaces, but
  nothing else — do not paraphrase, do not join non-adjacent text, do not fix typos).
- Unanswerable: `gold` must be empty; `reference_answer` says the norms do not contain the
  information and the system should say so; put 2–4 `absence_terms` (distinctive phrases you
  searched for). Out of scope: `gold` empty, `reference_answer` says it is outside ANS regulation.

## Rules
- Prefer norms and articles that matter to people dealing with health plans (coverage, deadlines,
  network, contracts, portability, reajustes rules, cancellation, carência, urgência, ouvidoria,
  information duties, penalties, operator obligations). Skip purely internal/administrative or
  accounting-only norms unless your group has little else.
- No two questions on the same fact. Do not repeat these existing questions: psicólogo deadline;
  laboratory exam deadline; no network provider in the municipality; deadline for non-assistance
  requests; "em análise" answers; experimental treatment; internação eletiva response+deadline;
  lab exam response deadline vs RN 566; 2026 reajuste percentage; Imposto de Renda.
- Do NOT ask whether a specific procedure/exam/drug is covered by the Rol (the Rol tables are
  annexes not in the corpus) and do not rely on annex tables.
- Every fact in a reference answer must come from the corpus text you read, never from memory.
- Check your work: run
  `C:\Users\maury\Desktop\maury\self-corrective-rag\.venv\Scripts\python.exe -m rag.eval.check_questions --file <your file>`
  from the project folder and fix every ERROR until it reports 0 errors. Unanswerable WARNs mean
  a term appears in the corpus: read those hits and drop or rewrite the question if the corpus
  actually answers it.
- Write only your own output file. Do not edit any other file in the project.
