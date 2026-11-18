# Ideas for later

Not commitments. Things worth trying once the current roadmap step is measured. Each one should
enter the results table as its own row, measured on the same 180 questions with the same judge.

## 1. Reference rows: floor and ceiling (done)

Measured on 2026-11-18, see the README. Correct answers of 120: floor 14, V0 69, V1 78, V2 79,
ceiling 88. What it settled:

- Retrieval has about 10 questions left to give over V1. The model leaves 32 on the table even
  with the gold passages in the prompt, 20 of them multi-hop.
- Extra passages do not lower the score but change which questions are right (22 flips between
  the ceiling and the gold-only run), and they hurt citations (88% supported with gold passages
  only, 71% with others around).

Still open: the **model ceiling**, V1 with a large model answering, to show what the small model
costs. It is the only row here that costs money (API) or plan usage.

## 2. A completeness instruction in the answering prompt

From reading the 32 ceiling failures (`eval/ceiling_failures.md`, 2026-11-18): in 15 of them the
missing piece is a condition, an exception or a second item that sits in a passage the model was
given. The prompt asks for a direct answer in up to three short paragraphs and never asks for
exceptions.

- **The experiment:** one general line in `ANSWER_SYSTEM` (state the conditions, exceptions and
  deadlines the cited passage attaches to the rule), then V1 and the two ceiling rows again.
  A few hours of local runs, no cost.
- **What could go wrong:** longer answers leave more room for the wrong-reading kind of error (8
  of the 32), and for citing more passages that do not support the claim.
- **Caveat to report with the result:** the instruction comes from reading failures on the same
  180 questions it is measured on. There is no held-out set. Keep it general, never tied to a
  norm or a question.

## 3. Fine-tuning the answering model

**Status after the reading:** not justified yet. Of the 32 ceiling failures, 10 are a strict
reference or judge, 15 are about completeness (idea 2), 6 are distraction by other passages, and
only 8 are misreadings that call for a better model. Eight questions are too few to measure a
gain on. Revisit if idea 2 is tried and the misreadings remain the main group.

The project question is "how much does architecture improve a small model". Fine-tuning is the
other lever: improve the model itself and keep the architecture fixed. One row, "V1 + fine-tuned
model", would contrast the two.

- **Why it now looks worth trying:** the ceiling says the larger remaining gain is in the model
  (32 questions) and not in retrieval (10). The failures are of the kind training can address:
  combining two articles, staying with what the passage says, not being swayed by the passages
  around the right one.
- **What it could improve:** behaviour, not knowledge. The norms themselves stay in the
  database, because they change and a trained model would go stale.
- **Training data:** a few hundred to a few thousand examples (question, passages, ideal answer),
  written by a large model and spot-checked by hand. They must not overlap the 180 evaluation
  questions, or the score is contaminated.
- **Hardware:** qwen2.5:14b does not fit for training in 8 GB of VRAM. Either rent a GPU for a
  few hours or fine-tune a smaller model (7b or 3b) locally.
- **Cheaper things to try first, same target:** a prompt written for multi-hop questions (answer
  each part, then combine), and the gold-only ceiling re-run with that prompt. If a prompt
  recovers most of the 32, training is not needed.
- **Caveat on the 32:** 18 of them are "partial" (right, but missing something that changes the
  practical answer) and 15 questions are never answered correctly in any setup. Some of that may
  be the reference answer or the judge being strict; worth reading before training anything.

## 4. Fine-tuning retrieval (embeddings or reranker)

Cheaper than idea 3 and fits in 8 GB, but the ceiling says retrieval has little left to give
(about 10 questions over V1, 6 over V2 without the answer check). Low priority.
