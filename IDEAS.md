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

## 2. Fine-tuning the answering model

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

## 3. Fine-tuning retrieval (embeddings or reranker)

Cheaper than idea 2 and fits in 8 GB, but the ceiling says retrieval has little left to give
(about 10 questions over V1, 6 over V2 without the answer check). Low priority.
