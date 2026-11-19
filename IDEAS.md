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

## 2. A completeness rule in the answering prompt (done, no gain in V1)

Measured on 2026-11-18, see the README ("Experiment: a completeness rule in the answering
prompt"). Correct of 120, official prompt -> with the rule: V1 78 -> 79, ceiling 88 -> 94,
gold-only ceiling 88 -> 96. The official prompt stays.

- The rule helps only on questions where the gold passages are in the prompt and V1 does not
  retrieve them (28 questions: ceiling 15 -> 21). Where V1 already has every gold article (92),
  nothing changes.
- With the rule, the distance from V1 to the ceiling is 15 questions, not 10. Retrieval for
  multi-hop questions is worth more than the first ceiling suggested.

Still open: **V2 without the answer check, with the rule.** V2's follow-up searches are the one
place where the system adds missing articles, so it is where the rule could pay. Expected gain is
small: the follow-up searches complete the context for only 3 more multi-hop questions.

## 3. Fine-tuning the answering model

**Status after the reading:** not justified yet. Of the 32 ceiling failures, 10 are a strict
reference or judge, 15 are about completeness (idea 2), 6 are distraction by other passages, and
only 8 are misreadings that call for a better model. Eight questions are too few to measure a
gain on. Idea 2 was tried: it fixed 4 of those 8 at the ceiling and none of it reached V1, which
points back at multi-hop retrieval before any training.

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

## 5. A generator that fits on the GPU (done)

Measured on 2026-11-19 with `qwen2.5:7b`, see the README ("A smaller answering model"). Correct
of 120, 14B -> 7B on the same passages: V0 69 -> 55, V1 78 -> 58, ceiling 88 -> 75, gold-only
ceiling 88 -> 82. V2, each model running its own loop: 79 -> 58. V4 end to end: 25 s -> 3.7 s
per question.

- The smaller model is close to the larger one when the prompt holds only the right passages
  and falls behind when other passages are around. Its weak spot is what to ignore.
- Fewer passages by the reranker's score was tried the same day (README, "Fewer passages for
  the 7B") and gave nothing: 58 of 120 with all six, 55 to 61 with four different cuts, none
  separable from chance. The score does not tell the passage that answers from its
  neighbours, so every cut that removes distraction also removes gold articles.
- What is left for the 7B: a filter that reads the passages (one cheap model call per passage,
  "does this answer the question?"; the 7B is fast enough to afford it, cost not measured),
  or the retrieval work for two-article questions, which helps both models.
- Fine-tuning (idea 3) has a better case with the 7B than with the 14B: it fits in 8 GB for
  training and there is more to gain (58 of 120 correct in V1). Its failures have not been read
  one by one yet; that comes first, as it did for the 14B.
