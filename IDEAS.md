# Ideas for later

Not commitments. Things worth trying once the current roadmap step is measured. Each one should
enter the results table as its own row, measured on the same 180 questions with the same judge.

## 1. Reference rows: floor and ceiling

The table says V1 beats V0, but not how much of the possible gain that is.

- **Floor:** the local model alone, no retrieval at all.
- **Architecture ceiling:** the local model given the gold articles directly, as if retrieval were
  perfect.
- **Optional, model ceiling:** V1 with a large model answering, to show what the small model costs.

With floor and ceiling, each version can be read as "share of the possible gain delivered".
Early hint that the ceiling is low: on multi-hop questions where the context was complete, V1 got
only 9 of 18 right. That points at the model's reasoning, not at retrieval.

Cost: about 1 hour each, local.

## 2. Fine-tuning the answering model

The project question is "how much does architecture improve a small model". Fine-tuning is the
other lever: improve the model itself and keep the architecture fixed. One row, "V1 + fine-tuned
model", would contrast the two.

- **What it could improve:** behaviour, not knowledge. Combining two articles, citing only what
  the passage says, declining when the passages do not answer. The norms themselves stay in the
  database, because they change and a trained model would go stale.
- **Training data:** a few hundred to a few thousand examples (question, passages, ideal answer),
  written by a large model and spot-checked by hand. They must not overlap the 180 evaluation
  questions, or the score is contaminated.
- **Hardware:** qwen2.5:14b does not fit for training in 8 GB of VRAM. Either rent a GPU for a
  few hours or fine-tune a smaller model (7b or 3b) locally.
- **When it is worth it:** decide after idea 1. If the ceiling row shows the model failing even
  with the right articles in hand, architecture cannot close that gap and fine-tuning (or a
  larger model) is the only lever left.

## 3. Fine-tuning retrieval (embeddings or reranker)

Cheaper than idea 2 and fits in 8 GB, but retrieval is not the current bottleneck (context recall
is already around 0.82 to 0.88). Low priority unless the ceiling row says otherwise.
