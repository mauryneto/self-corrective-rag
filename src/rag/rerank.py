"""Cross-encoder reranker (multilingual), run locally.

A cross-encoder reads the question and one candidate passage together and scores how well the
passage answers it: slower than comparing embeddings, much more precise. It rescoring a pool of
~30 candidates from the hybrid search and keeping the best few is the classic second stage.

Where it runs is explicit. V1 to V3 were measured with it on the CPU in single precision, and that
stays the default so those rows keep reproducing. ``fast_reranker`` is V4's: on the GPU in half
precision when there is one (about 0.3 s for 20 passages instead of 8 s).
"""

from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_RERANKER = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


@lru_cache(maxsize=4)
def _load(model_name: str, device: str, half: bool):
    from sentence_transformers import CrossEncoder

    kwargs = {}
    if half:
        import torch

        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return CrossEncoder(model_name, max_length=512, device=device, **kwargs)


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str = "cpu", half: bool = False,
                 batch_size: int = 16):
        self.model_name = model_name
        self.device, self.half = device, half
        self.model = _load(model_name, device, half)
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


def fast_reranker(model_name: str = DEFAULT_RERANKER) -> CrossEncoderReranker:
    """The reranker on the GPU in half precision; on a machine without one, the CPU reranker."""
    import torch

    if torch.cuda.is_available():
        return CrossEncoderReranker(model_name, device="cuda", half=True, batch_size=32)
    return CrossEncoderReranker(model_name)
