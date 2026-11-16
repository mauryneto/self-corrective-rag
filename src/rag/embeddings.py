"""Text embeddings through a local Ollama server (bge-m3 by default)."""

from __future__ import annotations

import httpx

from rag.config import get_settings


class OllamaEmbedder:
    def __init__(self, model: str | None = None, url: str | None = None, batch_size: int = 32):
        s = get_settings()
        self.model = model or s.embed_model
        self.url = (url or s.ollama_url).rstrip("/")
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=300)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.post(f"{self.url}/api/embed", json={"model": self.model, "input": batch})
            resp.raise_for_status()
            vectors.extend(resp.json()["embeddings"])
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
