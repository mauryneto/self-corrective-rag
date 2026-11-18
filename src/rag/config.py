from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@127.0.0.1:5433/rag"

    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024

    llm_provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_llm_model: str = "qwen2.5:14b"
    # Layers of the local model placed on the GPU. Left unset, Ollama picks a number from the VRAM
    # free at load time, which changes the arithmetic and therefore the answers between runs.
    # Pinning it (and keeping the embedding model loaded alongside) makes runs comparable.
    ollama_num_gpu: int | None = None
    ollama_num_ctx: int = 16384
    # Reset Ollama's prompt cache before every call so the same prompt always gives the same
    # answer (see rag.llm.OllamaLLM._reset_cache). Off only to save ~0.2 s per call.
    ollama_reproducible: bool = True
    anthropic_model: str = "claude-opus-5-5"

    top_k: int = 6

    data_dir: Path = ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"


@lru_cache
def get_settings() -> Settings:
    return Settings()
