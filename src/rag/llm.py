"""Provider-agnostic LLM calls: local models through Ollama, or Claude through the Anthropic API.

Every call returns the text plus a ``CallRecord`` (tokens, latency, model), which the pipeline
aggregates into the per-question cost and latency reported by the eval.
"""

from __future__ import annotations

import time
from typing import Protocol

import httpx

from rag.config import get_settings
from rag.costs import CallRecord
from rag.retry import with_retries


class LLMRefusal(RuntimeError):
    """The model declined to answer (Claude ``stop_reason == "refusal"``)."""


class LLM(Protocol):
    model: str

    def complete(self, system: str, prompt: str, step: str, max_tokens: int = 2048) -> tuple[str, CallRecord]: ...


class OllamaLLM:
    def __init__(self, model: str | None = None, url: str | None = None):
        s = get_settings()
        self.model = model or s.ollama_llm_model
        self.url = (url or s.ollama_url).rstrip("/")
        self.client = httpx.Client(timeout=600)

    def complete(self, system: str, prompt: str, step: str, max_tokens: int = 2048) -> tuple[str, CallRecord]:
        t0 = time.perf_counter()

        def call() -> dict:
            resp = self.client.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16384},
                },
            )
            resp.raise_for_status()
            return resp.json()

        data = with_retries(call)
        record = CallRecord(
            step=step,
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_s=time.perf_counter() - t0,
        )
        return data["message"]["content"], record


class AnthropicLLM:
    def __init__(self, model: str | None = None, effort: str = "medium"):
        import anthropic

        self.model = model or get_settings().anthropic_model
        self.effort = effort
        self.client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or an `ant auth` profile

    def complete(self, system: str, prompt: str, step: str, max_tokens: int = 16000) -> tuple[str, CallRecord]:
        t0 = time.perf_counter()
        # Server-side fallbacks: if the model declines on policy grounds, the API re-runs the request
        # on a fallback model chosen by refusal category, inside the same call.
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        record = CallRecord(
            step=step,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=time.perf_counter() - t0,
        )
        if response.stop_reason == "refusal":
            raise LLMRefusal(getattr(response.stop_details, "category", None) or "refusal")
        text = "".join(b.text for b in response.content if b.type == "text")
        return text, record


def get_llm() -> LLM:
    s = get_settings()
    return AnthropicLLM() if s.llm_provider == "anthropic" else OllamaLLM()
