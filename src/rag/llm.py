"""Provider-agnostic LLM calls: local models through Ollama, or Claude through the Anthropic API.

Every call returns the text plus a ``CallRecord`` (tokens, latency, model), which the pipeline
aggregates into the per-question cost and latency reported by the eval.

``complete(..., on_text=f)`` streams: ``f`` gets each piece of the answer as the model writes it,
and the call still returns the whole text and its record. Without ``on_text`` the request is the
one the eval measures.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Protocol

import httpx
from pydantic import BaseModel

from rag.config import get_settings
from rag.costs import CallRecord
from rag.retry import with_retries

OnText = Callable[[str], None]


class LLMRefusal(RuntimeError):
    """The model declined to answer (Claude ``stop_reason == "refusal"``)."""


class StreamInterrupted(RuntimeError):
    """A streamed answer stopped after part of it had been sent on: retrying would repeat that part."""


class LLM(Protocol):
    model: str

    def complete(self, system: str, prompt: str, step: str, max_tokens: int = 2048,
                 on_text: OnText | None = None) -> tuple[str, CallRecord]: ...

    def complete_json[M: BaseModel](self, system: str, prompt: str, schema: type[M], step: str,
                                    max_tokens: int = 1024) -> tuple[M, CallRecord]: ...


class OllamaLLM:
    def __init__(self, model: str | None = None, url: str | None = None):
        s = get_settings()
        self.model = model or s.ollama_llm_model
        self.url = (url or s.ollama_url).rstrip("/")
        self.num_gpu = s.ollama_num_gpu
        self.num_ctx = s.ollama_num_ctx
        self.reproducible = s.ollama_reproducible
        self.client = httpx.Client(timeout=600)

    def _options(self, max_tokens: int) -> dict:
        # Same num_ctx / num_gpu on every request: a different value makes Ollama reload the model.
        return ({"temperature": 0, "seed": 0, "num_predict": max_tokens, "num_ctx": self.num_ctx}
                | ({"num_gpu": self.num_gpu} if self.num_gpu is not None else {}))

    def _reset_cache(self) -> None:
        """Make the next request's arithmetic independent of the previous one.

        Ollama reuses the KV cache for the prefix a request shares with the previous one, and how
        much is reused changes how the rest of the prompt is batched, hence the floating-point
        rounding, hence (at temperature 0) sometimes the answer. A one-token request whose system
        prompt differs from every real one leaves only the chat template's opening tokens to reuse,
        the same for every request; and since it is also the first request after any model load,
        those tokens are always computed the same way. Costs one tiny request (~0.2 s)."""
        resp = self.client.post(f"{self.url}/api/chat", json={
            "model": self.model, "stream": False, "messages": [{"role": "system", "content": "."}],
            "options": self._options(1)})
        resp.raise_for_status()

    def _stream(self, body: dict, on_text: OnText) -> dict:
        """The request with ``stream: true``. Each piece of text goes to ``on_text`` as it arrives;
        returns what the non-streamed request returns (the whole message and the token counts)."""
        pieces: list[str] = []
        try:
            with self.client.stream("POST", f"{self.url}/api/chat", json=body) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if "error" in part:
                        raise StreamInterrupted(str(part["error"]))
                    piece = part.get("message", {}).get("content", "")
                    if piece:
                        pieces.append(piece)
                        on_text(piece)
                    if part.get("done"):
                        return part | {"message": {"content": "".join(pieces)}}
        except httpx.HTTPError as exc:
            if pieces:  # not transient any more: the caller already has the start of the answer
                raise StreamInterrupted(f"{type(exc).__name__}: {exc}") from exc
            raise
        raise StreamInterrupted("the stream ended before the final message")

    def _chat(self, system: str, prompt: str, step: str, max_tokens: int, fmt: dict | None,
              on_text: OnText | None = None) -> tuple[str, CallRecord]:
        t0 = time.perf_counter()
        body = {
            "model": self.model,
            "stream": on_text is not None,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "options": self._options(max_tokens),
        }
        if fmt is not None:
            body["format"] = fmt  # structured output: Ollama constrains decoding to this JSON schema

        def call() -> dict:
            if self.reproducible:
                self._reset_cache()
            if on_text is not None:
                return self._stream(body, on_text)
            resp = self.client.post(f"{self.url}/api/chat", json=body)
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

    def complete(self, system: str, prompt: str, step: str, max_tokens: int = 2048,
                 on_text: OnText | None = None) -> tuple[str, CallRecord]:
        return self._chat(system, prompt, step, max_tokens, None, on_text)

    def complete_json[M: BaseModel](self, system: str, prompt: str, schema: type[M], step: str,
                                    max_tokens: int = 1024) -> tuple[M, CallRecord]:
        text, record = self._chat(system, prompt, step, max_tokens, schema.model_json_schema())
        return schema.model_validate_json(text), record


class AnthropicLLM:
    def __init__(self, model: str | None = None, effort: str = "medium"):
        import anthropic

        self.model = model or get_settings().anthropic_model
        self.effort = effort
        self.client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or an `ant auth` profile

    def complete(self, system: str, prompt: str, step: str, max_tokens: int = 16000,
                 on_text: OnText | None = None) -> tuple[str, CallRecord]:
        """Not streamed: with ``on_text``, the answer is handed over in one piece when it is ready."""
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
        if on_text is not None:
            on_text(text)
        return text, record

    def complete_json[M: BaseModel](self, system: str, prompt: str, schema: type[M], step: str,
                                    max_tokens: int = 4000) -> tuple[M, CallRecord]:
        t0 = time.perf_counter()
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        record = CallRecord(step, response.model, response.usage.input_tokens, response.usage.output_tokens,
                            time.perf_counter() - t0)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise LLMRefusal(f"no structured output (stop_reason={response.stop_reason})")
        return response.parsed_output, record


def get_llm() -> LLM:
    s = get_settings()
    return AnthropicLLM() if s.llm_provider == "anthropic" else OllamaLLM()
