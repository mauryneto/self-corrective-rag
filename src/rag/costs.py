"""Token prices and per-call usage records.

Prices are USD per million tokens (input, output). Local models cost 0: the point of tracking
them anyway is that every version of the pipeline reports the same fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class CallRecord:
    step: str  # "generate", "judge", "rewrite", ...
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


@dataclass
class Usage:
    """All LLM calls made to answer one question."""

    calls: list[CallRecord] = field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        self.calls.append(record)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    def summary(self) -> dict:
        return {
            "llm_calls": len(self.calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_latency_s": round(sum(c.latency_s for c in self.calls), 3),
        }
