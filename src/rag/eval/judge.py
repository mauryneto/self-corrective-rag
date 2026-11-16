"""The evaluation judge: grades one RAG answer against the expected answer.

It is deliberately separate from anything inside the pipeline (the V2 judge that decides whether
to retry): an answer must never be graded by the same model and prompt that produced it.

Two interchangeable backends, same prompt and same JSON schema:
- ``claude-code``: runs the Claude Code CLI headless (``claude -p``) on the user's plan. No API key.
- ``anthropic``: the Anthropic API (needs ``ANTHROPIC_API_KEY``), for anyone reproducing the eval.

The judge itself is validated against human labels (see ``rag.eval.validate_judge``) before its
grades are trusted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from rag.costs import CallRecord

DEFAULT_JUDGE_MODEL = "claude-haiku-5-5"


class Verdict(BaseModel):
    """What the judge returns for one answer."""

    reasoning: str = Field(description="Uma ou duas frases justificando a avaliação, em português.")
    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description="correct: responde certo e completo; partial: certo mas incompleto; incorrect: errado, "
        "inventado, ou respondeu quando deveria ter recusado."
    )
    refused: bool = Field(description="True se a resposta do sistema se recusou a responder / disse que não sabe.")
    citations_supported: bool | None = Field(
        description="As afirmações da resposta são sustentadas pelos trechos citados? null se não há citações."
    )


JUDGE_SYSTEM = """\
Você avalia respostas de um assistente sobre a regulação da saúde suplementar (normas da ANS).
Você recebe: a pergunta, a resposta de referência escrita por um especialista, se a pergunta tem \
resposta na base de normas, a resposta do sistema e os trechos que o sistema citou.

Critérios:
- Compare o CONTEÚDO com a referência (prazos, números, condições, exceções). Formulação diferente \
não é erro.
- "correct": todos os pontos essenciais da referência estão presentes e nada contradiz a referência.
- "partial": o que foi dito está certo, mas falta um ponto essencial.
- "incorrect": contradiz a referência, inventa regra, cita a norma errada como base, ou responde \
algo que não está na base quando a pergunta NÃO tem resposta na base.
- Se a pergunta NÃO tem resposta na base, a resposta certa é recusar (dizer que as normas \
fornecidas não tratam do assunto): isso é "correct" com refused=true.
- Se a pergunta TEM resposta na base e o sistema recusou: "incorrect" com refused=true.
- citations_supported: verifique só se os trechos citados sustentam as afirmações feitas; null se \
não houver citações.
Seja rigoroso e consistente."""


def judge_prompt(question: str, reference: str, answerable: bool, answer: str, cited_texts: list[str]) -> str:
    cited = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(cited_texts, 1)) or "(nenhum trecho citado)"
    return (
        f"Pergunta: {question}\n\n"
        f"Tem resposta na base de normas: {'sim' if answerable else 'NÃO'}\n\n"
        f"Resposta de referência: {reference}\n\n"
        f"Resposta do sistema:\n{answer}\n\n"
        f"Trechos citados pelo sistema:\n{cited}"
    )


class JudgeError(RuntimeError):
    pass


class JudgeBackend(Protocol):
    model: str

    def grade(self, prompt: str) -> tuple[Verdict, CallRecord]: ...


class ClaudeCodeJudge:
    """``claude -p`` headless: no tools, no MCP servers, no session saved, structured JSON output."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, timeout_s: int = 180, executable: str | None = None):
        self.model = model
        self.timeout_s = timeout_s
        self.executable = executable or shutil.which("claude") or "claude"

    def command(self) -> list[str]:
        return [
            self.executable, "-p",
            "--model", self.model,
            "--tools", "",                 # the judge only reads; it gets no tools
            "--strict-mcp-config",         # and no MCP servers from the user's config
            "--no-session-persistence",
            "--output-format", "json",
            "--system-prompt", JUDGE_SYSTEM,
            "--json-schema", json.dumps(Verdict.model_json_schema()),
        ]

    def grade(self, prompt: str) -> tuple[Verdict, CallRecord]:
        t0 = time.perf_counter()
        # Run from an empty folder so no project instructions (CLAUDE.md) leak into the judge.
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                self.command(), input=prompt, capture_output=True, text=True, encoding="utf-8",
                timeout=self.timeout_s, cwd=cwd, env={**os.environ},
            )
        latency = time.perf_counter() - t0
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"claude -p did not return JSON: {proc.stdout[:300]!r} {proc.stderr[:300]!r}") from exc
        if out.get("is_error"):
            raise JudgeError(f"claude -p failed: {out.get('result')!r}")
        structured = out.get("structured_output")
        if structured is None:  # some versions put the JSON in "result"
            structured = json.loads(out.get("result") or "{}")
        verdict = Verdict.model_validate(structured)
        usage = out.get("usage") or {}
        record = CallRecord(
            step="judge", model=self.model,
            input_tokens=usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0), latency_s=latency,
        )
        return verdict, record


class AnthropicJudge:
    """Same judge through the Anthropic API (needs ANTHROPIC_API_KEY)."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()

    def grade(self, prompt: str) -> tuple[Verdict, CallRecord]:
        t0 = time.perf_counter()
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=Verdict,
        )
        record = CallRecord("judge", self.model, response.usage.input_tokens, response.usage.output_tokens,
                            time.perf_counter() - t0)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise JudgeError(f"judge returned no verdict (stop_reason={response.stop_reason})")
        return response.parsed_output, record


@dataclass
class JudgedAnswer:
    verdict: Verdict
    record: CallRecord


def get_judge(backend: str = "claude-code", model: str = DEFAULT_JUDGE_MODEL) -> JudgeBackend:
    if backend == "claude-code":
        return ClaudeCodeJudge(model)
    if backend == "anthropic":
        return AnthropicJudge(model)
    raise ValueError(f"unknown judge backend: {backend}")


def grade(judge: JudgeBackend, question: str, reference: str, answerable: bool, answer: str,
          cited_texts: list[str], retries: int = 2) -> JudgedAnswer:
    prompt = judge_prompt(question, reference, answerable, answer, cited_texts)
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            verdict, record = judge.grade(prompt)
            return JudgedAnswer(verdict, record)
        except (JudgeError, subprocess.TimeoutExpired, ValueError) as exc:
            last = exc
    raise JudgeError(f"judge failed after {retries + 1} attempts: {last}")
