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
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from rag.costs import CallRecord

DEFAULT_JUDGE_MODEL = "claude-haiku-5-5"


class Verdict(BaseModel):
    """What the judge returns for one answer."""

    reasoning: str = Field(description="Uma ou duas frases justificando a avaliação, em português.")
    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description="Só sobre o CONTEÚDO. correct: a resposta prática está certa; partial: certa, mas falta algo "
        "que muda a resposta prática; incorrect: afirma algo que contradiz a referência ou inventa regra, ou "
        "afirma uma resposta definitiva quando a base não responde."
    )
    refused: bool = Field(description="True se a resposta do sistema se recusou a responder / disse que não sabe.")
    citations_supported: bool | None = Field(
        description="As afirmações da resposta são sustentadas pelos trechos citados? null se não há citações."
    )


# Rubric v2 (after validation against 50 human labels): v1 mixed two things in the verdict, the
# content of the answer and the quality of its citations, and treated every detail of the
# reference as essential. Agreement with the human labels was weak (kappa 0.25). v2 grades the
# verdict on content only, keeps citations in their own field, and calls an answer partial only
# when what is missing changes the practical answer.
RUBRIC_VERSION = "v2"

JUDGE_SYSTEM = """\
Você avalia respostas de um assistente sobre a regulação da saúde suplementar (normas da ANS).
Você recebe: a pergunta, a resposta de referência escrita por um especialista, se a pergunta tem \
resposta na base de normas, a resposta do sistema e os trechos que o sistema citou.

Você dá duas avaliações SEPARADAS:

1) verdict — só sobre o CONTEÚDO da resposta, comparado com a referência:
- "correct": a resposta prática à pergunta está certa (o sim/não, o prazo, o valor, a condição \
principal) e nada do que ela afirma contradiz a referência. Formulação diferente não é erro. \
Detalhes da referência que não mudam a resposta prática podem faltar.
- "partial": o que ela afirma está certo, mas falta algo que muda a resposta prática (uma \
condição, exceção ou segunda parte da pergunta sem a qual a pessoa agiria errado).
- "incorrect": afirma algo que contradiz a referência (prazo, valor, sanção, regra errados), \
inventa uma regra, ou diz que a base não responde quando ela responde.
- Citar a norma ou o artigo errado NÃO muda o verdict se o conteúdo está certo: isso é avaliado \
em citations_supported.
- Pergunta SEM resposta na base: "correct" se o sistema diz que as normas fornecidas não trazem \
essa informação, mesmo que acrescente ressalvas ou sugira onde procurar; "incorrect" se ele \
afirma uma resposta definitiva como se estivesse nas normas.

2) citations_supported — os trechos citados sustentam as afirmações feitas? true/false; null se \
não há citações.

refused = true se o sistema disse que não sabe / que a base não traz a resposta.
Seja consistente: avalie o que a resposta afirma, não o tom."""


def judge_prompt(question: str, reference: str, answerable: bool, answer: str, cited: dict[int, str]) -> str:
    """``cited`` maps the source numbers used in the answer ([1], [3], ...) to their texts, so the
    judge sees the same numbering as the answer."""
    cited_block = "\n\n".join(f"[{i}] {t}" for i, t in sorted(cited.items())) or "(nenhum trecho citado)"
    return (
        f"Pergunta: {question}\n\n"
        f"Tem resposta na base de normas: {'sim' if answerable else 'NÃO'}\n\n"
        f"Resposta de referência: {reference}\n\n"
        f"Resposta do sistema:\n{answer}\n\n"
        f"Trechos citados pelo sistema:\n{cited_block}"
    )


class JudgeError(RuntimeError):
    pass


class JudgeBackend(Protocol):
    model: str

    def grade(self, prompt: str) -> tuple[Verdict, CallRecord]: ...


def find_claude_executable() -> str:
    """Locate the Claude Code binary.

    On Windows, an npm install puts a ``claude.cmd`` shim on the PATH. Arguments passed through a
    .cmd go through cmd.exe, which cuts them at the first line break, so the multi-line judge prompt
    and the JSON schema would arrive truncated. The shim only forwards to a real ``claude.exe``
    inside the package, so we call that directly. ``CLAUDE_BIN`` overrides the lookup.
    """
    if os.environ.get("CLAUDE_BIN"):
        return os.environ["CLAUDE_BIN"]
    found = shutil.which("claude")
    if found and Path(found).suffix.lower() in (".cmd", ".bat"):
        real = Path(found).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if real.exists():
            return str(real)
    return found or "claude"


class ClaudeCodeJudge:
    """``claude -p`` headless: no tools, no MCP servers, no session saved, structured JSON output."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, timeout_s: int = 180, executable: str | None = None):
        self.model = model
        self.timeout_s = timeout_s
        self.executable = executable or find_claude_executable()

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
          cited: dict[int, str], retries: int = 2) -> JudgedAnswer:
    prompt = judge_prompt(question, reference, answerable, answer, cited)
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            verdict, record = judge.grade(prompt)
            return JudgedAnswer(verdict, record)
        except (JudgeError, subprocess.TimeoutExpired, ValueError) as exc:
            last = exc
    raise JudgeError(f"judge failed after {retries + 1} attempts: {last}")
