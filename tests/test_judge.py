import json
import subprocess

import pytest

from rag.eval import judge as judge_mod
from rag.eval.judge import ClaudeCodeJudge, JudgeError, Verdict, grade, judge_prompt

OK_OUTPUT = {
    "type": "result", "is_error": False, "result": "",
    "structured_output": {"reasoning": "Prazo igual ao da referência.", "verdict": "correct",
                          "refused": False, "citations_supported": True},
    "usage": {"input_tokens": 900, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0,
              "output_tokens": 60},
}


class FakeProc:
    def __init__(self, stdout, stderr=""):
        self.stdout, self.stderr = stdout, stderr


def test_prompt_marks_unanswerable_questions():
    p = judge_prompt("Q?", "Não há norma sobre isso.", False, "Não sei.", {})
    assert "Tem resposta na base de normas: NÃO" in p
    assert "(nenhum trecho citado)" in p


def test_prompt_keeps_the_answer_source_numbers():
    p = judge_prompt("Q?", "R", True, "Prazo de 7 dias [3].", {3: "RN 566 — Art. 3 ..."})
    assert "[3] RN 566 — Art. 3" in p and "[1]" not in p


def test_claude_code_command_gives_the_judge_no_tools_and_no_mcp():
    cmd = ClaudeCodeJudge(executable="claude").command()
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-haiku-5-5"
    schema = json.loads(cmd[cmd.index("--json-schema") + 1])
    assert set(schema["required"]) == {"reasoning", "verdict", "refused", "citations_supported"}


def test_claude_code_judge_parses_structured_output(monkeypatch):
    calls = {}

    def fake_run(cmd, **kw):
        calls["input"] = kw["input"]
        return FakeProc(json.dumps(OK_OUTPUT))

    monkeypatch.setattr(subprocess, "run", fake_run)
    verdict, record = ClaudeCodeJudge(executable="claude").grade("the prompt")
    assert calls["input"] == "the prompt"
    assert verdict == Verdict(reasoning="Prazo igual ao da referência.", verdict="correct", refused=False,
                              citations_supported=True)
    assert record.input_tokens == 1000 and record.output_tokens == 60 and record.step == "judge"


def test_claude_code_judge_surfaces_login_errors(monkeypatch):
    out = {"type": "result", "is_error": True, "result": "Not logged in · Please run /login"}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: FakeProc(json.dumps(out)))
    with pytest.raises(JudgeError, match="Not logged in"):
        ClaudeCodeJudge(executable="claude").grade("x")


def test_grade_retries_then_succeeds():
    class Flaky:
        model = "m"

        def __init__(self):
            self.n = 0

        def grade(self, prompt):
            self.n += 1
            if self.n == 1:
                raise JudgeError("transient")
            return Verdict(reasoning="ok", verdict="partial", refused=False, citations_supported=None), \
                judge_mod.CallRecord("judge", "m", 1, 1, 0.1)

    j = Flaky()
    result = grade(j, "Q", "R", True, "A", {})
    assert result.verdict.verdict == "partial" and j.n == 2
