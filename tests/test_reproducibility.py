"""Reproducibility helpers: Ollama prompt-cache reset and the judge verdict cache."""

import json

import httpx

from rag.costs import CallRecord
from rag.eval.judge import JudgeCache, Verdict, grade
from rag.llm import OllamaLLM


def make_llm(reproducible: bool, num_gpu: int | None = 16):
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json={"message": {"content": "ok"}, "prompt_eval_count": 3, "eval_count": 1})

    llm = OllamaLLM(model="qwen2.5:14b", url="http://ollama")
    llm.client = httpx.Client(transport=httpx.MockTransport(handler))
    llm.reproducible, llm.num_gpu = reproducible, num_gpu
    return llm, sent


def test_every_call_is_preceded_by_a_cache_reset_with_the_same_load_options():
    llm, sent = make_llm(reproducible=True)
    llm.complete("sistema", "pergunta", step="generate")
    reset, real = sent
    assert reset["messages"] == [{"role": "system", "content": "."}] and reset["options"]["num_predict"] == 1
    # options that make Ollama reload the model must match, or the reset itself would reload it
    for opt in ("num_ctx", "num_gpu"):
        assert reset["options"][opt] == real["options"][opt]
    assert real["options"]["num_gpu"] == 16 and real["options"]["temperature"] == 0


def test_no_reset_when_reproducibility_is_off_and_no_num_gpu_when_unset():
    llm, sent = make_llm(reproducible=False, num_gpu=None)
    llm.complete("sistema", "pergunta", step="generate")
    assert len(sent) == 1 and "num_gpu" not in sent[0]["options"]


class CountingJudge:
    model = "claude-haiku-5-5"

    def __init__(self):
        self.calls = 0

    def grade(self, prompt):
        self.calls += 1
        return (Verdict(verdict="correct", refused=False, citations_supported=True, reasoning="ok"),
                CallRecord("judge", self.model, 10, 5, 1.0))


def test_identical_answers_reuse_the_cached_verdict_across_runs(tmp_path):
    path = tmp_path / "cache.jsonl"
    judge = CountingJudge()
    args = ("pergunta", "referência", True, "resposta [1]", {1: "trecho"})
    first = grade(judge, *args, cache=JudgeCache(path))
    again = grade(judge, *args, cache=JudgeCache(path))  # a new run reloads the cache from disk
    changed = grade(judge, "pergunta", "referência", True, "outra resposta [1]", {1: "trecho"}, cache=JudgeCache(path))
    assert judge.calls == 2
    assert not first.cached and again.cached and not changed.cached
    assert again.verdict == first.verdict
