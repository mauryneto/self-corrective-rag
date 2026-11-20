"""Demo: ask a question about the ANS norms and watch the answer being written.

    uv run uvicorn rag.api:app                        # the API
    uv run --group demo streamlit run demo/app.py     # this page, on http://127.0.0.1:8501

The page is a client of the API's ``POST /ask/stream`` and nothing more: what it shows is what any
client of the API gets. ``RAG_API_URL`` points it at an API somewhere else.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import httpx
import streamlit as st

from rag.sse import parse_events

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")

EXAMPLES = [
    ("One article", "Qual o prazo máximo para a operadora garantir uma consulta com psicólogo?"),
    ("Two articles", "Pedi autorização para uma internação eletiva. Em quanto tempo a operadora tem que me "
                     "responder e em quanto tempo tem que garantir a internação?"),
    ("Not in the norms", "Quantas operadoras de planos de saúde estão em atividade hoje no Brasil?"),
    ("Out of scope", "Meu seguro de carro cobre batida com animal na estrada?"),
]
VERSIONS = {
    "v4": "V4 · hybrid search, reranker on the GPU, scope gate",
    "v3": "V3 · hybrid search, reranker, scope gate",
    "v2": "V2 · V1 plus the self-corrective loop (slower)",
    "v1": "V1 · hybrid search and reranker",
    "v0": "V0 · vector search only",
}
STEPS = {
    "check_context": "Searching the norms and checking whether the passages cover the question…",
    "check_answer": "Checking the answer against the passages it cites…",
    "revise": "The check found a problem: the answer is being rewritten…",
}


def events(question: str, version: str) -> Iterator[tuple[str, dict]]:
    body = {"question": question, "version": version}
    with httpx.stream("POST", f"{API_URL}/ask/stream", json=body, timeout=httpx.Timeout(600, connect=5)) as resp:
        resp.raise_for_status()
        yield from parse_events(resp.iter_lines())


def md(text: str) -> str:
    return text.replace("$", "\\$")  # "R$ 100" is an amount, not the start of a formula


def show_sources(sources: list[dict]) -> None:
    st.caption("Passages in the prompt, best first. The score is the reranker's (vector similarity in V0).")
    for s in sources:
        cited = "  ·  cited in the answer" if s.get("cited") else ""
        with st.expander(f"[{s['index']}]  {s['norm']} · {s['unit']}  ·  score {s['score']:.2f}{cited}"):
            st.markdown(md(s["text"]).replace("\n", "  \n"))
            st.markdown(f"[Open the norm on the ANS portal]({s['url']})")


def run(question: str, version: str) -> dict:
    """Shows the answer while it is written; returns the finished response."""
    st.markdown(f"**{md(question)}**")
    status, answer = st.empty(), st.empty()
    status.info("Searching the norms…")
    text, t0 = "", time.perf_counter()
    try:
        return follow(events(question, version), status, answer, text, t0)
    except Exception:
        status.empty()  # the caller shows the error in its place
        raise


def follow(stream: Iterator[tuple[str, dict]], status, answer, text: str, t0: float) -> dict:
    for kind, data in stream:
        if kind == "sources":
            found = f"{len(data['sources'])} passages found in {data['retrieval_latency_s']:.1f} s."
            status.info(f"{found} None is about this question: declined without calling the model."
                        if data["declined"] else f"{found} The model is reading them…")
        elif kind == "step":
            if data["name"] == "revise":
                text = ""
                answer.empty()
            status.info(STEPS.get(data["name"], data["name"]))
        elif kind == "text":
            if not text:
                status.info(f"First words after {time.perf_counter() - t0:.1f} s. Writing…")
            text += data["text"]
            answer.markdown(md(text) + " ▌")
        elif kind == "error":
            raise RuntimeError(data["detail"])
        elif kind == "done":
            return data
    raise RuntimeError("the stream ended before the answer was finished")


def show(question: str, r: dict) -> None:
    st.markdown(f"**{md(question)}**")
    st.markdown(md(r["answer"]))
    usage = r["usage"]
    first, total, search, calls = st.columns(4)
    first.metric("First words", "n/a" if r["first_token_s"] is None else f"{r['first_token_s']:.1f} s")
    total.metric("Whole answer", f"{r['latency_s']:.1f} s")
    search.metric("Search", f"{r['retrieval_latency_s']:.1f} s")
    calls.metric("Model calls", usage["llm_calls"])
    st.caption(f"{r['version'].upper()} · {usage['input_tokens']:,} tokens in, {usage['output_tokens']:,} out · "
               f"US$ {usage['cost_usd']:.4f}")
    show_sources(r["sources"])


@st.cache_data(ttl=60, show_spinner=False)
def health() -> dict:
    """What the API has indexed and which model answers. A failure is not cached, so the page
    notices the API as soon as it is up."""
    return httpx.get(f"{API_URL}/health", timeout=httpx.Timeout(10, connect=1)).raise_for_status().json()


def pick(question: str) -> None:
    st.session_state.question = question
    st.session_state.go = True


st.set_page_config(page_title="ANS norms assistant", page_icon="📘")
st.title("ANS norms assistant")
st.caption("Answers questions about Brazilian health-insurance regulation from the normative resolutions of ANS, "
           "citing the articles it used, and says so when the norms do not answer. Questions in Portuguese.")

with st.sidebar:
    version = st.radio("Pipeline version", list(VERSIONS), format_func=VERSIONS.get)
    try:
        info = health()
        st.caption(f"{info['norms']} norms, {info['chunks']:,} passages indexed. Answering model: `{info['model']}`.")
    except httpx.HTTPError:
        st.warning(f"No API at {API_URL}. Start it with `uv run uvicorn rag.api:app`.")

st.session_state.setdefault("question", "")
for column, (label, example) in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
    column.button(label, help=example, on_click=pick, args=(example,), width="stretch")
with st.form("ask", border=False):
    st.text_input("Question", key="question", placeholder="Qual o prazo máximo para uma consulta básica?")
    if st.form_submit_button("Ask", type="primary"):
        st.session_state.go = True

area = st.empty()  # one slot for the answer: the one being written replaces the previous one
question = st.session_state.question.strip()
if st.session_state.pop("go", False) and len(question) >= 3:
    st.session_state.pop("result", None)
    with area.container():
        try:
            st.session_state.result = (question, run(question, version))
        except httpx.HTTPError as exc:
            st.error(f"The API at {API_URL} did not answer ({type(exc).__name__}). "
                     "Start it with `uv run uvicorn rag.api:app`.")
        except RuntimeError as exc:
            st.error(f"The pipeline failed: {exc}")
if "result" in st.session_state:
    with area.container():
        show(*st.session_state.result)
