"""Mechanical checks on the question set, so reviewing it needs no domain knowledge.

    uv run python -m rag.eval.check_questions

Errors (exit code 1):
- a gold norm that is not in the corpus, or an article that does not exist in it;
- an evidence quote that is not, word for word, in that article;
- duplicated questions.
Warnings:
- unanswerable questions whose ``absence_terms`` do appear in the corpus (maybe answerable after all).
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import defaultdict

from rag.chunking import chunk_document
from rag.eval.dataset import Question, load_questions
from rag.ingest.build_corpus import load_corpus


def normalize(text: str) -> str:
    """Lowercase, no accents, unified quotes/dashes/ordinals, single spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("º", "o").replace("°", "o").replace("˚", "o").replace("ª", "a")
    text = re.sub(r"[“”\"'‘’]", "", text)
    text = re.sub(r"[–—‐-]", "-", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def unit_texts(docs: list[dict]) -> dict[tuple[str, str], str]:
    """(doc_id, base unit) -> normalized text of all chunks of that unit (parts joined)."""
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for doc in docs:
        for c in chunk_document(doc):
            base = re.sub(r" \(parte \d+\)$", "", c.unit)
            out[(c.doc_id, base)].append(c.text.split("\n", 1)[1] if "\n" in c.text else c.text)
    return {k: normalize(" ".join(v)) for k, v in out.items()}


def check(questions: list[Question], docs: list[dict]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    units = unit_texts(docs)
    doc_ids = {d["doc_id"] for d in docs}
    corpus_text = normalize(" ".join(p for d in docs for p in d["paragraphs"]))

    seen: dict[str, str] = {}
    for q in questions:
        key = normalize(q.question)
        if key in seen:
            errors.append(f"{q.id}: same question as {seen[key]}")
        seen[key] = q.id

        for g in q.gold:
            if g.doc not in doc_ids:
                errors.append(f"{q.id}: norm {g.doc} is not in the corpus")
                continue
            text = units.get((g.doc, g.unit))
            if text is None:
                errors.append(f"{q.id}: {g.doc} has no unit '{g.unit}'")
                continue
            if not g.evidence.strip():
                errors.append(f"{q.id}: {g.doc} {g.unit} has no evidence quote")
            elif normalize(g.evidence) not in text:
                errors.append(f"{q.id}: evidence not found word for word in {g.doc} {g.unit}: {g.evidence[:80]!r}")

        if q.type == "unanswerable":
            if not q.absence_terms:
                warnings.append(f"{q.id}: unanswerable question without absence_terms")
            for term in q.absence_terms:
                hits = corpus_text.count(normalize(term))
                if hits:
                    warnings.append(f"{q.id}: '{term}' appears {hits}x in the corpus: check it is really unanswerable")
    return errors, warnings


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None, help="question file (default: eval/questions.jsonl)")
    args = parser.parse_args()
    questions = load_questions(args.file) if args.file else load_questions()
    errors, warnings = check(questions, load_corpus())
    by_type: dict[str, int] = defaultdict(int)
    for q in questions:
        by_type[q.type] += 1
    print(f"{len(questions)} questions: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
