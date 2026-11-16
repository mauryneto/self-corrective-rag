"""The eval question set: ``eval/questions.jsonl``, one JSON object per line.

Gold sources are (norm, unit) pairs such as ``{"doc": "rn-566", "unit": "Art. 3"}`` rather than
chunk ids, so they survive changes to the chunking parameters. A retrieved chunk counts as a hit
when it belongs to that norm and its unit is that article (or one of its parts).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from rag.config import ROOT

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"

QuestionType = Literal["direct", "multi_hop", "unanswerable", "out_of_scope"]


class GoldSource(BaseModel):
    doc: str  # e.g. "rn-566"
    unit: str  # e.g. "Art. 3"

    def matches(self, doc_id: str, unit: str) -> bool:
        return doc_id == self.doc and (unit == self.unit or unit.startswith(self.unit + " (parte"))


class Question(BaseModel):
    id: str
    type: QuestionType
    question: str
    reference_answer: str
    gold: list[GoldSource] = Field(default_factory=list)
    status: Literal["draft", "reviewed"] = "draft"
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.type in ("direct", "multi_hop")

    @model_validator(mode="after")
    def _gold_matches_type(self) -> Question:
        if self.answerable and not self.gold:
            raise ValueError(f"{self.id}: answerable questions need at least one gold source")
        if not self.answerable and self.gold:
            raise ValueError(f"{self.id}: unanswerable/out-of-scope questions must not have gold sources")
        if self.type == "multi_hop" and len({(g.doc, g.unit) for g in self.gold}) < 2:
            raise ValueError(f"{self.id}: multi_hop questions need sources from at least two articles")
        return self


def load_questions(path: Path = QUESTIONS_PATH, reviewed_only: bool = False) -> list[Question]:
    questions = [Question.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return [q for q in questions if q.status == "reviewed"] if reviewed_only else questions


def save_questions(questions: list[Question], path: Path = QUESTIONS_PATH) -> None:
    path.write_text("".join(json.dumps(q.model_dump(), ensure_ascii=False) + "\n" for q in questions),
                    encoding="utf-8")
