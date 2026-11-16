"""Write ``eval/review.md``: every question next to the verbatim evidence that supports it.

    uv run python -m rag.eval.review_sheet

Reviewing then means checking, for each question, that the reference answer says what the quoted
text says. No knowledge of health-insurance regulation is needed.
"""

from __future__ import annotations

from rag.config import ROOT
from rag.eval.dataset import load_questions

TYPE_LABELS = {
    "direct": "Diretas (um artigo responde)",
    "multi_hop": "Multi-hop (combina dois ou mais artigos)",
    "unanswerable": "Sem resposta na base (o sistema deve dizer que não sabe)",
    "out_of_scope": "Fora de escopo (o sistema deve recusar)",
}


def main() -> None:
    questions = load_questions()
    lines = ["# Revisão do conjunto de avaliação", "",
             "Para cada pergunta: a resposta esperada diz o mesmo que a citação? Se não, anote o id.", ""]
    for qtype, label in TYPE_LABELS.items():
        group = [q for q in questions if q.type == qtype]
        lines += [f"## {label} — {len(group)}", ""]
        for q in group:
            lines += [f"### {q.id} · {q.question}", "", f"**Resposta esperada:** {q.reference_answer}", ""]
            for g in q.gold:
                lines += [f"> **{g.doc.upper()}, {g.unit}:** “{g.evidence}”", ""]
            if q.absence_terms:
                lines += [f"*Termos procurados no corpus sem encontrar a resposta:* {', '.join(q.absence_terms)}", ""]
            if q.notes:
                lines += [f"*Nota:* {q.notes}", ""]
    path = ROOT / "eval" / "review.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path} ({len(questions)} questions)")


if __name__ == "__main__":
    main()
