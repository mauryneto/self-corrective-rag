"""Structural chunking of normative acts.

A legal norm is already split into citable units: articles ("Art. 5º"), which contain paragraphs
("§ 1º") and items ("I -"). Chunking along that structure keeps every chunk self-contained and
citable ("RN nº 465, Art. 5"), which fixed-size windows would break mid-sentence.

Rules:
- one chunk for the summary (ementa), so "what does RN X regulate?" has a direct hit;
- one chunk per article; an article longer than ``max_chars`` is split at paragraph boundaries;
- annexes ("ANEXO ...") are grouped into chunks of up to ``max_chars``;
- quoted text inside amending articles (e.g. '"Art. 5º ..."') stays inside the amending article;
- every chunk starts with a header naming the norm and the unit, so it reads well in a prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ARTICLE_RE = re.compile(r"^Art\.?\s*(\d+)\s*[º°o]?\.?(-[A-Z])?\b")
ANNEX_RE = re.compile(r"^ANEXO\b")
FILLER_RE = re.compile(r"^[\s.…\"'”“]*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    unit: str  # "Ementa", "Art. 5", "Art. 5 (parte 2)", "Anexo"
    text: str  # header + body, what gets embedded and shown to the LLM
    source_url: str


def _header(doc: dict, unit: str) -> str:
    kind = {"RN": "RN", "IN": "IN"}.get(doc["kind"], doc["kind"])
    def br(iso: str) -> str:
        return f"{iso[8:10]}/{iso[5:7]}/{iso[0:4]}"

    when = f"DOU {br(doc['dou_date'])}"
    if doc.get("updated"):  # consolidated text from the ANS portal
        when += f", texto atualizado em {br(doc['updated'])}"
    return f"{kind} nº {doc['number']} ({when}) — {unit}"


ROLE_RE = re.compile(r"^Diretora?(-| )Presidente( Substitut[oa])?$|^Diretora?( Adjunt[oa])?$", re.IGNORECASE)


def _is_signature_name(p: str) -> bool:
    return len(p) < 60 and p.isupper() and not ARTICLE_RE.match(p) and not ANNEX_RE.match(p)


# Some source paragraphs hold several articles (line breaks inside one <p> on the portal), e.g.
# "... 30 dias. Art. 11. O ex-empregado ...". Split right before an "Art. N" that follows the end
# of a sentence or clause and starts a new provision. Lowercase in-text references ("o art. 5º")
# and quoted amendments ('"Art. 5º ...') are left alone.
ARTICLE_BOUNDARY_RE = re.compile(
    r"(?<=[.;:!?)])\s+(?=Art\.?\s*\d+\s*[º°o]?\.?(?:-[A-Z])?\s+[A-ZÀ-Ý])"
)


def _split_articles(paragraph: str) -> list[str]:
    return [part.strip() for part in ARTICLE_BOUNDARY_RE.split(paragraph) if part.strip()]


def _clean_paragraphs(paragraphs: list[str]) -> list[str]:
    out = [part for p in paragraphs if p.strip() and not FILLER_RE.match(p) for part in _split_articles(p.strip())]
    # Drop the signature block: an all-caps name followed by the signer's role.
    keep = [True] * len(out)
    for i, p in enumerate(out):
        if ROLE_RE.match(p):
            keep[i] = False
            if i > 0 and _is_signature_name(out[i - 1]):
                keep[i - 1] = False
    return [p for p, k in zip(out, keep, strict=True) if k]


def _split_long(body: list[str], max_chars: int) -> list[list[str]]:
    """Group paragraphs into pieces of at most max_chars (a single huge paragraph stays whole)."""
    pieces: list[list[str]] = [[]]
    size = 0
    for p in body:
        if pieces[-1] and size + len(p) > max_chars:
            pieces.append([])
            size = 0
        pieces[-1].append(p)
        size += len(p) + 1
    return pieces


def chunk_document(doc: dict, max_chars: int = 1800) -> list[Chunk]:
    paragraphs = _clean_paragraphs(doc["paragraphs"])
    # The first paragraphs are the title (identifica) and the ementa; the DOU repeats them.
    body = [p for p in paragraphs if p != doc.get("dou_title") and p != doc.get("ementa")]

    units: list[tuple[str, list[str]]] = []
    current_unit, current = "Preâmbulo", []
    for p in body:
        art = ARTICLE_RE.match(p)
        if art or ANNEX_RE.match(p):
            if current:
                units.append((current_unit, current))
            current_unit = f"Art. {art.group(1)}{art.group(2) or ''}" if art else "Anexo"
            current = [p]
        else:
            current.append(p)
    if current:
        units.append((current_unit, current))

    chunks: list[Chunk] = []

    def add(unit: str, lines: list[str]) -> None:
        text = _header(doc, unit) + "\n" + "\n".join(lines)
        chunks.append(Chunk(f"{doc['doc_id']}#{len(chunks)}", doc["doc_id"], unit, text, doc["source_url"]))

    # Summary chunk: ementa + preamble (who issues it and on what legal basis).
    preamble = next((lines for unit, lines in units if unit == "Preâmbulo"), [])
    add("Ementa", [doc["ementa"], *preamble][:2])

    for unit, lines in units:
        if unit == "Preâmbulo":
            continue
        pieces = _split_long(lines, max_chars)
        for i, piece in enumerate(pieces):
            add(unit if i == 0 else f"{unit} (parte {i + 1})", piece)
    return chunks
