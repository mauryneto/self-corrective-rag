from rag.chunking import chunk_document


def make_doc(paragraphs, ementa="Dispõe sobre X."):
    return {
        "kind": "RN",
        "number": 999,
        "dou_date": "2026-07-20",
        "doc_id": "rn-999",
        "ementa": ementa,
        "dou_title": "RESOLUÇÃO NORMATIVA ANS Nº 999, DE 20 DE JULHO DE 2026",
        "source_url": "https://example.org/rn-999",
        "paragraphs": paragraphs,
    }


BASIC = [
    "RESOLUÇÃO NORMATIVA ANS Nº 999, DE 20 DE JULHO DE 2026",
    "Dispõe sobre X.",
    "A Diretoria Colegiada da ANS adota a seguinte Resolução Normativa.",
    "Art. 1º Esta Resolução dispõe sobre X.",
    "Art. 2º A operadora deverá garantir:",
    "I - atendimento em até 7 dias;",
    "II - informação ao beneficiário.",
    "..........................................................................",
    "Art. 3º Esta Resolução entra em vigor na data de sua publicação.",
    "FULANO DE TAL",
    "Diretor-Presidente",
]


def test_one_chunk_per_article_plus_ementa():
    chunks = chunk_document(make_doc(BASIC))
    assert [c.unit for c in chunks] == ["Ementa", "Art. 1", "Art. 2", "Art. 3"]


def test_items_stay_with_their_article_and_filler_is_dropped():
    art2 = chunk_document(make_doc(BASIC))[2]
    assert "I - atendimento em até 7 dias;" in art2.text
    assert "II - informação ao beneficiário." in art2.text
    assert "....." not in art2.text


def test_signature_block_is_dropped():
    last = chunk_document(make_doc(BASIC))[-1]
    assert "FULANO" not in last.text and "Diretor-Presidente" not in last.text


def test_header_names_norm_date_and_unit():
    art1 = chunk_document(make_doc(BASIC))[1]
    assert art1.text.splitlines()[0] == "RN nº 999 (DOU 20/07/2026) — Art. 1"


def test_ementa_chunk_has_summary_and_preamble():
    ementa = chunk_document(make_doc(BASIC))[0]
    assert "Dispõe sobre X." in ementa.text
    assert "Diretoria Colegiada" in ementa.text


def test_quoted_articles_stay_inside_the_amending_article():
    paragraphs = [
        "Art. 1º O art. 5º da RN nº 465 passa a vigorar com a seguinte redação:",
        '"Art. 5º A cobertura inclui Y." (NR)',
        "Art. 2º Esta Resolução entra em vigor na data de sua publicação.",
    ]
    chunks = chunk_document(make_doc(paragraphs))
    assert [c.unit for c in chunks] == ["Ementa", "Art. 1", "Art. 2"]
    assert "A cobertura inclui Y." in chunks[1].text


def test_letter_suffixed_articles():
    chunks = chunk_document(make_doc(["Art. 4º-A Fica incluído Z.", "Art. 5º Vigência imediata."]))
    assert [c.unit for c in chunks] == ["Ementa", "Art. 4-A", "Art. 5"]


def test_long_article_is_split_at_paragraph_boundaries():
    long_items = [f"{i} - item {'x' * 300}" for i in range(10)]
    chunks = chunk_document(make_doc(["Art. 1º Lista:", *long_items]), max_chars=1000)
    units = [c.unit for c in chunks]
    assert units[1] == "Art. 1" and units[2] == "Art. 1 (parte 2)"
    joined = "\n".join(c.text for c in chunks[1:])
    assert all(item in joined for item in long_items)  # nothing lost
    assert all(len(c.text) < 1000 + 400 for c in chunks)  # header + at most one overflow item


def test_annex_is_its_own_unit():
    chunks = chunk_document(make_doc(["Art. 1º Ver anexo.", "ANEXO I", "Linha 1 | valor", "Linha 2 | valor"]))
    assert chunks[-1].unit == "Anexo"
    assert "Linha 2 | valor" in chunks[-1].text


def test_chunk_ids_are_unique_and_stable():
    a = chunk_document(make_doc(BASIC))
    b = chunk_document(make_doc(BASIC))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_header_says_when_a_consolidated_text_was_updated():
    doc = make_doc(BASIC) | {"updated": "2026-11-03"}
    header = chunk_document(doc)[1].text.splitlines()[0]
    assert header == "RN nº 999 (DOU 20/07/2026, texto atualizado em 03/11/2026) — Art. 1"
