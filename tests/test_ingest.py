import json
from datetime import date

from rag.ingest.ans_portal import parse_norm_page, parse_row_cells
from rag.ingest.build_corpus import revoked_to_remove
from rag.ingest.dou import match_act, parse_article, parse_day_listing
from rag.ingest.models import NormRef

# --- ANS portal ---------------------------------------------------------------------------------

NORM_PAGE = """<html><body>
<h1>Resolução Normativa - RN ANS nº 465, de 25 fevereiro 2021</h1>
<div>Publicado em 02/03/2021 | Atualizado em 03/11/2026</div>
<div class="wgt-CKEditorViewerForMendixNode"><p>Atualiza o Rol de Procedimentos.</p></div>
<div class="wgt-CKEditorViewerForMendixNode">
  <p>A Diretoria Colegiada da ANS adota a seguinte Resolução Normativa.</p>
  <p>Art. 1º Esta Resolução atualiza o Rol.</p>
  <p><s>Art. 2º Redação antiga, revogada.</s></p>
  <p>Art. 2º Redação nova, em vigor. (Redação dada pela RN nº 500)</p>
  <table><tr><td>10101012</td><td>Consulta em consultório</td></tr></table>
  <p>&nbsp;</p>
  <p>FULANO DE TALDIRETOR-PRESIDENTE SUBSTITUTO</p>
  <p>Este texto não substitui o texto normativo original e nem o de suas alterações.</p>
</div></body></html>"""


def test_portal_page_keeps_text_in_force_and_drops_struck_passages():
    t = parse_norm_page(NORM_PAGE, "https://portal/link/legislacao/1")
    assert t.updated == date(2026, 11, 3) and t.published == date(2021, 3, 2)
    assert t.paragraphs == [
        "A Diretoria Colegiada da ANS adota a seguinte Resolução Normativa.",
        "Art. 1º Esta Resolução atualiza o Rol.",
        "Art. 2º Redação nova, em vigor. (Redação dada pela RN nº 500)",
        "10101012 | Consulta em consultório",
    ]


def test_portal_row_cells():
    row = parse_row_cells(["Resolução Normativa - RN ANS nº 683, de 13 novembro 2026", "16/11/2026",
                           "Altera a RN 465.", "Tema", "Assunto", "Vigente", ""], "RN")
    assert row.number == 683 and row.dou_date == date(2026, 11, 16) and row.doc_id == "rn-683"
    assert parse_row_cells(["Legislação", "Data DOU", "Ementa"], "RN") is None  # header row


def test_revoked_norms_are_removed_only_after_a_complete_crawl():
    on_disk = {f"rn-{i}" for i in range(100)}
    in_force = on_disk - {"rn-1", "rn-2"}
    assert revoked_to_remove(on_disk, in_force, list_complete=True) == (["rn-1", "rn-2"], None)
    assert revoked_to_remove(on_disk, in_force, list_complete=False)[0] == []


def test_a_broken_crawl_cannot_wipe_the_corpus():
    on_disk = {f"rn-{i}" for i in range(200)}
    removed, reason = revoked_to_remove(on_disk, {"rn-1"}, list_complete=True)
    assert removed == [] and "too many" in reason


# --- Diário Oficial (fallback) ------------------------------------------------------------------


def _listing(acts):
    return f'<html><script type="application/json" id="params">{json.dumps({"jsonArray": acts})}</script></html>'


REF = NormRef("RN", 679, "Resolução Normativa nº 679", date(2026, 9, 14), "Altera a RN 465.")
ANS = "Ministério da Saúde/Agência Nacional de Saúde Suplementar/Diretoria Colegiada"


def test_match_act_picks_the_ans_resolution_with_the_right_number():
    acts = [
        {"title": "RESOLUÇÃO Nº 679, DE 1 DE SETEMBRO DE 2026", "hierarchyStr": "Ministério de Portos", "urlTitle": "x"},
        {"title": "RESOLUÇÃO NORMATIVA ANS Nº 6790, DE 10 DE SETEMBRO", "hierarchyStr": ANS, "urlTitle": "wrong-number"},
        {"title": "RETIFICAÇÃO - RESOLUÇÃO NORMATIVA ANS Nº 679", "hierarchyStr": ANS, "urlTitle": "retificacao"},
        {"title": "RESOLUÇÃO NORMATIVA ANS Nº 679, DE 10 DE SETEMBRO DE 2026", "hierarchyStr": ANS, "urlTitle": "ok"},
    ]
    assert match_act(parse_day_listing(_listing(acts)), REF)["urlTitle"] == "ok"


def test_match_act_accepts_older_title_style():
    acts = [{"title": "RESOLUÇÃO NORMATIVA - RN Nº 679, DE 10 DE SETEMBRO DE 2026", "hierarchyStr": ANS,
             "urlTitle": "ok"}]
    assert match_act(acts, REF)["urlTitle"] == "ok"


def test_match_act_returns_none_when_absent():
    assert match_act([], REF) is None


def test_dou_client_falls_back_to_the_next_day():
    from rag.ingest.dou import DouClient

    act = {"title": "RESOLUÇÃO NORMATIVA ANS Nº 679, DE 10 DE SETEMBRO DE 2026", "hierarchyStr": ANS, "urlTitle": "rn-679"}
    article = '<div class="texto-dou"><p class="identifica">RN 679</p><p>Art. 1º Texto.</p></div>'

    class FakeResp:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    class FakeHttp:
        def __init__(self):
            self.days = []

        def get(self, url, params=None):
            if params:  # daily listing: the act only appears one day after the indexed date
                self.days.append(params["data"])
                return FakeResp(_listing([act] if params["data"] == "15-09-2026" else []))
            return FakeResp(article)

    http = FakeHttp()
    found = DouClient(http).find(REF)
    assert found is not None and found.url.endswith("/rn-679")
    assert http.days[0] == "14-09-2026" and "15-09-2026" in http.days


def test_parse_article_keeps_order_and_table_rows():
    html = """<div class="texto-dou">
      <p class="identifica">RESOLUÇÃO NORMATIVA ANS Nº 679</p>
      <p class="ementa">Altera a RN 465.</p>
      <p class="dou-paragraph">Art. 1º Texto.</p>
      <table><tr><td><p>Código</p></td><td>Procedimento</td></tr></table>
      <p class="dou-paragraph">Art. 2º Vigência.</p>
      <p class="d-none">escondido</p>
    </div>"""
    title, paragraphs = parse_article(html)
    assert title == "RESOLUÇÃO NORMATIVA ANS Nº 679"
    assert paragraphs == ["RESOLUÇÃO NORMATIVA ANS Nº 679", "Altera a RN 465.", "Art. 1º Texto.",
                          "Código | Procedimento", "Art. 2º Vigência."]
