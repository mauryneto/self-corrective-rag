"""Read the ANS legislation portal (componentes-portal.ans.gov.br), the source of *consolidated* texts.

The portal shows each norm as currently in force: amended passages appear struck through
(``<s>``) next to their new wording, and the header says when the text was last updated
("Atualizado em DD/MM/AAAA"). We drop the struck passages and keep the rest, which gives the
text in force today. The "Atualizado em" date drives incremental updates.

The portal is a JavaScript application, so it is read with a headless browser, the same way a
person would use it: the public list of norms, and the public page of each norm. Requests are
sequential and paced.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime

from bs4 import BeautifulSoup

LIST_URL = "https://componentes-portal.ans.gov.br/link/legislacoes"
KIND_LABELS = {"RN": "Resolução Normativa", "IN": "Instrução Normativa"}

_TITLE_RE = re.compile(r"n[º°o]\s*(\d+)", re.IGNORECASE)
_UPDATED_RE = re.compile(r"Atualizado em\s*(\d{2}/\d{2}/\d{4})")
_PUBLISHED_RE = re.compile(r"Publicado em\s*(\d{2}/\d{2}/\d{4})")
_RANGE_RE = re.compile(r"(\d+)\s*a\s*(\d+)\s*de\s*(\d+)")
_SIGNATURE_RE = re.compile(r"DIRETORA?[- ]PRESIDENTE|DIRETORA?\s+(ADJUNT|DE\s)", re.IGNORECASE)
_DISCLAIMER = "Este texto não substitui"


def _date(s: str) -> date:
    return datetime.strptime(s, "%d/%m/%Y").date()


@dataclass(frozen=True)
class PortalRow:
    """One row of the portal list (what can be read without opening the norm)."""

    kind: str
    number: int
    title: str
    dou_date: date
    ementa: str

    @property
    def doc_id(self) -> str:
        return f"{self.kind.lower()}-{self.number}"


@dataclass(frozen=True)
class PortalText:
    url: str
    published: date | None
    updated: date | None
    paragraphs: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat() if self.published else None
        d["updated"] = self.updated.isoformat() if self.updated else None
        return d


def parse_row_cells(cells: list[str], kind: str) -> PortalRow | None:
    """Cells: Legislação, Data DOU, Ementa, Tema, Assunto, Status, Ações."""
    if len(cells) < 3:
        return None
    # Belt and braces: the list is filtered to norms in force, but skip anything not "Vigente".
    if len(cells) >= 6 and cells[5].strip() and cells[5].strip().lower() != "vigente":
        return None
    title, dou, ementa = (re.sub(r"\s+", " ", c).strip() for c in cells[:3])
    m = _TITLE_RE.search(title)
    if not m or not re.fullmatch(r"\d{2}/\d{2}/\d{4}", dou):
        return None
    return PortalRow(kind, int(m.group(1)), title, _date(dou), ementa)


def parse_norm_page(html: str, url: str) -> PortalText:
    """Extract the consolidated text (struck passages removed) and the dates from a norm page."""
    soup = BeautifulSoup(html, "lxml")
    header = soup.get_text(" ")
    updated = _UPDATED_RE.search(header)
    published = _PUBLISHED_RE.search(header)

    # The page has several rich-text viewers (summary, full text, notes); the norm is the largest.
    viewers = soup.select("div[class*=CKEditorViewer]")
    body = max(viewers, key=lambda v: len(v.get_text()), default=None)
    paragraphs: list[str] = []
    if body is not None:
        # Passages replaced or revoked by later norms: <s>/<strike>/<del>, or (older texts) a
        # line-through style.
        struck_els = body.find_all(["s", "strike", "del"]) + body.find_all(
            style=lambda st: bool(st) and "line-through" in st.replace(" ", "").lower())
        for struck in struck_els:
            if not struck.decomposed:
                struck.decompose()
        for el in body.find_all(["p", "tr", "li"]):
            parents = {par.name for par in el.parents}
            if el.name in ("p", "li") and ("table" in parents or "li" in parents):
                continue  # read as part of its table row or list item
            if el.name == "tr":
                cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in el.find_all(["td", "th"])]
                text = " | ".join(c for c in cells if c)
            else:
                text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
            if not text or text.startswith(_DISCLAIMER):
                continue
            if len(text) < 120 and _SIGNATURE_RE.search(text) and not text.startswith("Art"):
                continue
            paragraphs.append(text)
    return PortalText(
        url=url,
        published=_date(published.group(1)) if published else None,
        updated=_date(updated.group(1)) if updated else None,
        paragraphs=paragraphs,
    )


class PortalBrowser:
    """Thin wrapper around a headless Chromium page. Use as a context manager."""

    def __init__(self, delay_ms: int = 800, headless: bool = True):
        self.delay_ms = delay_ms
        self.headless = headless

    def __enter__(self) -> PortalBrowser:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(locale="pt-BR")
        # Two tabs: one stays on the list while the other opens norms, so reading a norm
        # never loses the list's position.
        self.page = self._context.new_page()
        self.doc_page = self._context.new_page()
        for p in (self.page, self.doc_page):
            p.set_default_timeout(60_000)
        self.list_complete = False  # True once iter_rows has walked through the last page
        return self

    def __exit__(self, *exc) -> None:
        self._browser.close()
        self._pw.stop()

    # -- list ------------------------------------------------------------------------------
    def _open_list(self, kind: str) -> None:
        pg = self.page
        pg.goto(LIST_URL, wait_until="networkidle")
        selects = pg.locator("select")
        selects.first.wait_for()
        for i in range(selects.count()):
            options = selects.nth(i).locator("option").all_inner_texts()
            if KIND_LABELS[kind] in options and not selects.nth(i).input_value():
                selects.nth(i).select_option(label=KIND_LABELS[kind])
            elif options == ["", "Sim", "Não"]:
                selects.nth(i).select_option(label="Não")  # "Incluir atos normativos revogados: Não"
            elif "30" in options and "5" in options:
                selects.nth(i).select_option(label="30")  # page size
            pg.wait_for_load_state("networkidle")
        pg.wait_for_timeout(self.delay_ms)

    def _range(self) -> tuple[int, int, int] | None:
        m = _RANGE_RE.search(self.page.inner_text("body"))
        return tuple(int(x) for x in m.groups()) if m else None

    def iter_rows(self, kind: str = "RN", known_urls: dict[int, str] | None = None):
        """Yield (PortalRow, url) for every in-force norm of ``kind``.

        Finding a norm's URL means opening its action menu ("Acessar legislação"), which opens a
        new tab. URLs already in ``known_urls`` (keyed by number) are reused instead.
        """
        known_urls = known_urls or {}
        pg = self.page
        self.list_complete = False
        self._open_list(kind)
        while True:
            rows = pg.locator("[role=row]")
            for i in range(1, rows.count()):  # row 0 is the header
                row = rows.nth(i)
                cells = row.locator("[role=gridcell]")
                parsed = parse_row_cells(cells.all_inner_texts(), kind)
                if parsed is None:
                    continue
                url = known_urls.get(parsed.number) or self._open_row(cells.last)
                yield parsed, url
            current = self._range()
            if current is None:
                break
            if current[1] >= current[2]:
                self.list_complete = True
                break
            first_before = current[0]
            # The grid has pagination controls above and below it; only one is visible.
            # The pagination buttons sit in a zero-size footer that Playwright treats as hidden, so
            # the click is dispatched on the element directly, as the page's own handler expects.
            pg.get_by_label("Go to next page").first.dispatch_event("click")
            pg.wait_for_load_state("networkidle")
            for _ in range(50):  # wait until the grid shows the next page
                now = self._range()
                if now and now[0] != first_before:
                    break
                pg.wait_for_timeout(200)
            pg.wait_for_timeout(self.delay_ms)

    def _open_row(self, actions_cell) -> str | None:
        pg = self.page
        actions_cell.locator("*").first.click()
        item = pg.get_by_text("Acessar legislação", exact=True).locator("visible=true").first
        try:
            with self._context.expect_page(timeout=30_000) as popup_info:
                item.click()
            popup = popup_info.value
            # The tab opens on the app shell and then routes to the norm's own address.
            popup.wait_for_url(re.compile(r"/link/legislacao/\d+"), timeout=30_000)
            url = popup.url
            popup.close()
        except Exception:  # noqa: BLE001 - a row without a public page should not stop the crawl
            pg.keyboard.press("Escape")
            return None
        pg.wait_for_timeout(self.delay_ms)
        return url

    # -- one norm --------------------------------------------------------------------------
    def fetch(self, url: str) -> PortalText:
        pg = self.doc_page
        pg.goto(url, wait_until="networkidle")
        pg.locator("text=Publicado em").first.wait_for()
        pg.wait_for_timeout(self.delay_ms)
        return parse_norm_page(pg.content(), url)
