"""Fetch the full text of a norm from the Diário Oficial da União (in.gov.br).

Two steps, both plain HTML:
1. The daily edition page (``/leiturajornal?data=DD-MM-YYYY&secao=do1``) embeds a JSON list of
   every act published that day. We pick the act whose title names the norm.
2. The act page (``/web/dou/-/<urlTitle>``) holds the text as ``<p>`` elements inside
   ``div.texto-dou``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
from bs4 import BeautifulSoup

from rag.ingest.models import NormRef

BASE = "https://www.in.gov.br"
# Main section 1 first; extra editions ("do1e") also carry regulatory acts.
SECTIONS = ("do1", "do1e", "do1a")

KIND_WORDS = {"RN": "resolucao normativa", "IN": "instrucao normativa"}


@dataclass(frozen=True)
class DouArticle:
    url: str
    title: str
    paragraphs: list[str]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).lower().strip()


def parse_day_listing(html: str) -> list[dict]:
    """Return the list of acts embedded in a daily edition page."""
    soup = BeautifulSoup(html, "lxml")
    script = soup.find("script", id="params")
    if script is None or not script.string:
        return []
    return json.loads(script.string).get("jsonArray", [])


def match_act(acts: list[dict], ref: NormRef) -> dict | None:
    """Find the act that publishes ``ref`` among the acts of one day.

    Titles look like "RESOLUÇÃO NORMATIVA ANS Nº 679, DE 10 DE SETEMBRO DE 2026" (older ones:
    "RESOLUÇÃO NORMATIVA - RN Nº 465, ..."). We require the kind words, the exact number and the ANS
    in the publishing hierarchy, and skip corrections ("retificação").
    """
    words = KIND_WORDS[ref.kind]
    number_re = re.compile(rf"\bn[º°o]?\.?\s*{ref.number}\b")
    for act in acts:
        title = _norm(act.get("title", ""))
        where = _norm(act.get("hierarchyStr", ""))
        if "saude suplementar" not in where:
            continue
        if words in title and number_re.search(title) and "retifica" not in title:
            return act
    return None


def parse_article(html: str) -> tuple[str, list[str]]:
    """Return (title, paragraphs) from an act page."""
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("div.texto-dou")
    if body is None:
        return "", []
    title_el = body.select_one("p.identifica")
    title = title_el.get_text(" ", strip=True) if title_el else ""
    paragraphs: list[str] = []
    # Document order matters: walk <p> and table rows together. Table rows (e.g. annex lines)
    # become one line each, cells separated by " | ".
    for el in body.find_all(["p", "tr"]):
        if el.name == "p":
            if "d-none" in (el.get("class") or []) or any(par.name == "table" for par in el.parents):
                continue
            text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        else:
            cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in el.find_all(["td", "th"])]
            text = " | ".join(c for c in cells if c)
        if text:
            paragraphs.append(text)
    return title, paragraphs


class DouClient:
    def __init__(self, client: httpx.Client):
        self.client = client

    def day_acts(self, day: date, section: str) -> list[dict]:
        resp = self.client.get(
            f"{BASE}/leiturajornal", params={"data": day.strftime("%d-%m-%Y"), "secao": section}
        )
        resp.raise_for_status()
        return parse_day_listing(resp.text)

    def find(self, ref: NormRef, day_offsets: tuple[int, ...] = (0, 1, -1, 2)) -> DouArticle | None:
        """Look for the act on its D.O.U. date first, then on nearby days: the date in the ANS index
        is sometimes the signature or effective date rather than the publication date."""
        for offset in day_offsets:
            for section in SECTIONS:
                found = self._find_on(ref, ref.dou_date + timedelta(days=offset), section)
                if found:
                    return found
        return None

    def _find_on(self, ref: NormRef, day: date, section: str) -> DouArticle | None:
        act = match_act(self.day_acts(day, section), ref)
        if act is None:
            return None
        url = f"{BASE}/web/dou/-/{act['urlTitle']}"
        resp = self.client.get(url)
        resp.raise_for_status()
        title, paragraphs = parse_article(resp.text)
        if not paragraphs:
            return None
        return DouArticle(url=url, title=title or act.get("title", ""), paragraphs=paragraphs)
