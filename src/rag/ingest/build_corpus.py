"""Build or update the corpus: the ANS norms in force, with their *consolidated* text.

Usage:
    uv run python -m rag.ingest.build_corpus [--kind RN] [--limit N] [--force]

Source of truth: the ANS legislation portal. It lists the norms in force and shows each one as
amended up to today (passages changed by later norms are removed). When the portal has no text
for a norm, the original text is taken from the Diário Oficial da União as a fallback, and the
document is marked ``source: "dou"`` (not consolidated).

The command is incremental and safe to re-run:
- a norm whose "Atualizado em" date did not change is not rewritten;
- a norm that is no longer in force (revoked) is removed from the corpus;
- ``data/corpus/_changes.json`` lists what changed, for ``rag.index --changed``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import httpx

from rag.config import get_settings
from rag.ingest.ans_portal import PortalBrowser, PortalRow
from rag.ingest.dou import DouClient
from rag.ingest.models import NormRef

USER_AGENT = "self-corrective-rag/0.1 (portfolio project; research use)"


def _read(path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write(path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", default="RN", choices=["RN", "IN"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N norms of the list (quick run)")
    parser.add_argument("--force", action="store_true", help="rewrite every norm even if unchanged")
    args = parser.parse_args()

    out_dir = get_settings().corpus_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    urls_path = out_dir / f"_portal_urls_{args.kind.lower()}.json"
    known_urls = {int(k): v for k, v in (_read(urls_path) or {}).items()}

    changed, unchanged, removed, failed = [], [], [], []
    in_force: dict[str, tuple[PortalRow, str | None]] = {}

    with PortalBrowser() as portal, httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120,
                                                 follow_redirects=True) as http:
        dou = DouClient(http)
        for n, (row, url) in enumerate(portal.iter_rows(args.kind, known_urls), 1):
            if args.limit and n > args.limit:
                break
            in_force[row.doc_id] = (row, url)
            if url:
                known_urls[row.number] = url
                _write(urls_path, known_urls)
            path = out_dir / f"{row.doc_id}.json"
            existing = _read(path)
            try:
                text = portal.fetch(url) if url else None
            except Exception as exc:  # noqa: BLE001 - one bad page should not stop the run
                print(f"[{n}] {row.doc_id}: portal error {exc!r}")
                text = None

            portal_updated = text.updated.isoformat() if text and text.updated else None
            # A real norm has articles; the portal also lists a few placeholders (e.g. "norma teste").
            if text and not any(p.startswith("Art") for p in text.paragraphs):
                text = None
            if (existing and not args.force and existing.get("source") == "portal"
                    and existing.get("updated") == portal_updated and text):
                unchanged.append(row.doc_id)
                continue

            ref = NormRef(row.kind, row.number, row.title, row.dou_date, row.ementa)
            doc = {**ref.to_dict(), "doc_id": row.doc_id}
            if text:
                doc |= {"source": "portal", "source_url": url, "updated": portal_updated,
                        "paragraphs": text.paragraphs}
            else:
                article = None
                try:
                    article = dou.find(ref)
                except Exception as exc:  # noqa: BLE001 - the fallback must never stop the run
                    print(f"[{n}] {row.doc_id}: DOU error {exc!r}")
                if article is None:
                    print(f"[{n}] {row.doc_id}: no text on the portal nor in the DOU")
                    failed.append(row.doc_id)
                    continue
                doc |= {"source": "dou", "source_url": article.url, "updated": None,
                        "dou_title": article.title, "paragraphs": article.paragraphs}
            if existing and existing.get("paragraphs") == doc["paragraphs"] and not args.force:
                unchanged.append(row.doc_id)
                _write(path, doc)  # metadata may have changed; the text did not
                continue
            _write(path, doc)
            changed.append(row.doc_id)
            print(f"[{n}] {row.doc_id}: {'updated' if existing else 'new'} "
                  f"({doc['source']}, {len(doc['paragraphs'])} paragraphs)")

        list_complete = portal.list_complete

    on_disk = {p.stem: p for p in out_dir.glob(f"{args.kind.lower()}-*.json")}
    to_remove, reason = revoked_to_remove(set(on_disk), set(in_force), list_complete and not args.limit)
    if reason:
        print(reason)
    for doc_id in to_remove:
        on_disk[doc_id].unlink()
        removed.append(doc_id)
        print(f"{doc_id}: no longer in force, removed")

    changes = {"date": date.today().isoformat(), "kind": args.kind, "in_force": len(in_force),
               "changed": changed, "removed": removed, "unchanged": len(unchanged),
               "failed": list(dict.fromkeys(failed))}
    _write(out_dir / "_changes.json", changes)
    print(f"done: {len(in_force)} in force | {len(changed)} new/updated | {len(unchanged)} unchanged | "
          f"{len(removed)} removed | {len(failed)} without text")


def revoked_to_remove(on_disk: set[str], in_force: set[str], list_complete: bool) -> tuple[list[str], str | None]:
    """Norms on disk that left the in-force list were revoked and should be removed.

    Guard rails, so a broken or partial crawl can never wipe the corpus: nothing is removed unless
    the whole list was read, and never more than 10 norms or 20% of the corpus in one run.
    """
    gone = sorted(on_disk - in_force)
    if not gone:
        return [], None
    if not list_complete:
        return [], "the portal list was not read to the end: nothing removed"
    if len(gone) > max(10, 0.2 * len(on_disk)):
        return [], f"{len(gone)} of {len(on_disk)} norms would be removed: too many, nothing removed (check the crawl)"
    return gone, None


def load_corpus(kind: str | None = None) -> list[dict]:
    """Load the corpus documents (dicts with kind, number, ementa, source_url, paragraphs, ...)."""
    docs = []
    for path in sorted(get_settings().corpus_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if kind is None or doc["kind"] == kind:
            docs.append(doc)
    return docs


if __name__ == "__main__":
    main()
