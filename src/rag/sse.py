"""Server-Sent Events, both ends: the API writes them, the demo (or any client) reads them.

One event on the wire::

    event: text
    data: {"text": "O prazo"}

followed by a blank line. ``data`` is always one line of JSON.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator


def format_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def parse_events(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """``(event, data)`` for each event in a stream read line by line (``httpx``'s ``iter_lines``)."""
    event, data = "message", ""
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data += line[5:].strip()
        elif not line and data:
            yield event, json.loads(data)
            event, data = "message", ""
