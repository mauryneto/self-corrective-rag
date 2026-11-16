from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date


@dataclass(frozen=True)
class NormRef:
    """What identifies a norm: kind, number, publication date and summary."""

    kind: str  # "RN", "IN", ...
    number: int
    title: str
    dou_date: date
    ementa: str

    @property
    def doc_id(self) -> str:
        return f"{self.kind.lower()}-{self.number}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dou_date"] = self.dou_date.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> NormRef:
        return cls(**{**d, "dou_date": date.fromisoformat(d["dou_date"])})
