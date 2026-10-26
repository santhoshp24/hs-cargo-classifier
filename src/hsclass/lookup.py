"""
Two-file lookup, keyed by the normalised description.

* curated_lookup.csv  - maintained by people. The pipeline NEVER writes to it. Wins on conflicts.
* learned_lookup.csv  - written by the pipeline when a provider answers with enough confidence.

Keeping them apart means automatic learning can never silently overwrite human decisions, and the
learned file can be reviewed (and promoted into the curated file) at any time.
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from .hs_reference import VALID_CHAPTERS
from .normalise import normalise

LEARNED_FIELDS = ["key", "hs_chapter", "source", "confidence", "added_on"]
CURATED_FIELDS = ["description", "hs_chapter"]


@dataclass(frozen=True)
class Hit:
    chapter: int
    tier: str  # "curated" or "learned"


class Lookup:
    def __init__(self, curated_path: Path | None, learned_path: Path | None):
        self.curated_path = Path(curated_path) if curated_path else None
        self.learned_path = Path(learned_path) if learned_path else None
        self._curated: dict[str, int] = {}
        self._learned: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self.curated_path and self.curated_path.exists():
            with open(self.curated_path, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    key = normalise(row.get("description", ""))
                    ch = _as_chapter(row.get("hs_chapter"))
                    if key and ch is not None:
                        self._curated[key] = ch
        if self.learned_path and self.learned_path.exists():
            with open(self.learned_path, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    ch = _as_chapter(row.get("hs_chapter"))
                    key = row.get("key", "")
                    if key and ch is not None:
                        self._learned[key] = row

    # ------------------------------------------------------------------ queries
    def get(self, key: str) -> Hit | None:
        if key in self._curated:
            return Hit(self._curated[key], "curated")
        row = self._learned.get(key)
        if row is not None:
            return Hit(int(row["hs_chapter"]), "learned")
        return None

    def __len__(self) -> int:
        return len(set(self._curated) | set(self._learned))

    @property
    def sizes(self) -> dict[str, int]:
        return {"curated": len(self._curated), "learned": len(self._learned)}

    # ------------------------------------------------------------------ learning
    def learn(self, key: str, chapter: int, source: str, confidence: float) -> bool:
        """Remember an answer. Returns False if it was refused or already known."""
        if not key or chapter not in VALID_CHAPTERS:
            return False
        if key in self._curated or key in self._learned:
            return False
        self._learned[key] = {
            "key": key,
            "hs_chapter": str(chapter),
            "source": source,
            "confidence": f"{confidence:.2f}",
            "added_on": dt.date.today().isoformat(),
        }
        return True

    def add_curated(self, description: str, chapter: int) -> bool:
        """Human-reviewed answer (used by the `learn` command). Updates the curated file in memory."""
        key = normalise(description)
        if not key or chapter not in VALID_CHAPTERS:
            return False
        self._curated[key] = chapter
        self._learned.pop(key, None)
        return True

    # ------------------------------------------------------------------ saving
    def save_learned(self) -> None:
        if not self.learned_path:
            return
        self.learned_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.learned_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=LEARNED_FIELDS)
            w.writeheader()
            for key in sorted(self._learned):
                w.writerow({k: self._learned[key].get(k, "") for k in LEARNED_FIELDS})

    def append_curated(self, rows: list[tuple[str, int]]) -> None:
        """Append reviewed rows to the curated CSV on disk (human-approved path only)."""
        if not self.curated_path:
            raise ValueError("no curated path configured")
        self.curated_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.curated_path.exists()
        with open(self.curated_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(CURATED_FIELDS)
            for desc, ch in rows:
                w.writerow([desc, ch])


def _as_chapter(value) -> int | None:
    try:
        ch = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return ch if ch in VALID_CHAPTERS else None
