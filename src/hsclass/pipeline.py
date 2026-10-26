"""
The tiered classification flow.

    1. normalise the description into a bag-of-words key
    2. curated lookup, then learned lookup     -> instant, free, deterministic
    3. providers in order (e.g. Claude, Groq)  -> only for keys the lookups do not know
         confidence >= threshold : accepted and written to the learned lookup
         confidence <  threshold : kept as a candidate, next provider gets a try
    4. nothing confident enough                -> review queue (a person decides)

Work is done per UNIQUE key, so a description seen 500 times costs at most one provider answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .hs_reference import category_for, chapter_title, is_edible
from .lookup import Lookup
from .normalise import normalise
from .providers import ProviderError

REVIEW = "review"


@dataclass
class Result:
    record_id: str
    description: str
    key: str
    chapter: int | None
    method: str          # lookup_curated | lookup_learned | <provider name> | review
    confidence: float | None
    candidate: int | None = None  # best low-confidence guess, shown to the reviewer

    def as_row(self) -> dict:
        ch = self.chapter
        return {
            "record_id": self.record_id,
            "description": self.description,
            "key": self.key,
            "hs_chapter": "" if ch is None else ch,
            "chapter_title": chapter_title(ch) if ch else "",
            "cargo_category": category_for(ch) if ch else "Needs review",
            "edible": ("Edible" if is_edible(ch) else "Non-Edible") if ch else "",
            "method": self.method,
            "confidence": "" if self.confidence is None else f"{self.confidence:.2f}",
        }


def classify_rows(
    rows: list[dict],
    description_col: str,
    id_col: str | None,
    lookup: Lookup,
    providers: list,
    min_confidence: float = 0.75,
    batch_size: int = 25,
    learn: bool = True,
    log: Callable[[str], None] = lambda _msg: None,
) -> tuple[list[Result], dict]:
    keys: list[str] = []
    sample_text: dict[str, str] = {}
    for row in rows:
        text = str(row.get(description_col, "") or "")
        key = normalise(text)
        keys.append(key)
        if key and key not in sample_text:
            sample_text[key] = text

    unique = list(sample_text)
    resolved: dict[str, tuple[int, str, float | None]] = {}
    for key in unique:
        hit = lookup.get(key)
        if hit:
            resolved[key] = (hit.chapter, f"lookup_{hit.tier}", None)

    stats: dict = {
        "rows": len(rows),
        "unique_descriptions": len(unique),
        "unique_resolved_by_lookup": len(resolved),
        "provider_calls": {}, "provider_errors": {}, "provider_resolved": {},
        "learned_new": 0,
    }

    candidates: dict[str, tuple[int, float]] = {}
    for provider in providers:
        pending = [k for k in unique if k not in resolved]
        if not pending:
            break
        log(f"  {provider.name}: {len(pending)} description(s) to classify")
        stats["provider_calls"][provider.name] = 0
        stats["provider_errors"][provider.name] = 0
        stats["provider_resolved"][provider.name] = 0
        for i in range(0, len(pending), batch_size):
            batch_keys = pending[i:i + batch_size]
            batch = [(k, sample_text[k]) for k in batch_keys]
            stats["provider_calls"][provider.name] += 1
            try:
                answers = provider.classify(batch)
            except ProviderError as exc:
                stats["provider_errors"][provider.name] += 1
                log(f"  {provider.name}: batch failed ({exc}); passing it to the next tier")
                continue
            for key, (chapter, conf) in answers.items():
                if key not in sample_text or key in resolved:
                    continue
                if conf >= min_confidence:
                    resolved[key] = (chapter, provider.name, conf)
                    stats["provider_resolved"][provider.name] += 1
                    if learn and lookup.learn(key, chapter, provider.name, conf):
                        stats["learned_new"] += 1
                elif key not in candidates or conf > candidates[key][1]:
                    candidates[key] = (chapter, conf)

    results: list[Result] = []
    for idx, (row, key) in enumerate(zip(rows, keys), start=1):
        rid = str(row.get(id_col, "")) if id_col else str(idx)
        text = str(row.get(description_col, "") or "")
        if key in resolved:
            ch, method, conf = resolved[key]
            results.append(Result(rid, text, key, ch, method, conf))
        else:
            cand = candidates.get(key)
            results.append(Result(rid, text, key, None, REVIEW,
                                  cand[1] if cand else None, cand[0] if cand else None))

    by_method: dict[str, int] = {}
    for r in results:
        by_method[r.method] = by_method.get(r.method, 0) + 1
    stats["rows_by_method"] = by_method
    stats["unique_in_review"] = len({r.key for r in results if r.method == REVIEW})
    stats["lookup_sizes"] = lookup.sizes
    if learn:
        lookup.save_learned()
    return results, stats
