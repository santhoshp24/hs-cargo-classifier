"""Command line: generate | classify | evaluate | learn."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .generate import generate
from .hs_reference import VALID_CHAPTERS
from .lookup import Lookup
from .pipeline import REVIEW, classify_rows
from .providers import build_providers


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------ commands

def cmd_generate(a) -> int:
    info = generate(Path(a.out), rows=a.rows, seed=a.seed)
    print(f"Wrote synthetic data to {a.out}: {info}")
    return 0


def cmd_classify(a) -> int:
    rows = _read_csv(Path(a.input))
    if not rows:
        print("Input file has no rows.")
        return 1
    if a.description_col not in rows[0]:
        print(f"Column '{a.description_col}' not found. Columns: {', '.join(rows[0])}")
        return 1
    providers = build_providers(a.providers)
    lookup = Lookup(Path(a.curated), Path(a.learned))
    print(f"Lookup: {lookup.sizes['curated']} curated + {lookup.sizes['learned']} learned entries. "
          f"Providers: {', '.join(p.name for p in providers)}. Rows: {len(rows):,}")

    results, stats = classify_rows(
        rows, a.description_col, a.id_col if a.id_col in rows[0] else None, lookup, providers,
        min_confidence=a.min_confidence, batch_size=a.batch_size, learn=not a.no_learn, log=print)

    out = Path(a.out)
    fields = ["record_id", "description", "key", "hs_chapter", "chapter_title", "cargo_category",
              "edible", "method", "confidence"]
    _write_csv(out / "classified.csv", [r.as_row() for r in results], fields)

    # Review queue: one line per unique unresolved description, ready for a person to fill in.
    queue: dict[str, dict] = {}
    for r in results:
        if r.method != REVIEW:
            continue
        q = queue.setdefault(r.key, {"description": r.description, "occurrences": 0,
                                     "candidate_chapter": r.candidate or "",
                                     "candidate_confidence": "" if r.confidence is None else f"{r.confidence:.2f}",
                                     "hs_chapter": ""})
        q["occurrences"] += 1
    _write_csv(out / "review_queue.csv", sorted(queue.values(), key=lambda q: -q["occurrences"]),
               ["description", "occurrences", "candidate_chapter", "candidate_confidence", "hs_chapter"])
    (out / "run_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    n = stats["rows"] or 1
    by = stats["rows_by_method"]
    print("\nRun summary")
    print(f"  rows                    {stats['rows']:,}")
    print(f"  unique descriptions     {stats['unique_descriptions']:,}")
    print(f"  resolved by lookup      {stats['unique_resolved_by_lookup']:,} unique "
          f"(no provider call needed)")
    for name, c in stats["provider_resolved"].items():
        print(f"  resolved by {name:<10}  {c:,} unique  ({stats['provider_calls'][name]} call(s), "
              f"{stats['provider_errors'][name]} failed)")
    print(f"  sent to review          {stats['unique_in_review']:,} unique")
    print(f"  new lookup entries      {stats['learned_new']:,}")
    print("  rows by method          " + ", ".join(f"{m}={c:,} ({c / n:.0%})" for m, c in sorted(by.items())))
    print(f"\nWrote {out / 'classified.csv'}, {out / 'review_queue.csv'}, {out / 'run_stats.json'}")
    return 0


def cmd_evaluate(a) -> int:
    truth = {r["record_id"]: r[a.truth_col] for r in _read_csv(Path(a.truth)) if a.truth_col in r}
    classified = _read_csv(Path(a.classified))
    total = right = answered = 0
    per_method: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # method -> [answered, correct]
    errors: Counter = Counter()
    for r in classified:
        t = truth.get(r["record_id"])
        if t is None:
            continue
        total += 1
        if r["method"] == REVIEW or not r["hs_chapter"]:
            continue
        answered += 1
        ok = int(r["hs_chapter"]) == int(t)
        right += ok
        per_method[r["method"]][0] += 1
        per_method[r["method"]][1] += ok
        if not ok:
            errors[(int(t), int(r["hs_chapter"]))] += 1
    if not total:
        print("No overlapping record_ids between the files.")
        return 1
    print(f"Records compared: {total:,}")
    print(f"Coverage (answered, not sent to review): {answered / total:.1%}")
    print(f"Accuracy on answered records:            {right / max(answered, 1):.1%}")
    print("\nBy method (answered / accuracy):")
    for m, (n, c) in sorted(per_method.items()):
        print(f"  {m:<16} {n:>6,}  {c / n:.1%}")
    if errors:
        print("\nMost common mistakes (true chapter -> predicted chapter):")
        for (t, p), c in errors.most_common(5):
            print(f"  {t:02d} -> {p:02d}   x{c}")
    return 0


def cmd_learn(a) -> int:
    lookup = Lookup(Path(a.curated), None)
    new_rows, skipped = [], 0
    for r in _read_csv(Path(a.reviewed)):
        raw = (r.get("hs_chapter") or "").strip()
        if not raw:
            continue
        try:
            ch = int(raw)
        except ValueError:
            skipped += 1
            continue
        if ch not in VALID_CHAPTERS or not lookup.add_curated(r.get("description", ""), ch):
            skipped += 1
            continue
        new_rows.append((r["description"], ch))
    if new_rows:
        lookup.append_curated(new_rows)
    print(f"Added {len(new_rows)} reviewed description(s) to {a.curated}; skipped {skipped} invalid row(s).")
    return 0


# ------------------------------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hsclass", description="Tiered HS-chapter classification demo.")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create synthetic cargo data, a curated seed and a chapter dimension")
    g.add_argument("--out", default="data")
    g.add_argument("--rows", type=int, default=2000)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("classify", help="classify a CSV of cargo descriptions")
    c.add_argument("--input", default="data/sample_cargo.csv")
    c.add_argument("--description-col", default="cargo_description")
    c.add_argument("--id-col", default="record_id")
    c.add_argument("--curated", default="data/curated_lookup.csv")
    c.add_argument("--learned", default="data/learned_lookup.csv")
    c.add_argument("--out", default="output")
    c.add_argument("--providers", default="auto",
                   help="auto | comma list of anthropic,groq,heuristic (order = fallback order)")
    c.add_argument("--min-confidence", type=float, default=0.75)
    c.add_argument("--batch-size", type=int, default=25)
    c.add_argument("--no-learn", action="store_true", help="do not write to the learned lookup")
    c.set_defaults(func=cmd_classify)

    e = sub.add_parser("evaluate", help="compare classified.csv with the synthetic ground truth")
    e.add_argument("--classified", default="output/classified.csv")
    e.add_argument("--truth", default="data/sample_cargo.csv")
    e.add_argument("--truth-col", default="true_hs_chapter")
    e.set_defaults(func=cmd_evaluate)

    l = sub.add_parser("learn", help="promote a reviewed review_queue.csv into the curated lookup")
    l.add_argument("--reviewed", default="output/review_queue.csv")
    l.add_argument("--curated", default="data/curated_lookup.csv")
    l.set_defaults(func=cmd_learn)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
