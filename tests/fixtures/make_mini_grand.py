"""Build ``mini_grand.parquet``: a 16-row, real-row subset of the WMT26 grand file.

Usage::

    uv run python tests/fixtures/make_mini_grand.py GRAND OUT

``GRAND`` is the read-only ``wmt26-grand.parquet``; ``OUT`` is the fixture to write. Each of the
cases in ``CASES`` selects the first item (in ``item_id`` order) that matches it and has not
already been taken by an earlier case, so the fixture has exactly one row per case. The rows are
copied verbatim: ``pq.write_table(table.take(idx), OUT)`` keeps all 107 columns, their Arrow
types (``arrow.json`` extension columns included) and the schema-level metadata.

Item ids have six ``_###_``-separated parts::

    <collection>_###_<src>_###_<tgt>_###_<field4>_###_<doc_id>_###_<para_idx>
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SEP = "_###_"
SUBMISSION_875921 = "submission__task1__875921__predictions"
PD = "RefQuality-PD_"
PE = "RefQuality-PE_"

# A predicate looks at the six parts of an item id and at the item's row (a dict of raw cells;
# JSON columns are strings, or None for a null cell).
Predicate = Callable[[list[str], dict], bool]


def _json(cell):
    return None if cell is None else json.loads(cell)


def route(collection: str, src: str, tgt: str, field4: str | None = None) -> Predicate:
    """Match on collection, language pair and (optionally) the fourth id field."""

    def match(parts: list[str], row: dict) -> bool:
        return parts[:3] == [collection, src, tgt] and (field4 is None or parts[3] == field4)

    return match


def with_resource(predicate: Predicate, key: str) -> Predicate:
    def match(parts: list[str], row: dict) -> bool:
        return predicate(parts, row) and (_json(row["resources"]) or {}).get(key) is not None

    return match


def with_empty_hyp(predicate: Predicate) -> Predicate:
    def match(parts: list[str], row: dict) -> bool:
        return predicate(parts, row) and "" in (_json(row["hyps"]) or {}).values()

    return match


def doc_prefix(prefix: str, *, excluding: str | None = None) -> Predicate:
    """Match on the start of the doc id; ``excluding`` keeps nested prefixes disjoint."""

    def match(parts: list[str], row: dict) -> bool:
        doc = parts[4]
        return doc.startswith(prefix) and not (excluding and doc.startswith(excluding))

    return match


def covered_by(column: str) -> Predicate:
    def match(parts: list[str], row: dict) -> bool:
        return row[column] is not None

    return match


def official(src: str, tgt: str, field4: str | None = None) -> Predicate:
    return route("official", src, tgt, field4)


def challenge(src: str, tgt: str, field4: str) -> Predicate:
    return route("challenge", src, tgt, field4)


CASES: list[tuple[str, Predicate]] = [
    (
        "1 official ces-deu speech with resources.video",
        with_resource(official("ces_Latn", "deu_Latn", "speech"), "video"),
    ),
    ("2 official ces-ukr (ref text null)", official("ces_Latn", "ukr_Cyrl")),
    ("3 official eng-arz (postedit)", official("eng_Latn", "arz_Arab")),
    (
        "4 official eng-zho_Hans with resources.screenshot",
        with_resource(official("eng_Latn", "zho_Hans"), "screenshot"),
    ),
    ("5 official eng-zho_Hant_TW", official("eng_Latn", "zho_Hant_TW")),
    ("6 official zho_Hans-jpn", official("zho_Hans", "jpn_Jpan")),
    (
        "7 official item with an empty-string hyp",
        with_empty_hyp(lambda parts, row: parts[0] == "official"),
    ),
    ("8 GAMBITP (no ref)", lambda parts, row: parts[3] == "GAMBITP"),
    ("9 FECE en-de_DE", challenge("en", "de_DE", "FECE")),
    ("10 QEbreak cs-uk", challenge("cs", "uk", "QEbreak")),
    ("11 AGGREE ja-zh", challenge("ja", "zh", "AGGREE")),
    ("12 LIGATUR en-hi", challenge("en", "hi", "LIGATUR")),
    ("13 doc prefix RefQuality-PE_", doc_prefix(PE)),
    ("14 doc prefix RefQuality-PD_", doc_prefix(PD, excluding=PD + PE)),
    ("15 doc prefix RefQuality-PD_RefQuality-PE_", doc_prefix(PD + PE)),
    (
        "16 covered by submission 875921 (PragmaSpan, partial coverage)",
        covered_by(SUBMISSION_875921),
    ),
]


def select(table: pa.Table) -> list[tuple[str, int]]:
    """Return ``(case label, row index)`` per case, first unused match in ``item_id`` order."""
    ids = table.column("item_id").to_pylist()
    needed = ("ref", "hyps", "resources", SUBMISSION_875921)
    cells = {name: table.column(name).to_pylist() for name in needed}
    order = sorted(range(len(ids)), key=ids.__getitem__)
    taken: set[int] = set()
    chosen: list[tuple[str, int]] = []
    for label, predicate in CASES:
        for i in order:
            if i in taken:
                continue
            parts = ids[i].split(SEP)
            if len(parts) == 6 and predicate(parts, {name: col[i] for name, col in cells.items()}):
                taken.add(i)
                chosen.append((label, i))
                break
        else:
            raise SystemExit(f"no unused item matches case: {label}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("grand", type=Path, help="path to wmt26-grand.parquet (read-only)")
    parser.add_argument("out", type=Path, help="fixture Parquet file to write")
    args = parser.parse_args(argv)

    table = pq.read_table(args.grand)
    chosen = select(table)
    ids = table.column("item_id").to_pylist()
    idx = sorted((i for _, i in chosen), key=ids.__getitem__)  # keep item_id order
    assert len(idx) == len(CASES) == 16 and len(set(idx)) == 16

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(idx), args.out)

    out_schema = pq.read_schema(args.out)
    if not out_schema.equals(table.schema, check_metadata=True):
        raise SystemExit("schema or schema metadata differs from the grand file")

    for label, i in chosen:
        print(f"{label}\n    {ids[i]}")
    size = args.out.stat().st_size
    print(
        f"wrote {args.out}: {len(idx)} rows, {len(out_schema.names)} columns, {size:,} bytes; "
        "schema and metadata match the grand file"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
