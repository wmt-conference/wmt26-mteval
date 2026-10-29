#!/usr/bin/env python3
"""Rebuild the Task 2 input folders from the integrated grand parquet.

The Hugging Face release ships ``wmt26-grand.parquet`` but neither the original Task 2 folders
nor the raw Pearmut dump. This script writes, below an output root (the repository root by
default), the files ``README_TASK2.md`` reads: the test set, the converted human ground truth,
the direct Task 2 submissions and baselines, and the Task 1 submissions and baseline snapshots
of the conversion track. Each run folder also gets its public ``model_card.json``.

Records are reconstructed in original line order with the reader Task 3 uses, so the files
Task 3 also reads get the bytes ``parquet_inputs.json`` pins. Nothing else is read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmt26_mteval_task3.parquet_source import Task3ParquetSource

# Source-manifest namespaces read by the Task 2 direct and conversion tracks.
NAMESPACES = frozenset({
    "test_set", "ground_truth", "task2",
    "task1", "task1_deduplicated", "task1_prededup_and_enhy_fix",
})
# Top-level folders this script creates. It writes nothing if any of them already exists.
TARGETS = ("test-set-files", "human_annotations", "submissions", "baselines")


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the grand parquet, the public model cards and the output root."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--grand", type=Path, default=Path("wmt26-grand.parquet"),
        help="integrated grand parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--model-cards", type=Path, default=Path("model_cards"),
        help="public model-card root with task1/ and task2/ (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("."),
        help="root to write the Task 2 input folders into (default: %(default)s)",
    )
    return parser


def stage(grand: Path, cards: Path, out: Path) -> dict[str, int]:
    """Write every Task 2 input file below ``out`` and return the records written per file.

    Input: the grand parquet, the public model-card root and the output root. Output: the
    relative path of each reconstructed JSONL file mapped to its record count. Raises
    ``FileExistsError`` before reading anything if a target folder already exists, ``ValueError``
    if a file's record count differs from the parquet's source manifest, and
    ``FileNotFoundError`` if a run has no public model card.
    """

    existing = [name for name in TARGETS if (out / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to write into existing folders: {', '.join(existing)}")
    source = Task3ParquetSource(grand, cards)
    counts = {}
    for logical, entry in sorted(source.entries.items()):
        if entry["task_namespace"] not in NAMESPACES:
            continue
        path = out / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("wb") as handle:
            for _, record in source.iter_records(logical):
                handle.write((json.dumps(record, ensure_ascii=False, allow_nan=False,
                                         separators=(",", ":")) + "\n").encode("utf-8"))
                count += 1
        if count != entry["records"]:
            raise ValueError(f"{logical}: {count} records written, the manifest says {entry['records']}")
        if entry["kind"] in {"submission", "baseline"}:
            task = "task2" if entry["task_namespace"] == "task2" else "task1"
            card = cards / task / logical.parent / "model_card.json"
            if not card.is_file():
                raise FileNotFoundError(f"{card}: public model card is missing")
            shutil.copyfile(card, path.with_name("model_card.json"))
        counts[logical.as_posix()] = count
        print(f"Reconstructed {logical.as_posix()}: {count} records", flush=True)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Stage the Task 2 inputs and print how many files and records were written."""

    arguments = build_parser().parse_args(argv)
    counts = stage(arguments.grand, arguments.model_cards, arguments.out)
    print(f"Wrote {len(counts)} files with {sum(counts.values())} records below {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
