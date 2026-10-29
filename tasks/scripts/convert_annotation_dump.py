#!/usr/bin/env python3
"""Convert a Pearmut dump into Task 1 human ground-truth annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmt26_mteval_task1.ground_truth_conversion import (
    GroundTruthConversionStats,
    convert_annotation_path,
    item_pair,
    load_dump_directions,
    load_test_set,
    print_recap,
    write_ground_truth,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for source dump, official test set, and destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="Pearmut dump JSON")
    parser.add_argument("test_set", type=Path, help="authoritative test-set JSONL")
    parser.add_argument("destination", type=Path, help="ground-truth annotation JSONL")
    parser.add_argument(
        "--direction", action="append", dest="directions", metavar="SRC-TGT",
        help="direction to retain; repeat for a subset (default: official dump directions)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate selected directions and write human ground-truth annotations."""

    arguments = build_parser().parse_args(argv)
    dump_directions = load_dump_directions(arguments.annotations)
    requested = frozenset(arguments.directions or dump_directions)
    missing_dump = sorted(requested - dump_directions)
    if missing_dump:
        raise ValueError("Selected directions missing from dump: " + ", ".join(missing_dump))

    test_set = load_test_set(arguments.test_set, requested)
    supported = frozenset(item_pair(item_id) for item_id in test_set)
    if arguments.directions:
        missing_test = sorted(requested - supported)
        if missing_test:
            raise ValueError(
                "Selected directions missing from official test set: "
                + ", ".join(missing_test)
            )
        selected = requested
    else:
        selected = supported

    stats = GroundTruthConversionStats(selected_directions=selected)
    screens = convert_annotation_path(
        arguments.annotations, test_set, selected, stats,
    )
    write_ground_truth(screens, arguments.destination)
    print_recap(test_set, screens, stats)
    outside = sorted(requested - supported)
    print("Dump directions outside official test set: " + (
        ", ".join(outside) if outside else "none"
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
