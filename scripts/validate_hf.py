"""Run the validation gates on the staged contents of the Hugging Face repository.

    uv run python scripts/validate_hf.py --out OUT --grand GRAND --expected EXPECTED [--no-checksum]

Checks that the card and the manifest exist, the row counts, Table 2, personal data, the schemas,
the keys that link the configs, the texts of ``items`` and ``pairs`` and a round trip of every run
against the grand file, and that the Task 2 scores dropped from the ``alt`` baselines are those of
the official ones (see ``wmt26_mteval.validate``).  Prints each failure and then the number of
failures, and exits 1 if there is any.  OUT and GRAND are only read.  The grand file is checked
against its known sha256 unless ``--no-checksum`` is given (for subsets of it, such as a test
fixture).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wmt26_mteval.grand import GRAND_SHA256, ChecksumError, Grand
from wmt26_mteval.validate import validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="staged output of build_hf.py")
    parser.add_argument("--grand", type=Path, required=True, help="wmt26-grand.parquet")
    parser.add_argument(
        "--expected", type=Path, required=True, help="JSON with the expected row counts"
    )
    parser.add_argument(
        "--no-checksum", action="store_true", help="do not check the sha256 of the grand file"
    )
    args = parser.parse_args(argv)

    checksum = None if args.no_checksum else GRAND_SHA256
    try:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        grand = Grand(args.grand, checksum)
    except (ChecksumError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    failures = validate(args.out, grand, expected)
    for failure in failures:
        print(failure)
    print(f"{len(failures)} failure{'' if len(failures) == 1 else 's'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
