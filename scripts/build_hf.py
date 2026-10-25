"""Build the staged contents of the Hugging Face dataset repository.

    uv run python scripts/build_hf.py --grand GRAND --out OUT [--no-checksum]

Writes the Parquet shards, the dataset card ``README.md`` and ``build_manifest.json`` under OUT,
which must be missing or empty.  The grand file is read only, and checked against its known sha256
unless ``--no-checksum`` is given (for subsets of it, such as a test fixture).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from wmt26_mteval.build import build_all
from wmt26_mteval.grand import GRAND_SHA256, ChecksumError

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grand", type=Path, required=True, help="wmt26-grand.parquet")
    parser.add_argument("--out", type=Path, required=True, help="empty or missing output folder")
    parser.add_argument(
        "--no-checksum", action="store_true", help="do not check the sha256 of the grand file"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    expected = None if args.no_checksum else GRAND_SHA256
    try:
        manifest = build_all(args.grand, REPO, args.out, expected_sha256=expected)
    except (ChecksumError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"built {args.out} from grand sha256 {manifest['grand_sha256']}")
    for config, info in manifest["configs"].items():
        print(f"{config}: {info['rows']} rows in {len(info['files'])} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
