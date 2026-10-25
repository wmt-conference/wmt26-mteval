"""Render the generated documents: ``docs/FORMAT.md`` and ``data/MANIFEST.tsv``.

    uv run python scripts/render_docs.py --grand GRAND [--no-checksum] [--repo REPO]

``docs/FORMAT.md`` comes from the field descriptions in ``wmt26_mteval.schemas``; the manifest
lists the source files recorded in the grand file.  Both are rewritten in REPO (default: the
repository that holds this script), and a test checks that the committed files equal a fresh
render.  The grand file is read only, and checked against its known sha256 unless
``--no-checksum`` is given (for subsets of it, such as a test fixture).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wmt26_mteval.dataset_card import render_format_md, render_manifest_tsv
from wmt26_mteval.grand import GRAND_SHA256, ChecksumError, Grand

REPO = Path(__file__).resolve().parents[1]
FORMAT_PATH = Path("docs") / "FORMAT.md"
MANIFEST_PATH = Path("data") / "MANIFEST.tsv"
LANG_CODES_PATH = Path("data") / "metadata" / "lang_codes.tsv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grand", type=Path, required=True, help="wmt26-grand.parquet")
    parser.add_argument(
        "--repo", type=Path, default=REPO, help="repository to update (default: this one)"
    )
    parser.add_argument(
        "--no-checksum", action="store_true", help="do not check the sha256 of the grand file"
    )
    args = parser.parse_args(argv)

    checksum = None if args.no_checksum else GRAND_SHA256
    try:
        grand = Grand(args.grand, checksum)
        format_md = render_format_md(args.repo / LANG_CODES_PATH)
    except (ChecksumError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for path, text in ((FORMAT_PATH, format_md), (MANIFEST_PATH, render_manifest_tsv(grand))):
        target = args.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
