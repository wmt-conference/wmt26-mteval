"""Write the allow-list scrubbed model card of every official run.

    uv run python scripts/scrub_model_cards.py --raw RAW --grand GRAND --out data/model_cards

Only the allow-listed answers (``wmt26_mteval.cards.CARD_FIELDS``) reach the output.  The report
names each run and field whose answer was altered (an e-mail address removed, or personal names
replaced), never the answer itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wmt26_mteval.cards import scrub_runs
from wmt26_mteval.grand import Grand
from wmt26_mteval.runs import list_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="organiser drop (the raw folder)")
    parser.add_argument("--grand", type=Path, required=True, help="wmt26-grand.parquet")
    parser.add_argument("--out", type=Path, required=True, help="folder for <run_id>.json files")
    args = parser.parse_args(argv)

    runs = [run for run in list_runs(Grand(args.grand)) if run.variant == "official"]
    changes = scrub_runs(args.raw, runs, args.out)
    print(f"wrote {len(runs)} scrubbed model cards to {args.out}")
    for run_id, key, change in changes:
        print(f"{change}: run {run_id}, field {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
