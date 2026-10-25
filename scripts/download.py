"""Download one config of the WMT26 MT-Eval data from the Hugging Face Hub.

    uv run python scripts/download.py --config pairs --out data/hf [--revision main]

Fetches only the files of the chosen config (``grand`` is the single file ``wmt26-grand.parquet``)
from the dataset ``wmt/wmt26-mteval-data`` into OUT, with ``huggingface_hub.snapshot_download``.
A token in the environment or in the Hugging Face cache is used if there is one.  Besides
``huggingface_hub`` the script needs nothing, so it also runs outside this project.  The configs
are described in docs/FORMAT.md; ``submissions`` and ``baselines`` are by far the largest, with
millions of rows each.
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

REPO_ID = "wmt/wmt26-mteval-data"
GRAND_FILE = "wmt26-grand.parquet"
# The configs of the repository.  This script does not import wmt26_mteval, so the names are
# repeated here; a test checks them against ``schemas.SCHEMAS``.
CONFIGS = (
    "items",
    "pairs",
    "submissions",
    "baselines",
    "systems",
    "mt_systems",
    "challenge_sets",
    "grand",
)


def allow_patterns(config: str) -> list[str]:
    """The file patterns of ``config`` in the repository; an unknown config raises ValueError."""
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; choose one of {', '.join(CONFIGS)}")
    return [GRAND_FILE] if config == "grand" else [f"data/{config}/*"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, choices=CONFIGS, help="the config to download")
    parser.add_argument("--out", required=True, help="folder to download into")
    parser.add_argument("--revision", default="main", help="branch, tag or commit (default: main)")
    args = parser.parse_args(argv)

    folder = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=allow_patterns(args.config),
        local_dir=args.out,
    )
    print(f"downloaded {args.config} to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
