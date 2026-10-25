"""Build the staged contents of the Hugging Face dataset repository from the grand file.

``build_all`` writes one Parquet shard layout per config under ``out_dir/data``, the dataset card
``README.md`` and, last of all, ``build_manifest.json``, which records the checksum of the grand
file, the code version, the checksum and row count of every shard and the statistics that the card
states.  The output is deterministic: the same inputs and the same pyarrow give byte-identical
files.  The grand file itself is never copied.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wmt26_mteval.context import Context
from wmt26_mteval.dataset_card import CARD_TEMPLATE_PATH, card_stats, render_card
from wmt26_mteval.grand import GRAND_SHA256, Grand, sha256_of
from wmt26_mteval.runs import list_runs
from wmt26_mteval.schemas import SCHEMAS
from wmt26_mteval.tables.items import build_items, build_pairs
from wmt26_mteval.tables.metadata import (
    CHALLENGE_SETS_PATH,
    Coverage,
    build_challenge_sets,
    build_mt_systems,
    build_systems,
    coverage_of,
    priority_lps,
)
from wmt26_mteval.tables.predictions import build_run_table

MANIFEST_NAME = "build_manifest.json"
README_NAME = "README.md"
ROW_GROUP_ROWS = 50_000

logger = logging.getLogger(__name__)


class _Writer:
    """Writes the shards of a build and keeps the manifest entry of each, per config."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.files: dict[str, list[dict]] = {config: [] for config in SCHEMAS}

    def write(self, config: str, name: str, table: pa.Table) -> None:
        """Write ``table`` as ``data/<config>/<name>`` and record it; a name used twice raises."""
        if not table.schema.equals(SCHEMAS[config], check_metadata=True):
            raise ValueError(f"{config}/{name}: the table does not have the {config!r} schema")
        relative = Path("data") / config / name
        path = self.out_dir / relative
        if path.exists():
            raise ValueError(f"{relative.as_posix()}: two tables are written to one file")
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table,
            path,
            compression="zstd",
            use_dictionary=True,
            row_group_size=ROW_GROUP_ROWS,
        )
        logger.info("wrote %s (%d rows)", relative.as_posix(), table.num_rows)
        self.files[config].append(
            {"path": relative.as_posix(), "sha256": sha256_of(path), "rows": table.num_rows}
        )

    def configs(self) -> dict[str, dict]:
        """The ``configs`` part of the manifest: each config's total rows and its sorted files."""
        return {
            config: {
                "rows": sum(f["rows"] for f in files),
                "files": sorted(files, key=lambda f: f["path"]),
            }
            for config, files in self.files.items()
        }


def _git_state(repo: Path) -> tuple[str | None, bool | None]:
    """``(HEAD commit, has uncommitted changes)`` of ``repo``; ``(None, None)`` without git.

    Untracked files count as changes, since the build reads files from the checkout.
    """

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        )
        return done.stdout.strip()

    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _check_empty(out_dir: Path) -> None:
    """Raise ``FileExistsError`` unless ``out_dir`` is missing or an empty directory."""
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"{out_dir}: output path exists and is not an empty directory")


def build_all(
    grand_path: Path,
    repo: Path,
    out_dir: Path,
    expected_sha256: str | None = GRAND_SHA256,
) -> dict:
    """Build every config into ``out_dir`` and return the build manifest, as also written to disk.

    ``out_dir`` must be missing or empty (``FileExistsError`` otherwise, before anything is read),
    so that no shard of an earlier build can survive in it.  The files are

    - ``data/items/test-00000.parquet``, and likewise ``systems``, ``mt_systems`` and
      ``challenge_sets``;
    - ``data/pairs/test-<lp>.parquet``, one per language pair, holding that pair's rows;
    - ``data/submissions/test-<task>-<run_id>.parquet``, one per submission;
    - ``data/baselines/test-<run_id>-<variant>.parquet``, one per baseline snapshot;
    - ``README.md``, the dataset card rendered from ``hf/README.template.md`` under ``repo`` and
      the manifest (it is not a shard, so no config lists it);
    - ``build_manifest.json``, written last: a directory without it is an unfinished build.

    ``expected_sha256`` is checked by ``Grand`` (``None`` skips the check).  The manifest's
    ``grand_sha256`` is the checksum of the file that was read: the verified one, or a freshly
    computed one when the check was skipped.  ``git_commit`` and ``git_dirty`` describe ``repo``
    and are null when git cannot tell.  The ``systems`` coverage comes from the ``official`` run
    tables.  Each run table is written and released before the next one is built.

    ``stats`` holds the numbers the card states (see ``dataset_card.card_stats``), counted from
    the tables just built, the total rows of every config and the rows and runs of the ``alt``
    baseline snapshots.
    """
    out_dir = Path(out_dir)
    _check_empty(out_dir)
    template = Path(repo) / CARD_TEMPLATE_PATH
    if not template.is_file():  # fail now, not after the shards are written
        raise FileNotFoundError(f"{template}: the dataset card template does not exist")
    grand = Grand(grand_path, expected_sha256)
    grand_sha256 = expected_sha256 if expected_sha256 is not None else sha256_of(grand.path)
    git_commit, git_dirty = _git_state(repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = _Writer(out_dir)
    ctx = Context.from_repo(repo)

    items = build_items(grand, ctx)
    writer.write("items", "test-00000.parquet", items)

    pairs = build_pairs(grand, ctx)
    stats = card_stats(items, pairs)
    for lp in sorted(pc.unique(pairs.column("lp")).to_pylist()):
        writer.write("pairs", f"test-{lp}.parquet", pairs.filter(pc.equal(pairs.column("lp"), lp)))
    writer.write("mt_systems", "test-00000.parquet", build_mt_systems(pairs))
    challenge_sets = build_challenge_sets(Path(repo) / CHALLENGE_SETS_PATH, pairs)
    writer.write("challenge_sets", "test-00000.parquet", challenge_sets)
    del items, pairs, challenge_sets

    runs = list_runs(grand)
    priority = priority_lps(ctx)
    coverage: dict[str, Coverage] = {}
    alt_runs = alt_rows = 0
    for run in runs:
        table = build_run_table(grand, run, ctx)
        if run.variant == "official":
            coverage[run.run_id] = coverage_of(table, priority)
        else:
            alt_runs, alt_rows = alt_runs + 1, alt_rows + table.num_rows
        if run.kind == "submission":
            writer.write("submissions", f"test-{run.task}-{run.run_id}.parquet", table)
        else:
            writer.write("baselines", f"test-{run.run_id}-{run.variant}.parquet", table)
        del table

    writer.write("systems", "test-00000.parquet", build_systems(repo, runs, coverage))

    configs = writer.configs()
    manifest = {
        "grand_sha256": grand_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "pyarrow_version": pa.__version__,
        "configs": configs,
        "stats": {
            "rows": {config: info["rows"] for config, info in configs.items()},
            **stats,
            "baseline_alt_rows": alt_rows,
            "baseline_alt_runs": alt_runs,
        },
    }
    card = render_card(template, manifest)
    (out_dir / README_NAME).write_text(card, encoding="utf-8")
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
