import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from wmt26_mteval.context import Context
from wmt26_mteval.grand import Grand
from wmt26_mteval.runs import list_runs
from wmt26_mteval.tables.items import build_pairs
from wmt26_mteval.tables.metadata import (
    CHALLENGE_SETS_PATH,
    build_systems,
    coverage_of,
    priority_lps,
)
from wmt26_mteval.tables.predictions import (
    PREDICTION_KEYS,
    T1,
    build_run_table,
    normalise_raw_record,
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Root of the repository checkout (the directory holding ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def mini_grand_path(repo_root: Path) -> Path:
    """Path to the 16-row Parquet subset of the grand file (see ``make_mini_grand.py``)."""
    return repo_root / "tests" / "fixtures" / "mini_grand.parquet"


@pytest.fixture(scope="session")
def ctx(repo_root: Path) -> Context:
    """Language normalisation and priority-direction context built from the repository data."""
    return Context.from_repo(repo_root)


@pytest.fixture(scope="session")
def mini_grand(mini_grand_path: Path) -> Grand:
    """The 16-row mini grand file, opened without its checksum (it is a subset)."""
    return Grand(mini_grand_path, expected_sha256=None)


@pytest.fixture(scope="session")
def mini_pairs(mini_grand: Grand, ctx: Context):
    """The ``pairs`` table of the mini grand file."""
    return build_pairs(mini_grand, ctx)


@pytest.fixture(scope="session")
def mini_systems(repo_root: Path, mini_grand: Grand, ctx: Context):
    """The ``systems`` table for the mini grand file's runs and the committed scrubbed cards."""
    runs = list_runs(mini_grand)
    priority = priority_lps(ctx)
    coverage = {
        run.run_id: coverage_of(build_run_table(mini_grand, run, ctx), priority)
        for run in runs
        if run.variant == "official"
    }
    return build_systems(repo_root, runs, coverage)


def _env_path(name: str, what: str) -> Path:
    """The path in environment variable ``name``; skip the test when it is not set."""
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set to {what}")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{name}={value} does not exist")
    return path


@pytest.fixture(scope="session")
def full_out() -> Path:
    """The staged output built from the real grand file: ``OUT=/path/to/build/hf``."""
    path = _env_path("OUT", "a build of the real wmt26-grand.parquet")
    assert path.is_dir(), f"OUT={path} is not a directory"
    return path


@pytest.fixture(scope="session")
def full_grand() -> Grand:
    """The real grand file, checksum verified: ``GRAND=/path/to/wmt26-grand.parquet``."""
    return Grand(_env_path("GRAND", "the real wmt26-grand.parquet"))


@pytest.fixture(scope="session")
def expected_full(repo_root: Path) -> dict:
    """The row counts and switches that the validation gates expect of the real release."""
    return json.loads((repo_root / "tests" / "data" / "expected_full.json").read_text("utf-8"))


@pytest.fixture(scope="session")
def expected_mini(repo_root: Path, mini_grand: Grand, mini_grand_path: Path, mini_pairs) -> dict:
    """What ``validate`` expects of a build of the mini grand file, counted from the fixture.

    The run rows are counted from the raw records (one row per system found in any prediction
    dict of the run's tasks), not from the tables, so that the counts do not share the code
    under test.
    """
    runs = list_runs(mini_grand)

    def run_rows(run) -> int:
        tasks = (T1,) if run.variant == "alt" else PREDICTION_KEYS
        total = 0
        for _, raw in mini_grand.iter_records(run.column):
            record = normalise_raw_record(raw)
            total += len(set().union(*(record.get(task, {}).keys() for task in tasks)))
        return total

    challenge_sets = yaml.safe_load((repo_root / CHALLENGE_SETS_PATH).read_text("utf-8"))
    collections = Counter(mini_pairs.column("collection").to_pylist())
    return {
        "check_table2": False,
        "rows": {
            "items": pq.ParquetFile(mini_grand_path).metadata.num_rows,
            "pairs": mini_pairs.num_rows,
            "submissions": sum(run_rows(r) for r in runs if r.kind == "submission"),
            "baselines": sum(run_rows(r) for r in runs if r.kind == "baseline"),
            "systems": sum(r.variant == "official" for r in runs),
            "mt_systems": len(set(mini_pairs.column("system").to_pylist())),
            "challenge_sets": len(challenge_sets),
        },
        "pairs_by_collection": dict(collections),
    }
