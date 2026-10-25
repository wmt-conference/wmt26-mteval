from pathlib import Path

import pytest

from wmt26_mteval.context import Context
from wmt26_mteval.grand import Grand
from wmt26_mteval.runs import list_runs
from wmt26_mteval.tables.items import build_pairs
from wmt26_mteval.tables.metadata import build_systems, coverage_of, priority_lps
from wmt26_mteval.tables.predictions import build_run_table


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
