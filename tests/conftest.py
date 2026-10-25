from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Root of the repository checkout (the directory holding ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def mini_grand_path(repo_root: Path) -> Path:
    """Path to the 16-row Parquet subset of the grand file (see ``make_mini_grand.py``)."""
    return repo_root / "tests" / "fixtures" / "mini_grand.parquet"
