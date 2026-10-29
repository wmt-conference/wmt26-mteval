"""Check that every Task 3 CLI module runs without a separate launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "preparation", "pipeline", "significance", "paper_tables", "threshold.cli", "reproduction",
])
def test_module_help_without_launcher(module: str, tmp_path: Path) -> None:
    """Run each module outside the repository and show help without needing input data."""
    environment = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source, environment.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-m", f"wmt26_mteval_task3.{module}", "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "--help" in result.stdout
    assert not result.stderr
    assert not list(tmp_path.iterdir())
