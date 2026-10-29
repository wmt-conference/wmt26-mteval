"""Check that the saved TeX layout is populated from generated CSV values."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from wmt26_mteval_task3.io import DataError
from wmt26_mteval_task3.reproduction import load_targets
from wmt26_mteval_task3.threshold.cli import PRIORITY_LANGUAGE_PAIRS
from wmt26_mteval_task3.threshold.presentation import write_baseline_conversion_deltas


def delta_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Write independent synthetic Task 1 and locally optimal Task 2 metric rows."""
    threshold = tmp_path / "threshold"
    threshold.mkdir()
    task1 = tmp_path / "task1.csv"
    baseline = load_targets()["families"][3]["entries"]
    with task1.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entry_key", "language_pair", "mcc"])
        for entry in baseline:
            for pair in PRIORITY_LANGUAGE_PAIRS:
                writer.writerow([entry["entry_id"], pair, "0.10"])
    with (threshold / "per_language_threshold_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track", "entry_id", "language_pair", "language_optimal_mcc"])
        for entry in baseline:
            for pair in PRIORITY_LANGUAGE_PAIRS:
                writer.writerow(["baseline", entry["entry_id"], pair, "0.22"])
    return threshold, task1


def test_delta_table_uses_computed_values_not_saved_numbers(tmp_path: Path) -> None:
    """Every cell must reflect the synthetic +12.00 delta, not historical results."""
    threshold, task1 = delta_sources(tmp_path)
    path = write_baseline_conversion_deltas(threshold, task1, PRIORITY_LANGUAGE_PAIRS)
    text = path.read_text()
    assert text.count("$+12.00$") == 72
    assert "@@DELTA_" not in text
    assert "$+3.78$" not in text


def test_delta_table_rejects_missing_metric_cells(tmp_path: Path) -> None:
    """Missing source metrics must not silently leave literal saved values behind."""
    threshold, task1 = delta_sources(tmp_path)
    lines = task1.read_text().splitlines()
    task1.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(DataError, match="Missing baseline delta source"):
        write_baseline_conversion_deltas(threshold, task1, PRIORITY_LANGUAGE_PAIRS)
