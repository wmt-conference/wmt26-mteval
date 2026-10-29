"""Populate the saved baseline-delta TeX layout from regenerated metric tables."""

from __future__ import annotations

import csv
from pathlib import Path

from ..io import DataError
from ..reproduction import load_targets
from .core import load_official_metrics


def write_baseline_conversion_deltas(threshold_dir: Path, task1_metrics: Path,
                                     priority_pairs: tuple[str, ...]) -> Path:
    """Insert 100*(local-optimal Task 2 MCC minus Task 1 MCC) into the original layout.

    Read only generated CSVs. The template contains formatting and labels, never
    saved numerical results. This reproduces a previously hand-written table.
    """
    task1 = load_official_metrics(task1_metrics, "entry_key")
    with (threshold_dir / "per_language_threshold_summary.csv").open(newline="") as handle:
        optimized = {(row["entry_id"], row["language_pair"]): float(row["language_optimal_mcc"])
                     for row in csv.DictReader(handle) if row["track"] == "baseline"}
    references = {(entry_id, row["language_pair"]): float(row["mcc"])
                  for entry_id, rows in task1.items() for row in rows}
    family = next(f for f in load_targets()["families"] if f["name"] == "task1_baselines_converted")
    template = Path(__file__).parent / "templates/paper_figures/paper_baseline_conversion_deltas.tex"
    text = template.read_text()
    for row_index, entry in enumerate(family["entries"]):
        for pair_index, pair in enumerate(priority_pairs):
            key = (entry["entry_id"], pair)
            if key not in optimized or key not in references:
                raise DataError(f"Missing baseline delta source: {key}")
            value = 100 * (optimized[key] - references[key])
            text = text.replace(f"@@DELTA_{row_index}_{pair_index}@@", f"${value:+.2f}$")
    if "@@DELTA_" in text:
        raise DataError("Unpopulated baseline-delta table cell")
    destination = threshold_dir / "paper_figures/paper_baseline_conversion_deltas.tex"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return destination
