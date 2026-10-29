"""Generate the Task 1 paper source-length figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wmt26_mteval_task1.markdown_reporting import markdown_table, write_markdown_companion
from wmt26_mteval_task1.score_statistics import LENGTH_BUCKETS, BatchStatisticsResult


def _matplotlib() -> Any:
    """Import Matplotlib lazily and configure its non-interactive backend."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires Matplotlib. Install the project with "
            "'python -m pip install -e .' to generate the paper figure."
        ) from exc
    return plt


def write_paper_length_effect(
    output_dir: Path, result: BatchStatisticsResult
) -> Path:
    """Plot median evaluator MPP F1 and IQR over source-length quartiles.

    Input is the complete priority-evaluator statistics result. The returned
    PDF reproduces the compact, unit-scale figure used in the paper.
    """

    summaries = {
        row.length_bucket: row
        for row in result.length_summaries
        if row.bucket_scope == "within_direction"
        and row.length_basis == "source_tokens"
    }
    if set(summaries) != set(LENGTH_BUCKETS):
        raise ValueError("source-length paper figure requires four populated quartiles")
    rows = [summaries[bucket] for bucket in LENGTH_BUCKETS]
    if any(
        row.median_f1 is None
        or row.first_quartile_f1 is None
        or row.third_quartile_f1 is None
        for row in rows
    ):
        raise ValueError("source-length paper figure has undefined quartile statistics")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "task1-length-effect.pdf"
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(8, 4))
    x_values = list(range(4))
    axis.plot(
        x_values,
        [row.median_f1 for row in rows],
        marker="o",
        color="#2563eb",
        label="Evaluator median",
    )
    axis.fill_between(
        x_values,
        [row.first_quartile_f1 for row in rows],
        [row.third_quartile_f1 for row in rows],
        color="#2563eb",
        alpha=0.2,
        label="Evaluator IQR",
    )
    axis.set_xticks(x_values, LENGTH_BUCKETS)
    axis.set_ylim(0.15, 0.45)
    axis.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    axis.set_xlabel("Source-length quartile", fontsize=16)
    axis.set_ylabel("MPP F1", fontsize=16)
    axis.tick_params(labelsize=16)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    write_markdown_companion(
        path, "Task 1: MPP F1 by source-length quartile",
        markdown_table(
            ("Source-length quartile", "Evaluator median F1",
             "Evaluator Q1 F1", "Evaluator Q3 F1", "Evaluators"),
            (
                (row.length_bucket, f"{100 * row.median_f1:.2f}",
                 f"{100 * row.first_quartile_f1:.2f}",
                 f"{100 * row.third_quartile_f1:.2f}",
                 getattr(row, "observation_count", "—"))
                for row in rows
            ),
        ) + "\n\nQuartiles are based on within-direction source-token lengths.",
    )
    return path
