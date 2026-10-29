"""CSV and Markdown outputs for threshold sensitivity results."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from .core import AnalysisResult, CV_FOLDS


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write supplied rows using deterministic field order and newline conventions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_result_tables(output_dir: Path, results: list[AnalysisResult]) -> None:
    """Write detailed sensitivity CSVs and the original Markdown interpretation."""
    rows = [result.row for result in results]
    language_rows = [
        language.row for result in results for language in result.languages
    ]
    write_csv(output_dir / "threshold_summary.csv", rows)
    write_csv(output_dir / "per_language_threshold_summary.csv", language_rows)
    (output_dir / "REPORT.md").write_text(
        markdown_report(rows), encoding="utf-8", newline="\n"
    )


def fmt_threshold(value: Any) -> str:
    """Format a selected threshold for the sensitivity report."""
    return f"{float(value):.6g}"


def fmt_mcc(value: Any) -> str:
    """Format a correlation or signed improvement for the sensitivity report."""
    return f"{float(value):.4f}"


def markdown_report(rows: list[dict[str, Any]]) -> str:
    """Render the original post-hoc sensitivity summary and interpretation."""
    lines = [
        "# Task 2 Conversion Threshold Analysis",
        "",
        (
            "This is a post-hoc sensitivity analysis. It does not change the frozen "
            "selection, submitted threshold rules, official result tables, or manifests."
        ),
        "",
        "## Method",
        "",
        (
            "All aggregates use the eight WMT26 priority language pairs: `ces-deu`, "
            "`ces-vie`, `eng-arz`, `eng-ekk`, `eng-ind`, `eng-isl`, `eng-sme`, and "
            "`zho-jpn`. Systems must cover all eight pairs; incomplete systems are "
            "excluded."
        ),
        "",
        (
            "For each participant conversion and organizer baseline, the primary analysis "
            "independently optimizes one threshold per language pair and takes the unweighted "
            "mean of the resulting per-language MCCs. Error-free is the positive class. "
            "Every realizable score boundary is tested, equal scores are never split, and "
            "each selected rule is written as `score >= threshold`."
        ),
        "",
        (
            "These per-language results are evaluated on the same labels used to choose the "
            "thresholds and are therefore optimistic upper bounds. The detailed CSV contains "
            "every selected language threshold and its confusion matrix."
        ),
        "",
        (
            "For comparison, the analysis also selects one threshold by pooled MCC over all "
            f"assessments. Its CV result uses deterministic {CV_FOLDS}-fold item-grouped "
            "cross-validation: complete item IDs stay in one fold, and each held-out fold "
            "uses a threshold selected on the other folds."
        ),
        "",
        "## Summary",
        "",
    ]
    for track, label in (
        ("participant", "Participant conversions"),
        ("baseline", "Organizer baselines"),
    ):
        selected = [row for row in rows if row["track"] == track]
        language_vs_declared = [
            row["language_optimal_macro_mcc_delta_vs_declared"]
            for row in selected
        ]
        language_vs_pooled = [
            row["language_optimal_macro_mcc_delta_vs_pooled_optimum"]
            for row in selected
        ]
        cv = [row["cv_pooled_mcc_delta_vs_declared"] for row in selected]
        lines.append(
            f"- {label}: per-language optimization raises macro MCC by a median "
            f"{statistics.median(language_vs_declared):+.4f} versus the declared rule and "
            f"{statistics.median(language_vs_pooled):+.4f} beyond the one-threshold pooled "
            f"optimum. The pooled-threshold item-grouped CV delta has median "
            f"{statistics.median(cv):+.4f}."
        )

    lines.extend(
        [
            "",
            (
                "The organizer Task 2 baseline conversions are shown separately because "
                "the paper-table pipeline marks them as having known comparability issues."
            ),
            "",
            "## Primary Per-language Results",
            "",
            "| Track | ID | Submission | Base model | Declared macro MCC | Pooled-threshold macro MCC | Per-language optimized macro MCC | Delta vs declared | Extra vs pooled threshold |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda value: (value["track"], str(value["entry_id"]))):
        submission = str(row["submission_name"]).replace("|", "\\|")
        base_model = str(row["base_model"]).replace("|", "\\|")
        lines.append(
            f"| {row['track']} | {row['entry_id']} | {submission} | {base_model} | "
            f"{fmt_mcc(row['declared_macro_language_mcc'])} | "
            f"{fmt_mcc(row['optimal_macro_language_mcc'])} | "
            f"{fmt_mcc(row['language_optimal_macro_mcc'])} | "
            f"{row['language_optimal_macro_mcc_delta_vs_declared']:+.4f} | "
            f"{row['language_optimal_macro_mcc_delta_vs_pooled_optimum']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Pooled-threshold Comparison",
            "",
            "| Track | ID | Submission | Base model | Declared | Pooled MCC | Optimized >= | Max pooled MCC | Apparent delta | Macro delta | CV pooled MCC | CV delta |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda value: (
            value["track"],
            -value["apparent_pooled_mcc_delta"],
            str(value["entry_id"]),
        ),
    ):
        submission = str(row["submission_name"]).replace("|", "\\|")
        base_model = str(row["base_model"]).replace("|", "\\|")
        lines.append(
            f"| {row['track']} | {row['entry_id']} | {submission} | {base_model} | "
            f"{row['declared_comparison']} {fmt_threshold(row['declared_threshold'])} | "
            f"{fmt_mcc(row['declared_pooled_mcc'])} | "
            f"{fmt_threshold(row['optimal_threshold'])} | "
            f"{fmt_mcc(row['optimal_pooled_mcc'])} | "
            f"{row['apparent_pooled_mcc_delta']:+.4f} | "
            f"{row['pooled_optimum_macro_language_mcc_delta']:+.4f} | "
            f"{fmt_mcc(row['cv_item_grouped_pooled_mcc'])} | "
            f"{row['cv_pooled_mcc_delta_vs_declared']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The primary per-language gains are same-data estimates with no held-out "
                "validation. They optimize many more parameters than the pooled experiment, "
                "so they should not be interpreted as expected generalization gains."
            ),
            "",
            (
                "Operational per-language thresholds should be calibrated on separate data "
                "and evaluated on untouched labels."
            ),
            "",
            (
                "For display, horizontal axes are normalized to 0-1 for systems using 0-100 "
                "scores; systems already using 0-1 are unchanged. FACET and doc-FACET retain "
                "their raw non-positive severity scale from -107 through 0."
            ),
            "",
        ]
    )
    return "\n".join(lines)
