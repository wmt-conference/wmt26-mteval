"""Reports for Task 1 to Task 3 conversions."""

from __future__ import annotations

import importlib.metadata
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .io import sha256_file
from .models import HumanReference, OfficialInventory
from .reporting import (
    human_reference_rows,
    rank_pair_rows,
    write_csv,
    write_json,
    write_markdown_tables,
)
from .task1_conversion import Task1ConversionResult, Task1Decision



def _conversion_info_rows(
    decisions: list[Task1Decision],
    display_names: dict[int, str],
    results: list[Task1ConversionResult],
    id_column: str,
) -> list[dict[str, Any]]:
    """Return one diagnostic/status row per configured conversion decision."""
    results_by_id = {result.decision.submission_id: result for result in results}
    rows = []
    for decision in decisions:
        result = results_by_id.get(decision.submission_id)
        rows.append(
            {
                id_column: decision.submission_id,
                "display_name": display_names[decision.submission_id],
                "organization_name": decision.organization_name,
                "submission_name": decision.submission_name,
                "base_model": decision.base_model,
                "uses_reference": (
                    "" if decision.uses_reference is None else decision.uses_reference
                ),
                "opt_in": "" if decision.permission is None else decision.permission,
                "permission_raw": decision.permission_raw,
                "conversion_status": decision.status,
                "reason": decision.reason,
                "span_validation_status": (
                    ""
                    if result is None
                    else "preserved_zero_length_spans"
                    if result.zero_length_spans
                    else "valid"
                ),
                "attempted_pair_count": 0 if result is None else len(result.attempted_pairs),
                "attempted_pairs": (
                    "" if result is None else ";".join(result.attempted_pairs)
                ),
                "official_labels": "" if result is None else result.official_labels,
                "converted_error_free": (
                    "" if result is None else result.converted_error_free
                ),
                "converted_error": "" if result is None else result.converted_error,
                "submitted_error_spans": (
                    "" if result is None else result.submitted_error_spans
                ),
                "zero_length_spans": (
                    "" if result is None else result.zero_length_spans
                ),
                "omission_marker_spans": (
                    "" if result is None else result.omission_marker_spans
                ),
                "instruction_error_marker_spans": (
                    "" if result is None else result.instruction_error_marker_spans
                ),
                "nonstandard_zero_length_spans": (
                    "" if result is None else result.nonstandard_zero_length_spans
                ),
                "top_level_omission_labels": (
                    "" if result is None else result.top_level_omission_labels
                ),
                "top_level_instruction_fault_labels": (
                    ""
                    if result is None
                    else result.top_level_instruction_fault_labels
                ),
                "archive_sha256": decision.archive_sha256,
                "predictions_sha256": decision.predictions_sha256,
                "model_card_sha256": decision.model_card_sha256,
                "converted_prediction_sha256": (
                    "" if result is None else result.converted_prediction_sha256
                ),
            }
        )
    return rows


def _metric_rows(
    results: list[Task1ConversionResult],
    summary_rows: list[dict[str, Any]],
    id_column: str,
) -> list[dict[str, Any]]:
    """Combine conversion metrics with the matching human coverage summaries."""
    summaries = {row["language_pair"]: row for row in summary_rows}
    rows = []
    for result in results:
        for metric in result.metric_rows:
            summary = summaries[metric["language_pair"]]
            rows.append(
                {
                    id_column: metric["submission_id"],
                    "display_name": metric["display_name"],
                    "organization_name": metric["organization_name"],
                    "submission_name": metric["submission_name"],
                    "base_model": result.decision.base_model,
                    "language_pair": metric["language_pair"],
                    "inventory_items": summary["inventory_items"],
                    "annotated_items": metric["annotated_items"],
                    "item_coverage": summary["item_coverage"],
                    "inventory_item_systems": summary["inventory_item_systems"],
                    "annotated_item_systems": summary["annotated_item_systems"],
                    "item_system_coverage": summary["item_system_coverage"],
                    "human_assessments": metric["annotated_labels"],
                    "gold_error_free": metric["gold_error_free"],
                    "gold_error": metric["gold_error"],
                    "true_positive": metric["true_positive"],
                    "false_positive": metric["false_positive"],
                    "false_negative": metric["false_negative"],
                    "true_negative": metric["true_negative"],
                    "mcc": metric["mcc"],
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "f1": metric["f1"],
                }
            )
    return rows


def _table_rows(
    results: list[Task1ConversionResult],
    metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    id_column: str,
) -> list[dict[str, Any]]:
    """Build ranked per-language table rows, retaining blank cells for uncovered pairs."""
    metrics = {
        (row[id_column], row["language_pair"]): row for row in metric_rows
    }
    rows = []
    for summary in summary_rows:
        pair_rows = []
        for result in results:
            decision = result.decision
            metric = metrics.get((decision.submission_id, summary["language_pair"]))
            pair_rows.append(
                {
                    "rank": "",
                    id_column: decision.submission_id,
                    "display_name": result.display_name,
                    "organization_name": decision.organization_name,
                    "submission_name": decision.submission_name,
                    "base_model": decision.base_model,
                    "language_pair": summary["language_pair"],
                    "covered": metric is not None,
                    "inventory_items": summary["inventory_items"],
                    "annotated_items": summary["annotated_items"],
                    "item_coverage": summary["item_coverage"],
                    "inventory_item_systems": summary["inventory_item_systems"],
                    "annotated_item_systems": summary["annotated_item_systems"],
                    "item_system_coverage": summary["item_system_coverage"],
                    "human_assessments": summary["human_assessments"],
                    "gold_error_free": summary["error_free"],
                    "gold_error": summary["error"],
                    "mcc": "" if metric is None else metric["mcc"],
                    "precision": "" if metric is None else metric["precision"],
                    "recall": "" if metric is None else metric["recall"],
                    "f1": "" if metric is None else metric["f1"],
                }
            )
        rows.extend(rank_pair_rows(pair_rows))
    return rows


def _write_conversion_markdown(
    path: Path, rows: list[dict[str, Any]], id_column: str
) -> None:
    """Write the per-system conversion-status table as Markdown."""
    def text(value: Any) -> str:
        """Compute text from the supplied analysis state and return its representation."""
        return str(value).replace("|", "\\|")

    lines = [
        "# Task 1 to Task 3 Conversion Decisions",
        "",
        "| ID | System | Opt-in | Status | Span handling | Zero-length spans |",
        "|---:|---|:---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[id_column]} | {text(row['display_name'])} | "
            f"{row['opt_in']} | {row['conversion_status']} | "
            f"{row['span_validation_status']} | {row['zero_length_spans']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


