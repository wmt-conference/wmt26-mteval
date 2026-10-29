"""Reports for Task 2 to Task 3 conversions."""

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
from .task2_conversion import ConversionDecision, ConversionResult



def _conversion_info_rows(
    decisions: list[ConversionDecision],
    display_names: dict[int | str, str],
    results: list[ConversionResult],
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
                "threshold": "" if decision.threshold is None else decision.threshold,
                "threshold_raw": decision.threshold_raw,
                "comparison": decision.comparison,
                "inclusivity_raw": decision.inclusivity_raw,
                "conversion_status": decision.status,
                "reason": decision.reason,
                "attempted_pair_count": 0 if result is None else len(result.attempted_pairs),
                "attempted_pairs": (
                    "" if result is None else ";".join(result.attempted_pairs)
                ),
                "source_score_min": "" if result is None else result.source_score_min,
                "source_score_max": "" if result is None else result.source_score_max,
                "official_labels": "" if result is None else result.official_labels,
                "converted_error_free": (
                    "" if result is None else result.converted_error_free
                ),
                "converted_error": "" if result is None else result.converted_error,
                "archive_sha256": decision.archive_sha256,
                "segments_sha256": decision.segments_sha256,
                "model_card_sha256": decision.model_card_sha256,
                "converted_prediction_sha256": (
                    "" if result is None else result.converted_prediction_sha256
                ),
            }
        )
    return rows


def _metric_rows(
    results: list[ConversionResult],
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
                    "threshold": result.decision.threshold,
                    "comparison": result.decision.comparison,
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
    results: list[ConversionResult],
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
                    "threshold": decision.threshold,
                    "comparison": decision.comparison,
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
        "# Task 2 to Task 3 Conversion Decisions",
        "",
        "| ID | System | Base model | Uses reference | Opt-in | Threshold rule | Status | Reason |",
        "|---:|---|---|:---:|:---:|---:|---|---|",
    ]
    for row in rows:
        threshold_rule = (
            f"{row['comparison']} {float(row['threshold']):g}"
            if row["comparison"] and row["threshold"] != ""
            else ""
        )
        lines.append(
            f"| {row[id_column]} | {text(row['display_name'])} | "
            f"{text(row['base_model'])} | "
            f"{row['uses_reference']} | {row['opt_in']} | {threshold_rule} | "
            f"{row['conversion_status']} | {text(row['reason'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


