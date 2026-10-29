"""Write deterministic manifests, audit data, and paper-table inputs."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import sha256_file
from .models import HumanReference, OfficialInventory, SubmissionMetadata, SubmissionSpec



def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write supplied rows using deterministic field order and newline conventions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    """Write sorted, indented JSON provenance to the requested path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def human_reference_rows(
    inventory: OfficialInventory, human_reference: HumanReference
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate human evidence by pair and return summary and private audit rows."""
    items_by_pair: defaultdict[str, set[str]] = defaultdict(set)
    labels_by_pair: Counter[str] = Counter()
    error_free_by_pair: Counter[str] = Counter()
    omission_by_pair: Counter[str] = Counter()
    instruction_by_pair: Counter[str] = Counter()
    ordinary_by_pair: Counter[str] = Counter()
    keys_by_pair: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    audit_rows = []

    for item_id, system in sorted(human_reference.labels):
        judgments = human_reference.judgments_for((item_id, system))
        for assessment_index, judgment in enumerate(judgments):
            pair = judgment.language_pair
            items_by_pair[pair].add(item_id)
            keys_by_pair[pair].add((item_id, system))
            labels_by_pair[pair] += 1
            error_free_by_pair[pair] += judgment.error_free
            omission_by_pair[pair] += judgment.omission_count
            instruction_by_pair[pair] += judgment.instruction_error_count
            ordinary_by_pair[pair] += judgment.ordinary_span_count
            audit_rows.append(
                {
                    "language_pair": pair,
                    "item_id": item_id,
                    "system": system,
                    "assessment_index": assessment_index,
                    "annotator_id": judgment.annotator_id,
                    "submit_time": judgment.submit_time,
                    "score": judgment.score,
                    "error_free": judgment.error_free,
                    "submitted_judgments_for_key": human_reference.judgment_counts[
                        (item_id, system)
                    ],
                    "annotators_for_key": human_reference.annotator_counts[
                        (item_id, system)
                    ],
                    "ordinary_span_count": judgment.ordinary_span_count,
                    "omission_count": judgment.omission_count,
                    "instruction_error_count": judgment.instruction_error_count,
                    "spans_json": json.dumps(
                        [span.as_dict() for span in judgment.spans],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                    "source_group": judgment.group_name,
                    "source_record_index": judgment.record_index,
                    "source_item_index": judgment.item_index,
                }
            )

    summary_rows = []
    for pair in sorted(inventory.item_ids_by_pair):
        inventory_items = len(inventory.item_ids_by_pair[pair])
        inventory_labels = inventory.label_counts_by_pair[pair]
        annotated_items = len(items_by_pair[pair])
        human_assessments = labels_by_pair[pair]
        annotated_item_systems = len(keys_by_pair[pair])
        error_free = error_free_by_pair[pair]
        summary_rows.append(
            {
                "language_pair": pair,
                "inventory_items": inventory_items,
                "annotated_items": annotated_items,
                "item_coverage": annotated_items / inventory_items,
                "inventory_item_systems": inventory_labels,
                "annotated_item_systems": annotated_item_systems,
                "item_system_coverage": annotated_item_systems / inventory_labels,
                "human_assessments": human_assessments,
                "error_free": error_free,
                "error": human_assessments - error_free,
                "ordinary_spans": ordinary_by_pair[pair],
                "omission_markers": omission_by_pair[pair],
                "instruction_error_markers": instruction_by_pair[pair],
            }
        )
    return summary_rows, audit_rows


def build_display_names(
    submission_metadata: list[SubmissionMetadata],
) -> dict[int, str]:
    """Build publication-facing names without Codabench account names."""
    organization_counts = Counter(
        metadata.organization_name for metadata in submission_metadata
    )
    base_names = {}
    for metadata in submission_metadata:
        organization = metadata.organization_name
        if organization_counts[organization] == 1:
            base_names[metadata.spec.submission_id] = organization
        else:
            base_names[metadata.spec.submission_id] = (
                f"{organization}: {metadata.submission_name}"
            )
    base_name_counts = Counter(base_names.values())
    return {
        submission_id: (
            f"{base_name} [{submission_id}]"
            if base_name_counts[base_name] > 1
            else base_name
        )
        for submission_id, base_name in base_names.items()
    }


def _add_reference_columns(
    metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    submission_metadata: list[SubmissionMetadata],
    display_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Attach display metadata and human coverage to raw direct-submission metrics."""
    summaries = {row["language_pair"]: row for row in summary_rows}
    metadata_by_id = {
        metadata.spec.submission_id: metadata for metadata in submission_metadata
    }
    result = []
    for row in metric_rows:
        summary = summaries[row["language_pair"]]
        metadata = metadata_by_id[row["submission_id"]]
        result.append(
            {
                "submission_id": row["submission_id"],
                "display_name": display_names[row["submission_id"]],
                "organization_name": metadata.organization_name,
                "submission_name": metadata.submission_name,
                "base_model": metadata.base_model,
                "language_pair": row["language_pair"],
                "inventory_items": summary["inventory_items"],
                "annotated_items": row["annotated_items"],
                "item_coverage": summary["item_coverage"],
                "inventory_item_systems": summary["inventory_item_systems"],
                "annotated_item_systems": summary["annotated_item_systems"],
                "item_system_coverage": summary["item_system_coverage"],
                "human_assessments": row["annotated_labels"],
                "gold_error_free": row["gold_error_free"],
                "gold_error": row["gold_error"],
                "true_positive": row["true_positive"],
                "false_positive": row["false_positive"],
                "false_negative": row["false_negative"],
                "true_negative": row["true_negative"],
                "mcc": row["mcc"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
            }
        )
    return result


def table_rows(
    specs: list[SubmissionSpec],
    metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    submission_metadata: list[SubmissionMetadata],
    display_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Build direct-submission table rows with shared human coverage and competition ranks."""
    metrics = {
        (row["submission_id"], row["language_pair"]): row for row in metric_rows
    }
    metadata_by_id = {
        metadata.spec.submission_id: metadata for metadata in submission_metadata
    }
    rows = []
    for summary in summary_rows:
        pair = summary["language_pair"]
        pair_rows = []
        for spec in specs:
            metric = metrics.get((spec.submission_id, pair))
            metadata = metadata_by_id[spec.submission_id]
            pair_rows.append(
                {
                    "rank": "",
                    "submission_id": spec.submission_id,
                    "display_name": display_names[spec.submission_id],
                    "organization_name": metadata.organization_name,
                    "submission_name": metadata.submission_name,
                    "base_model": metadata.base_model,
                    "language_pair": pair,
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


def rank_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort one language-pair table by MCC and assign competition ranks."""
    def entry_key(row: dict[str, Any]) -> Any:
        """Compute entry key from the supplied analysis state and return its representation."""
        return row["entry_key"] if "entry_key" in row else row["submission_id"]

    rows.sort(
        key=lambda row: (
            not row["covered"],
            -float(row["mcc"]) if row["covered"] else 0.0,
            entry_key(row),
        )
    )
    previous_mcc = None
    previous_rank = 0
    for position, row in enumerate(rows, start=1):
        if not row["covered"]:
            continue
        if previous_mcc is None or row["mcc"] != previous_mcc:
            previous_rank = position
            previous_mcc = row["mcc"]
        row["rank"] = previous_rank
    return rows


def _format_metric(value: Any) -> str:
    """Compute format metric from the supplied analysis state and return its representation."""
    return "" if value == "" else f"{float(value):.4f}"


def write_markdown_tables(
    path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    title: str = "Task 3 Human-Reference Results",
) -> None:
    """Write per-language metrics and human-coverage tables as Markdown."""
    rows_by_pair: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_pair[row["language_pair"]].append(row)
    lines = [
        f"# {title}",
        "",
        "MCC is the primary metric. Precision, recall, and F1 use error-free as "
        "the positive class. Blank cells indicate that a submission did not cover "
        "the language pair.",
    ]
    for summary in summary_rows:
        pair = summary["language_pair"]
        lines.extend(
            [
                "",
                f"## {pair}",
                "",
                f"Human coverage: {summary['annotated_items']:,}/"
                f"{summary['inventory_items']:,} items "
                f"({summary['item_coverage']:.1%}); "
                f"{summary['annotated_item_systems']:,}/"
                f"{summary['inventory_item_systems']:,} unique item-system pairs "
                f"({summary['item_system_coverage']:.1%}); "
                f"{summary['human_assessments']:,} human assessments. "
                f"Error-free assessments: {summary['error_free']:,}; "
                f"error assessments: {summary['error']:,}.",
                "",
                "| Rank | Submission | Base model | MCC | Precision | Recall | F1 |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows_by_pair[pair]:
            lines.append(
                f"| {row['rank']} | {row['display_name']} | {row['base_model']} | "
                f"{_format_metric(row['mcc'])} | "
                f"{_format_metric(row['precision'])} | "
                f"{_format_metric(row['recall'])} | "
                f"{_format_metric(row['f1'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


