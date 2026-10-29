"""Compare GPT Sol evaluators on identical, successfully terminated gold pairs."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from span_mt_metrics_eval import compute

from wmt26_mteval_task1.audit_io import file_digest
from wmt26_mteval_task1.constants import SEVERITY_WEIGHTS, mpp_metric_options
from wmt26_mteval_task1.io import validate_model_card
from wmt26_mteval_task1.models import (
    ModelCardTemplate,
    GroundTruthSet,
    Span,
    SubmissionDescriptor,
    TestSetIndex,
)
from wmt26_mteval_task1.scoring import (
    _prepare_submission,
    _segment_weights,
    aggregate_mpp_components,
    scored_assessments_from_result,
)
from wmt26_mteval_task1.termination import read_gold_termination_statuses

MEDIUM = "gpt_56_sol_medium_no_ref_task1"
XHIGH = "gpt56_sol_xhigh_no_ref_task1"


@dataclass(frozen=True)
class GptSolStopResult:
    """Pooled MPP, prediction statistics, retention counts, and reproducibility metadata."""

    scores: tuple[dict[str, Any], ...]
    prediction_statistics: tuple[dict[str, Any], ...]
    directions: tuple[dict[str, Any], ...]
    pairs: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]


def _prediction_statistics(records) -> dict[str, int | float | None]:
    """Summarize unique predictions without repeating them for each annotation.

    The input contains one record per human assessment. Predictions are identical
    across repeated annotations of an item-system pair, so each pair is retained once
    here. Text-span length excludes synthetic omission spans; error totals and
    mean severity includes omissions, matching the score-statistics contract.
    """

    predictions = {
        (record.item_id, record.system): record for record in records
    }.values()
    assessment_count = 0
    text_span_count = 0
    text_span_characters = 0
    omission_count = 0
    minor_error_count = 0
    major_error_count = 0
    for record in predictions:
        assessment_count += 1
        for span in record.prediction:
            severity = span["severity"]
            if severity == "minor":
                minor_error_count += 1
            else:
                major_error_count += 1
            if _is_omission_span(span, record.target_length):
                omission_count += 1
            else:
                text_span_count += 1
                text_span_characters += span["end"] - span["start"]
    total_error_count = text_span_count + omission_count
    return {
        "prediction_assessments": assessment_count,
        "text_span_count": text_span_count,
        "omission_count": omission_count,
        "total_error_count": total_error_count,
        "mean_errors_per_assessment": (
            total_error_count / assessment_count if assessment_count else None
        ),
        "mean_text_span_length_characters": (
            text_span_characters / text_span_count if text_span_count else None
        ),
        "minor_error_count": minor_error_count,
        "major_error_count": major_error_count,
        "mean_severity_weight": (
            (
                SEVERITY_WEIGHTS["minor"] * minor_error_count
                + SEVERITY_WEIGHTS["major"] * major_error_count
            )
            / total_error_count
            if total_error_count
            else None
        ),
    }


def _is_omission_span(span: Span, target_length: int) -> bool:
    """Return whether one normalized span is the scorer's synthetic omission."""

    return span["start"] == target_length and span["end"] == target_length + 1


def analyze_gpt_sol_stop(
    baselines: Iterable[SubmissionDescriptor],
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    template: ModelCardTemplate | None,
    *,
    matching_algorithm: str = "greedy",
) -> GptSolStopResult:
    """Compute one pooled, pair-weighted micro MPP call per evaluator on shared stop pairs.

    LP ranking gates are deliberately bypassed for this diagnostic intersection.
    Span validation, reference policy, and human-assessment multiplicity are shared
    with ordinary scoring. Missing or empty comparisons raise ValueError.
    """

    options = mpp_metric_options(matching_algorithm)
    selected = {}
    for baseline in baselines:
        name = baseline.source_baseline_name
        if name in (MEDIUM, XHIGH):
            if name in selected:
                raise ValueError(f"multiple baselines resolve to {name}")
            selected[name] = baseline
    missing = {MEDIUM, XHIGH} - selected.keys()
    if missing:
        raise ValueError(
            "missing required GPT Sol baseline(s): " + ", ".join(sorted(missing))
        )
    prepared = {}
    stops = {}
    fingerprints = {}
    for name in (MEDIUM, XHIGH):
        descriptor = selected[name]
        metadata = validate_model_card(descriptor.model_card_path, template)
        prepared[name] = _prepare_submission(
            descriptor,
            test_set,
            ground_truth,
            require_complete_predictions=False,
            reference_based=metadata.evaluator_characteristics["reference_use"],
        )
        statuses = read_gold_termination_statuses(
            descriptor.predictions_path, test_set, ground_truth
        )
        stops[name] = {
            (r.item_id, r.system)
            for r in prepared[name].assessments
            if statuses.get((r.item_id, r.system)) == "stop"
        }
        fingerprints[name] = {
            "predictions_path": str(descriptor.predictions_path),
            "predictions_sha256": file_digest(descriptor.predictions_path),
            "model_card_sha256": file_digest(descriptor.model_card_path),
        }
    shared = stops[MEDIUM] & stops[XHIGH]
    if not shared:
        raise ValueError("GPT Sol shared-stop intersection has no gold-annotated pairs")
    scores = []
    prediction_statistics = []
    keys = None
    for name in (MEDIUM, XHIGH):
        records = sorted(
            (r for r in prepared[name].assessments if (r.item_id, r.system) in shared),
            key=lambda r: (r.item_id, r.system, r.reference_index),
        )
        current_keys = [(r.item_id, r.system, r.reference_index) for r in records]
        if keys is not None and current_keys != keys:
            raise ValueError(
                "shared-stop human-assessment alignment differs between evaluators"
            )
        keys = current_keys
        result = compute(
            [r.prediction for r in records],
            [r.reference for r in records],
            **options,
        )
        precision, recall, f1 = aggregate_mpp_components(
            scored_assessments_from_result(
                records,
                _segment_weights(records),
                result,
            )
        )
        scores.append(
            {
                "evaluator": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "retained_pairs": len(shared),
                "retained_assessments": len(records),
            }
        )
        prediction_statistics.append({"evaluator": name, **_prediction_statistics(records)})
    delta = {
        key: scores[1][key] - scores[0][key] for key in ("precision", "recall", "f1")
    }
    scores.append(
        {
            "evaluator": "xhigh_minus_medium",
            **delta,
            "retained_pairs": len(shared),
            "retained_assessments": len(keys),
        }
    )
    pair_rows = tuple(
        {
            "item_id": item,
            "system": system,
            "language_pair": test_set.item_directions[item],
            "human_assessments": len(ground_truth.assessments[item][system]),
        }
        for item, system in sorted(shared)
    )
    lp_pairs = Counter(row["language_pair"] for row in pair_rows)
    lp_assessments = Counter()
    for row in pair_rows:
        lp_assessments[row["language_pair"]] += row["human_assessments"]
    directions = sorted(
        {
            direction
            for value in prepared.values()
            for direction, count in value.coverage.provided_pair_counts.items()
            if count
        }
    )
    direction_rows = []
    for direction in directions:
        row = {
            "language_pair": direction,
            "gold_pairs": next(
                p.coverage.termination_by_direction[direction].gold_pairs
                for p in prepared.values()
                if direction in p.coverage.termination_by_direction
            ),
            "retained_pairs": lp_pairs[direction],
            "retained_assessments": lp_assessments[direction],
        }
        for label, name in (("medium", MEDIUM), ("xhigh", XHIGH)):
            row[f"{label}_stop_pairs"] = sum(
                test_set.item_directions[item] == direction for item, _ in stops[name]
            )
        direction_rows.append(row)
    provenance = {
        "metric_options": options,
        "matching_algorithm": matching_algorithm,
        "averaging": "micro",
        "weighting": "Each human assessment has weight 1/N for its item-system pair; pool across LPs.",
        "filter": "Both evaluators have explicit stop on the same official gold pair; independent of ranking eligibility.",
        "inputs": {
            **fingerprints,
            **{
                label: {"path": str(path), "sha256": file_digest(path)}
                for label, path in (
                    ("test_set", test_set.path),
                    ("ground_truth", ground_truth.path),
                )
            },
            **(
                {"model_card_template": {
                    "path": str(template.path),
                    "sha256": file_digest(template.path),
                }} if template is not None else {}
            ),
        },
    }
    return GptSolStopResult(
        tuple(scores),
        tuple(prediction_statistics),
        tuple(direction_rows),
        pair_rows,
        provenance,
    )
