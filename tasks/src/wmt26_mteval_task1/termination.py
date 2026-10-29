"""Gold-aligned interpretation of optional generation termination metadata."""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from pathlib import Path

from wmt26_mteval_task1.models import (
    Pair,
    GroundTruthSet,
    TestSetIndex,
    TerminationCoverage,
)

TERMINATION_STATUSES = (
    "stop",
    "null",
    "length",
    "metadata_absent",
    "missing_reason",
    "malformed_metadata",
    "malformed_reason",
    "unexpected_reason",
)


def build_termination_coverage(
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    statuses: Mapping[Pair, str],
    *,
    reference_based: bool = False,
    directions: Iterable[str] | None = None,
) -> dict[str, TerminationCoverage]:
    """Count distinct applicable gold pairs and their supplied statuses per LP.

    Missing predictions do not enter the numerator; structural coverage handles
    them independently. Reference-system pairs and challenge pairs never enter
    either numerator or denominator, even for manually constructed ground-truth sets.
    """

    applicable = test_set.coverage_pairs_by_direction(reference_based=reference_based)
    selected = set(applicable) if directions is None else set(directions)
    gold_by_direction = {
        direction: set() for direction in applicable if direction in selected
    }
    for item_id, systems in ground_truth.assessments.items():
        direction = test_set.item_directions.get(item_id)
        if direction not in gold_by_direction:
            continue
        reference_system = test_set.reference_systems_by_direction.get(direction)
        for system in systems:
            pair = (item_id, system)
            if pair in applicable[direction] and system != reference_system:
                gold_by_direction[direction].add(pair)
    return {
        direction: TerminationCoverage(
            len(pairs),
            dict(Counter(statuses[pair] for pair in pairs if pair in statuses)),
        )
        for direction, pairs in sorted(gold_by_direction.items())
    }


def termination_coverage_messages(
    values: Mapping[str, TerminationCoverage],
) -> list[str]:
    """Describe metadata-bearing LPs and their ranking consequences for issue reports."""

    messages = []
    for direction, value in sorted(values.items()):
        if not value.counts or set(value.counts) == {"metadata_absent"}:
            continue
        counts = ", ".join(
            f"{status}={value.counts.get(status, 0)}" for status in TERMINATION_STATUSES
        )
        rate = (
            f"{100 * value.failure_rate:.4f}%"
            if value.failure_rate is not None
            else "not assessed (no gold)"
        )
        consequence = (
            "Excluded from this LP ranking and aggregates requiring this LP."
            if value.excluded
            else "Passes termination coverage."
        )
        messages.append(
            f"{direction}: gold_pairs={value.gold_pairs}; {counts}; null+length={rate}. "
            f"{consequence} Threshold is strictly above 5%; ordinary MPP inputs are unchanged."
        )
    return messages


def termination_status(record: Mapping[str, Any], system: str) -> str:
    """Classify one system's metadata without validating or changing its prediction."""

    if "stage1_finish_reason" not in record:
        return "metadata_absent"
    reasons = record["stage1_finish_reason"]
    if not isinstance(reasons, dict):
        return "malformed_metadata"
    if system not in reasons:
        return "missing_reason"
    reason = reasons[system]
    if reason is None:
        return "null"
    if not isinstance(reason, str):
        return "malformed_reason"
    return reason if reason in {"stop", "length"} else "unexpected_reason"


def termination_summary(counts: Counter[str]) -> str | None:
    """Return a warning summary, or None when an entry has no metadata at all."""

    if not counts or set(counts) == {"metadata_absent"}:
        return None
    values = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    return (
        f"Gold-aligned termination metadata (ordinary MPP inputs unchanged; coverage uses the >5% rule): {values}. "
        "Empty and non-empty translations follow the same scoring policy. "
        "See the baseline audit for direction, empty-target, and gold-aligned counts."
    )


def read_gold_termination_statuses(
    path: Path, test_set: TestSetIndex, ground_truth: GroundTruthSet
) -> dict[Pair, str]:
    """Read supplied finish statuses on supported official gold pairs without changing predictions."""

    from wmt26_mteval_task1.io import get_prediction_assessments, read_jsonl

    statuses = {}
    official = test_set.coverage_pairs_by_direction(reference_based=False)
    for line, row in read_jsonl(path):
        item_id = row.get("item_id")
        direction = test_set.item_directions.get(item_id)
        if direction not in official or item_id not in ground_truth.assessments:
            continue
        for system in get_prediction_assessments(row, f"{path}:{line}"):
            pair = (item_id, system)
            if (
                pair in official[direction]
                and system in ground_truth.assessments[item_id]
            ):
                statuses[pair] = termination_status(row, system)
    return statuses
