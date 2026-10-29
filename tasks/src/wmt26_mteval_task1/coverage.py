"""Official item-system completeness calculations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set

from wmt26_mteval_task1.constants import PRIMARY_LANGUAGE_PAIRS
from wmt26_mteval_task1.models import (
    CoverageReport,
    Pair,
    TestSetIndex,
    TerminationCoverage,
)


def build_coverage_report(
    test_set: TestSetIndex,
    provided_pairs_by_direction: Mapping[str, Set[Pair]],
    *,
    official_rows: int,
    challenge_rows: int,
    total_rows: int,
    reference_based: bool = False,
    directions: Iterable[str] | None = None,
    termination_by_direction: Mapping[str, TerminationCoverage] | None = None,
) -> CoverageReport:
    """Classify coverage against the directions applicable to an evaluator."""

    expected_by_direction = test_set.coverage_pairs_by_direction(
        reference_based=reference_based
    )
    if directions is not None:
        selected = set(directions)
        expected_by_direction = {
            direction: pairs
            for direction, pairs in expected_by_direction.items()
            if direction in selected
        }

    complete: list[str] = []
    partial: list[str] = []
    absent: list[str] = []
    provided_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}

    for direction, expected in sorted(expected_by_direction.items()):
        provided = set(provided_pairs_by_direction.get(direction, set()))
        retained = provided & set(expected)
        provided_counts[direction] = len(retained)
        expected_counts[direction] = len(expected)
        termination = (termination_by_direction or {}).get(direction)
        if expected <= retained and not (termination and termination.excluded):
            complete.append(direction)
        elif retained:
            partial.append(direction)
        else:
            absent.append(direction)

    complete_set = set(complete)
    provided_directions = {
        direction
        for direction, pairs in provided_pairs_by_direction.items()
        if pairs and direction in expected_by_direction
    }
    all_primary_complete = PRIMARY_LANGUAGE_PAIRS <= complete_set
    primary_only = (
        all_primary_complete and provided_directions <= PRIMARY_LANGUAGE_PAIRS
    )
    return CoverageReport(
        complete_directions=tuple(complete),
        partial_directions=tuple(partial),
        absent_directions=tuple(absent),
        provided_pair_counts=provided_counts,
        expected_pair_counts=expected_counts,
        official_rows=official_rows,
        challenge_rows=challenge_rows,
        total_rows=total_rows,
        all_official_complete=len(complete) == len(expected_by_direction),
        all_primary_complete=all_primary_complete,
        primary_only=primary_only,
        termination_by_direction=termination_by_direction or {},
    )
