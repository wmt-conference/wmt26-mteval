"""Shared Task 1 ranking scopes, eligibility, and deterministic ordering."""

from __future__ import annotations

from dataclasses import dataclass

from wmt26_mteval_task1.constants import PRIMARY_LANGUAGE_PAIRS
from wmt26_mteval_task1.models import (
    BatchScoreResult,
    SubmissionDescriptor,
    SubmissionScore,
)

EntryKey = tuple[str, str]


def entry_key(descriptor: SubmissionDescriptor) -> EntryKey:
    """Return a report-stable, type-qualified evaluator identity."""
    if descriptor.is_baseline:
        return "baseline", descriptor.baseline_name
    return "submission", descriptor.submission_id


def tie_breaker(score: SubmissionScore) -> tuple[int, int, str]:
    """Return the existing participant-first ordering for equal scores."""
    descriptor = score.descriptor
    if descriptor.is_baseline:
        return 1, 0, descriptor.baseline_name.casefold()
    return 0, int(descriptor.submission_id), ""


def scope_score(score: SubmissionScore, directions: tuple[str, ...]) -> float:
    """Return equal-weight direction F1, requiring every requested direction."""
    values = {row.language_pair: row.f1 for row in score.direction_scores}
    return sum(values[direction] for direction in directions) / len(directions)


@dataclass(frozen=True)
class RankingScope:
    """One ranking's fixed scoring directions and ordered eligible evaluators."""

    name: str
    directions: tuple[str, ...]
    entries: tuple[SubmissionScore, ...]


def ranking_scopes(result: BatchScoreResult) -> tuple[RankingScope, ...]:
    """Build aggregate and per-direction rankings on cohort-independent support.

    Official batches carry selected and gold direction sets. Older manually
    constructed API results fall back to their available direction scores.
    """
    available = {
        row.language_pair for score in result.scores for row in score.direction_scores
    }
    selected = set(result.selected_directions) if result.selected_directions else available
    gold = set(result.gold_directions) if result.gold_directions is not None else available
    common = selected - set(result.directions_without_references)
    definitions = [
        (
            "priority",
            set(PRIMARY_LANGUAGE_PAIRS),
            set(PRIMARY_LANGUAGE_PAIRS) & selected & gold,
        ),
        ("all_directions", common, common & gold),
    ]
    scopes: list[RankingScope] = []
    for name, required, scored in definitions:
        directions = tuple(sorted(scored))
        entries = []
        for score in result.scores:
            covered = set(score.coverage.complete_directions)
            eligible = required <= covered
            if not result.selected_directions:
                eligible = (
                    score.coverage.all_primary_complete
                    if name == "priority"
                    else score.coverage.all_official_complete
                )
            values = {row.language_pair for row in score.direction_scores}
            if directions and eligible and scored <= values:
                entries.append(score)
        entries.sort(
            key=lambda score: (-scope_score(score, directions), *tie_breaker(score))
        )
        scopes.append(RankingScope(name, directions, tuple(entries)))

    for direction in sorted(available):
        entries = [
            score
            for score in result.scores
            if direction not in score.coverage.termination_excluded_directions
            and any(row.language_pair == direction for row in score.direction_scores)
        ]
        entries.sort(
            key=lambda score: (-scope_score(score, (direction,)), *tie_breaker(score))
        )
        scopes.append(RankingScope("direction:" + direction, (direction,), tuple(entries)))
    return tuple(scopes)
