"""Typed Task 2 observations, evidence, and input descriptors."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

@dataclass(frozen=True)
class Screen:
    """One retained contrastive human-annotation screen.

    ``systems`` preserves display order and ``scores`` contains exactly one
    finite holistic score for each displayed system. ``task1_predictions``
    contains the parallel canonical span assessment for each system when the
    screen came from the unified converter; it is ``None`` for legacy Task 2
    screen files that contain scores only.
    """

    screen_id: str
    item_id: str
    direction: str
    annotator_id: str
    systems: tuple[str, ...]
    scores: Mapping[str, float]
    task1_predictions: Mapping[str, Mapping[str, Any]] | None = None



@dataclass(frozen=True)
class DirectionStatistics:
    """Coverage and repeated-evidence statistics for one direction."""

    direction: str
    blindset_items: int
    blindset_item_systems: int
    retained_screens: int
    human_score_observations: int
    missing_item_systems: int
    item_systems: int
    repeated_item_systems: int
    extra_item_system_scores: int
    maximum_item_system_scores: int
    multiple_distinct_annotators: int
    same_annotator_repeated: int
    co_screen_item_pairs: int
    repeated_co_screen_pairs: int
    extra_pair_screens: int
    maximum_pair_screens: int



@dataclass(frozen=True)
class HumanEvidence:
    """Validated and aggregated human scores for one language direction.

    ``segment_scores`` maps item -> system -> mean score over every retained
    screen. ``co_screen_scores`` maps (item, alphabetically-first system,
    alphabetically-second system) to the two corresponding means over only
    screens in which that pair was displayed together.
    """

    direction: str
    systems: tuple[str, ...]
    items: tuple[str, ...]
    segment_scores: Mapping[str, Mapping[str, float]]
    co_screen_scores: Mapping[tuple[str, str, str], tuple[float, float]]
    statistics: DirectionStatistics



@dataclass(frozen=True)
class SubmissionDescriptor:
    """One flat Task 2 submission and its optional model card."""

    submission_id: str
    scores_path: Path
    model_card_path: Path
    submission_name: str | None = None

    @property
    def display_name(self) -> str:
        """Return the human submission name, falling back to its stable ID."""

        return self.submission_name or self.submission_id



@dataclass(frozen=True)
class PredictionLoadResult:
    """Validation outcome for one submission in one requested direction.

    ``status`` is one of ``valid``, ``not_submitted``, or ``invalid``.  Valid
    results carry complete item/system predictions and score-range diagnostics;
    the other statuses carry a human-readable explanation instead.
    """

    direction: str
    status: str
    predictions: Mapping[str, Mapping[str, float]] | None
    message: str
    score_count: int = 0
    minimum_score: float | None = None
    maximum_score: float | None = None
    scores_below_zero: int = 0
    scores_above_hundred: int = 0
