"""Typed data objects returned by the public package API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


Pair = tuple[str, str]
Span = dict[str, Any]
Assessment = dict[str, Any]


@dataclass(frozen=True)
class ItemId:
    """A validated six-field WMT26 item identifier."""

    raw: str
    set_id: str
    source_language: str
    target_language: str
    domain: str
    document_id: str
    segment_id: str

    @property
    def language_pair(self) -> str:
        """Return the source-target language pair encoded by this item."""

        return f"{self.source_language}-{self.target_language}"


@dataclass(frozen=True)
class SubmissionDescriptor:
    """Filesystem location and identity parsed from a submission directory."""

    path: Path
    submission_id: str
    username: str
    phase_id: str
    suffix: str
    is_baseline: bool = False
    baseline_family: str = ""
    model_card_override: Path | None = None

    @property
    def baseline_name(self) -> str:
        """Return the family-independent baseline identity used in reports."""

        if not self.is_baseline:
            return ""
        return self.source_baseline_name

    @property
    def source_baseline_name(self) -> str:
        """Return the original baseline basename for cohort/pair matching."""

        if not self.is_baseline:
            return ""
        suffix = {
            "normal": "__normal",
            "deduplicated": "__deduplicated",
            "pre_dedup_pre_enhy_fix": "__pre_dedup_pre_enhy_fix",
        }.get(self.baseline_family)
        if suffix is not None:
            return self.path.name.removesuffix(suffix)
        return self.path.name

    @property
    def cache_name(self) -> str:
        """Return the physical baseline identity used only for MPP cache isolation."""

        return self.path.name if self.is_baseline else ""

    @property
    def report_label(self) -> str:
        """Human-readable identity used in mixed submission/baseline reports."""

        return (
            f"Baseline: {self.baseline_name}"
            if self.is_baseline
            else self.submission_id
        )

    @property
    def predictions_path(self) -> Path:
        """Return the conventional predictions JSONL path for this entry."""

        return self.path / "predictions.jsonl"

    @property
    def model_card_path(self) -> Path:
        """Return the conventional model-card JSON path for this entry."""

        return self.model_card_override or self.path / "model_card.json"


@dataclass(frozen=True)
class Issue:
    """A structured, reportable problem found during discovery or processing."""

    stage: str
    message: str
    submission_id: str = ""
    username: str = ""
    path: str = ""
    line_number: int | None = None
    severity: str = "error"


@dataclass(frozen=True)
class ModelCardTemplate:
    """Authoritative model-card questions and required-answer metadata."""

    path: Path
    entries_by_question: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ModelCardMetadata:
    """The non-sensitive model-card metadata used by reports and analyses."""

    organization_name: str | None = None
    submission_name: str | None = None
    submission_description: str | None = None
    reference_description: str | None = None
    model_size: str | None = None
    closed_source_model_name: str | None = None
    api_provider: str | None = None
    system_link: str | None = None
    evaluator_characteristics: Mapping[str, bool | None] = field(default_factory=dict)
    valid: bool = False


@dataclass(frozen=True)
class TestSetIndex:
    """Authoritative test pairs plus direction-level reference policy."""

    path: Path
    translation_lengths: Mapping[str, Mapping[str, int]]
    item_directions: Mapping[str, str]
    expected_pairs_by_direction: Mapping[str, frozenset[Pair]]
    challenge_row_count: int
    official_row_count: int
    # Exact Unicode-code-point length of the stored source string. Kept last
    # with a default so existing positional construction remains compatible.
    source_lengths: Mapping[str, int] = field(default_factory=dict)
    # Directions for which every official row supplies a usable ref.text.
    reference_available_directions: frozenset[str] = frozenset()
    # Direction -> system whose hypothesis equals ref.text on every row.
    reference_systems_by_direction: Mapping[str, str] = field(default_factory=dict)
    # Exact fourth item-ID field for every official item.
    item_domains: Mapping[str, str] = field(default_factory=dict)

    @property
    def official_directions(self) -> frozenset[str]:
        """Return all official translation directions in the test set."""

        return frozenset(self.expected_pairs_by_direction)

    @property
    def scorable_pairs_by_direction(self) -> Mapping[str, frozenset[Pair]]:
        """Return official pairs after removing direction reference systems."""

        return {
            direction: frozenset(
                pair
                for pair in pairs
                if pair[1] != self.reference_systems_by_direction.get(direction)
            )
            for direction, pairs in self.expected_pairs_by_direction.items()
        }

    def coverage_pairs_by_direction(
        self, *, reference_based: bool
    ) -> Mapping[str, frozenset[Pair]]:
        """Return pairs required for one evaluator's coverage classification."""

        if not reference_based:
            return self.expected_pairs_by_direction
        return {
            direction: pairs
            for direction, pairs in self.scorable_pairs_by_direction.items()
            if direction in self.reference_available_directions
        }


@dataclass(frozen=True)
class GroundTruthSet:
    """Normalized, test-set-supported gold annotations."""

    path: Path
    # item -> system -> tuple of normalized span lists, one per assessment
    assessments: Mapping[str, Mapping[str, tuple[list[Span], ...]]]
    pair_counts_by_direction: Mapping[str, int]
    zero_length_span_count: int = 0
    # item -> system -> annotator IDs aligned with assessments;
    # legacy singles may be None
    annotator_ids: Mapping[str, Mapping[str, tuple[str | None, ...]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class TerminationCoverage:
    """Gold-pair finish counts and the fixed five-percent coverage decision for one LP."""

    gold_pairs: int
    counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def invalid_pairs(self) -> int:
        """Return the number of explicit null or length gold-backed predictions."""

        return self.counts.get("null", 0) + self.counts.get("length", 0)

    @property
    def failure_rate(self) -> float | None:
        """Return the failure fraction, or None when this LP has no gold pairs."""

        return self.invalid_pairs / self.gold_pairs if self.gold_pairs else None

    @property
    def excluded(self) -> bool:
        """Test strictly above five percent using exact integer arithmetic."""

        return self.invalid_pairs * 20 > self.gold_pairs


@dataclass(frozen=True)
class CoverageReport:
    """Official completeness plus descriptive challenge-row coverage."""

    complete_directions: tuple[str, ...]
    partial_directions: tuple[str, ...]
    absent_directions: tuple[str, ...]
    provided_pair_counts: Mapping[str, int]
    expected_pair_counts: Mapping[str, int]
    official_rows: int
    challenge_rows: int
    total_rows: int
    all_official_complete: bool
    all_primary_complete: bool
    primary_only: bool
    termination_by_direction: Mapping[str, TerminationCoverage] = field(
        default_factory=dict
    )

    @property
    def termination_excluded_directions(self) -> tuple[str, ...]:
        """Return LPs excluded from ranking by their gold-aligned finish reasons."""

        return tuple(
            sorted(
                d
                for d, value in self.termination_by_direction.items()
                if value.excluded
            )
        )


@dataclass(frozen=True)
class SubmissionAnalysis:
    """Descriptive audit result for one submission without MPP evaluation."""

    descriptor: SubmissionDescriptor
    model_card: ModelCardMetadata
    coverage: CoverageReport
    issues: tuple[Issue, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether this submission audit result contains no errors."""

        return not any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class DirectionScore:
    """MPP values for one official translation direction."""

    language_pair: str
    precision: float
    recall: float
    f1: float
    aligned_assessments: int


@dataclass(frozen=True)
class PermutationDirection:
    """Aligned item-system/reference keys and weighted four-component rows.

    Components are a float64 NumPy matrix, one row per support key. Reference
    indices and weights identify the human assessments grouped into each row.
    """

    support: tuple[tuple[str, str, tuple[tuple[int, float], ...]], ...]
    components: NDArray[np.float64] = field(repr=False)


@dataclass(frozen=True)
class SubmissionScore:
    """Successful scoring output for one submission."""

    descriptor: SubmissionDescriptor
    model_card: ModelCardMetadata
    coverage: CoverageReport
    direction_scores: tuple[DirectionScore, ...]
    ignored_prediction_zero_length_spans: int = 0
    issues: tuple[Issue, ...] = ()
    permutation_inputs: Mapping[str, PermutationDirection] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def macro_precision(self) -> float:
        """Return the arithmetic mean precision across scored directions."""

        return sum(score.precision for score in self.direction_scores) / len(
            self.direction_scores
        )

    @property
    def macro_recall(self) -> float:
        """Return the arithmetic mean recall across scored directions."""

        return sum(score.recall for score in self.direction_scores) / len(
            self.direction_scores
        )

    @property
    def macro_f1(self) -> float:
        """Return the arithmetic mean F1 across scored directions."""

        return sum(score.f1 for score in self.direction_scores) / len(
            self.direction_scores
        )


@dataclass(frozen=True)
class BatchScoreResult:
    """Successful scores, issues, reference policy, matcher, and direction scope."""

    scores: tuple[SubmissionScore, ...]
    issues: tuple[Issue, ...]
    reference_systems_by_direction: Mapping[str, str] = field(default_factory=dict)
    directions_without_references: tuple[str, ...] = ()
    matching_algorithm: str = "greedy"
    selected_directions: tuple[str, ...] = ()
    gold_directions: tuple[str, ...] | None = None

    @property
    def failed(self) -> bool:
        """Return whether any batch-level scoring issue has error severity."""

        return any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class MatchingComparisonRow:
    """Greedy and optimal metrics for one entry and translation direction."""

    entry_type: str
    entry_id: str
    entry: str
    language_pair: str
    aligned_assessments: int
    greedy_precision: float
    greedy_recall: float
    greedy_f1: float
    optimal_precision: float
    optimal_recall: float
    optimal_f1: float
    delta_precision: float
    delta_recall: float
    delta_f1: float
    greedy_compute_seconds: float | None = None
    optimal_compute_seconds: float | None = None


@dataclass(frozen=True)
class MatchingComparisonTiming:
    """Per-entry timing totals for one matching algorithm."""

    entry_type: str
    entry_id: str
    entry: str
    algorithm: str
    metric_seconds: float | None
    wall_seconds: float
    computed: bool


@dataclass(frozen=True)
class MatchingComparisonResult:
    """Paired matching results, timings, and scoring issues."""

    rows: tuple[MatchingComparisonRow, ...]
    timings: tuple[MatchingComparisonTiming, ...]
    issues: tuple[Issue, ...] = ()
    selected_directions: tuple[str, ...] = ()
