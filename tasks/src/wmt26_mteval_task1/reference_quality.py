"""Matched reference-quality challenge indexing and MPP analysis."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from wmt26_mteval_task1.termination import (
    build_termination_coverage,
    read_gold_termination_statuses,
    termination_coverage_messages,
)
from wmt26_mteval_task1.constants import CHALLENGE_SET_ID, OFFICIAL_SET_ID
from wmt26_mteval_task1.io import (
    get_prediction_assessments,
    normalize_assessment,
    parse_item_id,
    read_jsonl,
    validate_model_card,
)
from wmt26_mteval_task1.models import (
    Issue,
    ModelCardMetadata,
    ModelCardTemplate,
    GroundTruthSet,
    SubmissionDescriptor,
    TestSetIndex,
)
from wmt26_mteval_task1.mpp_cache import MppCache
from wmt26_mteval_task1.reporting_policy import is_xcomet_baseline
from wmt26_mteval_task1.scoring import (
    PreparedAssessment,
    ScoredAssessment,
    aggregate_mpp_components,
    score_prepared_assessments,
)

OFFICIAL_VS_PSEUDO = "official_vs_pseudo"
NATIVE_VS_POSTEDIT = "native_vs_postedit"
NATIVE_VS_PSEUDO = "native_vs_pseudo"
POSTEDIT_VS_PSEUDO = "postedit_vs_pseudo"
COMPARISONS = (
    OFFICIAL_VS_PSEUDO,
    NATIVE_VS_POSTEDIT,
    NATIVE_VS_PSEUDO,
    POSTEDIT_VS_PSEUDO,
)
REFERENCE_CANDIDATE_SYSTEMS = frozenset(
    {"Human (from scratch)", "Human (postediting)", "PseudoRef"}
)
ALL_REFERENCE_BASED = "all_reference_based"
WITHOUT_XCOMET = "without_xcomet"
COHORTS = (ALL_REFERENCE_BASED, WITHOUT_XCOMET)


@dataclass(frozen=True)
class ReferenceMirror:
    """One official item and one RefQuality variant of the same MT sample."""

    language_pair: str
    official_item_id: str
    challenge_item_id: str
    official_reference_type: str
    challenge_reference_type: str
    common_systems: tuple[str, ...]
    reference_text_changed: bool


@dataclass(frozen=True)
class ReferenceQualityDirection:
    """A complete direction participating in one reference comparison."""

    comparison: str
    language_pair: str
    left_condition: str
    right_condition: str
    left_reference_type: str
    right_reference_type: str
    mirrors: tuple[ReferenceMirror, ...]

    @property
    def item_count(self) -> int:
        """Return the number of matched item pairs in this direction."""

        return len(self.mirrors)

    @property
    def changed_reference_count(self) -> int:
        """Return how many mirror pairs have different reference strings."""

        return sum(mirror.reference_text_changed for mirror in self.mirrors)

    @property
    def identical_reference_count(self) -> int:
        """Return how many mirror pairs have identical reference strings."""

        return self.item_count - self.changed_reference_count

    @property
    def shared_test_pair_count(self) -> int:
        """Return shared test-set item-system pairs before gold filtering."""

        return sum(len(mirror.common_systems) for mirror in self.mirrors)


@dataclass(frozen=True)
class ReferenceQualityIndex:
    """Strict RefQuality mirrors and their four supported comparisons."""

    path: Path
    directions_by_comparison: Mapping[str, Mapping[str, ReferenceQualityDirection]]
    refquality_row_count: int
    pseudo_row_count: int
    postedit_row_count: int
    prefix_counts: Mapping[str, int]


@dataclass(frozen=True)
class ReferenceQualityEligibility:
    """One entry's prediction-coverage status for one comparison."""

    descriptor: SubmissionDescriptor
    model_card: ModelCardMetadata | None
    comparison: str
    status: str
    complete_directions: tuple[str, ...] = ()
    partial_directions: tuple[str, ...] = ()
    absent_directions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceConditionScore:
    """MPP precision, recall, and F1 for one reference condition."""

    precision: float
    recall: float
    f1: float
    aligned_assessments: int


@dataclass(frozen=True)
class ReferenceQualityDirectionScore:
    """Paired left/right MPP scores for one translation direction."""

    language_pair: str
    left_reference_type: str
    right_reference_type: str
    item_count: int
    changed_reference_count: int
    identical_reference_count: int
    scored_pair_count: int
    left: ReferenceConditionScore
    right: ReferenceConditionScore


@dataclass(frozen=True)
class ReferenceQualityEntryScore:
    """One entry's direction and macro scores for one comparison."""

    descriptor: SubmissionDescriptor
    model_card: ModelCardMetadata
    comparison: str
    directions: tuple[ReferenceQualityDirectionScore, ...]
    macro_left: ReferenceConditionScore
    macro_right: ReferenceConditionScore

    @property
    def delta_macro_precision(self) -> float:
        """Return right-minus-left macro precision."""

        return self.macro_right.precision - self.macro_left.precision

    @property
    def delta_macro_recall(self) -> float:
        """Return right-minus-left macro recall."""

        return self.macro_right.recall - self.macro_left.recall

    @property
    def delta_macro_f1(self) -> float:
        """Return right-minus-left macro F1."""

        return self.macro_right.f1 - self.macro_left.f1


@dataclass(frozen=True)
class ReferenceQualityCohortSummary:
    """Equal-entry-weighted deltas for one comparison and evaluator cohort."""

    cohort: str
    comparison: str
    entry_count: int
    directions: tuple[str, ...]
    mean_delta_precision: float
    mean_delta_recall: float
    mean_delta_f1: float


@dataclass(frozen=True)
class ReferenceQualityResult:
    """Complete reference-quality scores, cohorts, coverage, and issues."""

    index: ReferenceQualityIndex
    scores_by_comparison: Mapping[str, tuple[ReferenceQualityEntryScore, ...]]
    common_directions_by_comparison: Mapping[str, tuple[str, ...]]
    eligibility: tuple[ReferenceQualityEligibility, ...]
    issues: tuple[Issue, ...]
    matching_algorithm: str = "greedy"
    scores_by_cohort_and_comparison: Mapping[
        str, Mapping[str, tuple[ReferenceQualityEntryScore, ...]]
    ] = field(default_factory=dict)
    common_directions_by_cohort_and_comparison: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ] = field(default_factory=dict)
    cohort_membership: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    cohort_summaries: tuple[ReferenceQualityCohortSummary, ...] = ()

    @property
    def failed(self) -> bool:
        """Return whether any entry-level error occurred."""

        return any(issue.severity == "error" for issue in self.issues)

    @property
    def cohort_scores_by_comparison(
        self,
    ) -> Mapping[str, Mapping[str, tuple[ReferenceQualityEntryScore, ...]]]:
        """Return cohort-indexed score tables under a concise public alias."""

        return self.scores_by_cohort_and_comparison

    @property
    def common_directions_by_cohort(
        self,
    ) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
        """Return fixed comparison directions for every evaluator cohort."""

        return self.common_directions_by_cohort_and_comparison


@dataclass
class _EntryPreparation:
    """Validated metadata, relevant predictions, and provisional eligibility."""

    descriptor: SubmissionDescriptor
    model_card: ModelCardMetadata
    predictions: dict[str, dict[str, Any]]
    eligibility: dict[str, ReferenceQualityEligibility]


@dataclass(frozen=True)
class _ParsedVariant:
    """Validated condition and stripped official document identifier."""

    prefix: str
    condition: str
    document_id: str


def _display_reference_type(raw_type: str) -> str:
    """Translate the raw human label to the requested native terminology."""

    return "native (human)" if raw_type == "human" else raw_type


def _parse_refquality_document(document_id: str) -> _ParsedVariant | None:
    """Parse an allowed RefQuality document prefix, or ignore another set."""

    remaining = document_id
    tokens: list[str] = []
    while True:
        if remaining.startswith("RefQuality-PE_"):
            tokens.append("RefQuality-PE")
            remaining = remaining[len("RefQuality-PE_") :]
        elif remaining.startswith("RefQuality-PD_"):
            tokens.append("RefQuality-PD")
            remaining = remaining[len("RefQuality-PD_") :]
        else:
            break
    if not tokens:
        return None
    token_tuple = tuple(tokens)
    if token_tuple == ("RefQuality-PE",):
        condition = "postedit"
    elif token_tuple in {
        ("RefQuality-PD",),
        ("RefQuality-PD", "RefQuality-PE"),
    }:
        condition = "pseudo"
    else:
        raise ValueError(
            f"unsupported RefQuality document prefix sequence {token_tuple!r}"
        )
    if not remaining:
        raise ValueError("RefQuality document prefix has no official document ID")
    return _ParsedVariant("+".join(tokens), condition, remaining)


def _reference_payload(record: Mapping[str, Any], context: str) -> tuple[str, str]:
    """Return a validated reference type and text from one test-set row."""

    reference = record.get("ref")
    if not isinstance(reference, dict):
        raise ValueError(f"{context}: ref must be an object")
    reference_type = reference.get("type")
    reference_text = reference.get("text")
    if reference_type not in {"human", "postedit", "pseudo"}:
        raise ValueError(f"{context}: ref.type must be human, postedit, or pseudo")
    if not isinstance(reference_text, str):
        raise ValueError(f"{context}: ref.text must be a string")
    return reference_type, reference_text


def _hypotheses(record: Mapping[str, Any], context: str) -> dict[str, str]:
    """Return a strictly validated non-empty test-set hypothesis mapping."""

    hypotheses = record.get("hyps")
    if not isinstance(hypotheses, dict) or not hypotheses:
        raise ValueError(f"{context}: hyps must be a non-empty object")
    if any(not isinstance(system, str) or not system for system in hypotheses):
        raise ValueError(f"{context}: system names must be non-empty strings")
    if any(not isinstance(text, str) for text in hypotheses.values()):
        raise ValueError(f"{context}: every hypothesis must be a string")
    return hypotheses


def load_reference_quality_index(
    path: Path,
    test_set: TestSetIndex,
) -> ReferenceQualityIndex:
    """Load and strictly match all RefQuality rows to complete official directions."""

    official_records: dict[str, tuple[int, dict[str, Any]]] = {}
    challenge_records: list[tuple[int, dict[str, Any], _ParsedVariant]] = []
    seen_challenge_ids: set[str] = set()
    prefix_counts: Counter[str] = Counter()
    for line_number, record in read_jsonl(path):
        context = f"{path}:{line_number}"
        item = parse_item_id(record.get("item_id"), context)
        if item.set_id == OFFICIAL_SET_ID:
            official_records[item.raw] = (line_number, record)
            continue
        if item.set_id != CHALLENGE_SET_ID:
            continue
        variant = _parse_refquality_document(item.document_id)
        if variant is None:
            continue
        if item.raw in seen_challenge_ids:
            raise ValueError(f"{context}: duplicate RefQuality item_id {item.raw!r}")
        seen_challenge_ids.add(item.raw)
        prefix_counts[variant.prefix] += 1
        challenge_records.append((line_number, record, variant))

    if not challenge_records:
        raise ValueError(f"{path}: no RefQuality challenge rows were found")

    mirrors_by_condition: dict[str, dict[str, dict[str, ReferenceMirror]]] = {
        "pseudo": {},
        "postedit": {},
    }
    for line_number, challenge, variant in challenge_records:
        context = f"{path}:{line_number}"
        challenge_item = parse_item_id(challenge["item_id"], context)
        official_id = "_###_".join(
            (
                OFFICIAL_SET_ID,
                challenge_item.source_language,
                challenge_item.target_language,
                challenge_item.domain,
                variant.document_id,
                challenge_item.segment_id,
            )
        )
        official_value = official_records.get(official_id)
        if official_value is None:
            raise ValueError(
                f"{context}: RefQuality row has no official mirror {official_id!r}"
            )
        official_line, official = official_value
        official_context = f"{path}:{official_line}"
        official_item = parse_item_id(official_id, official_context)
        if challenge.get("src") != official.get("src") or not isinstance(
            challenge.get("src"), str
        ):
            raise ValueError(
                f"{context}: source text differs from official mirror {official_id!r}"
            )
        official_type, official_text = _reference_payload(official, official_context)
        challenge_type, challenge_text = _reference_payload(challenge, context)
        if challenge_type != variant.condition:
            raise ValueError(
                f"{context}: prefix denotes {variant.condition!r} but "
                f"ref.type is {challenge_type!r}"
            )
        if variant.condition == "postedit" and official_type != "human":
            raise ValueError(
                f"{context}: a postedit comparison requires an official human reference"
            )
        if variant.condition == "pseudo" and official_type not in {
            "human",
            "postedit",
        }:
            raise ValueError(
                f"{context}: pseudo comparison requires an official human or postedit reference"
            )
        official_hypotheses = _hypotheses(official, official_context)
        challenge_hypotheses = _hypotheses(challenge, context)
        challenge_only = set(challenge_hypotheses) - set(official_hypotheses)
        if challenge_only:
            raise ValueError(
                f"{context}: challenge-only hypotheses are not allowed: "
                + ", ".join(sorted(challenge_only))
            )
        mismatched = sorted(
            system
            for system, text in challenge_hypotheses.items()
            if official_hypotheses[system] != text
        )
        if mismatched:
            raise ValueError(
                f"{context}: shared hypotheses differ from the official mirror: "
                + ", ".join(mismatched[:3])
            )
        direction = official_item.language_pair
        by_official = mirrors_by_condition[variant.condition].setdefault(direction, {})
        if official_id in by_official:
            raise ValueError(
                f"{context}: duplicate {variant.condition} mirror for {official_id!r}"
            )
        by_official[official_id] = ReferenceMirror(
            language_pair=direction,
            official_item_id=official_id,
            challenge_item_id=challenge_item.raw,
            official_reference_type=official_type,
            challenge_reference_type=challenge_type,
            common_systems=tuple(
                sorted(
                    (set(official_hypotheses) & set(challenge_hypotheses))
                    - REFERENCE_CANDIDATE_SYSTEMS
                )
            ),
            reference_text_changed=official_text != challenge_text,
        )

    directions_by_comparison: dict[str, dict[str, ReferenceQualityDirection]] = {
        comparison: {} for comparison in COMPARISONS
    }

    def add_directions(
        condition: str,
        comparison: str,
        by_direction: Mapping[str, Mapping[str, ReferenceMirror]],
    ) -> None:
        """Validate and register one complete comparison family."""

        for direction, by_official in sorted(by_direction.items()):
            expected_ids = {
                item_id
                for item_id, item_direction in test_set.item_directions.items()
                if item_direction == direction
            }
            actual_ids = set(by_official)
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                raise ValueError(
                    f"{path}: {condition} RefQuality direction {direction!r} is "
                    f"not a complete official mirror; missing={len(missing)}, "
                    f"extra={len(extra)}"
                )
            mirrors = tuple(by_official[item_id] for item_id in sorted(by_official))
            challenge_types = {mirror.challenge_reference_type for mirror in mirrors}
            official_types = {mirror.official_reference_type for mirror in mirrors}
            if len(official_types) != 1 or len(challenge_types) != 1:
                raise ValueError(
                    f"{path}: reference types vary within {condition} direction "
                    f"{direction!r}"
                )
            directions_by_comparison[comparison][direction] = ReferenceQualityDirection(
                comparison=comparison,
                language_pair=direction,
                left_condition="official",
                right_condition=condition,
                left_reference_type=_display_reference_type(next(iter(official_types))),
                right_reference_type=_display_reference_type(
                    next(iter(challenge_types))
                ),
                mirrors=mirrors,
            )

    add_directions("pseudo", OFFICIAL_VS_PSEUDO, mirrors_by_condition["pseudo"])
    add_directions("postedit", NATIVE_VS_POSTEDIT, mirrors_by_condition["postedit"])
    for direction, definition in directions_by_comparison[
        OFFICIAL_VS_PSEUDO
    ].items():
        official_type = definition.mirrors[0].official_reference_type
        comparison = (
            NATIVE_VS_PSEUDO if official_type == "human" else POSTEDIT_VS_PSEUDO
        )
        directions_by_comparison[comparison][direction] = replace(
            definition,
            comparison=comparison,
            left_condition="native" if official_type == "human" else "postedit",
        )

    pseudo_count = sum(
        len(mirrors) for mirrors in mirrors_by_condition["pseudo"].values()
    )
    postedit_count = sum(
        len(mirrors) for mirrors in mirrors_by_condition["postedit"].values()
    )
    return ReferenceQualityIndex(
        path=path,
        directions_by_comparison={
            comparison: dict(directions)
            for comparison, directions in directions_by_comparison.items()
        },
        refquality_row_count=len(challenge_records),
        pseudo_row_count=pseudo_count,
        postedit_row_count=postedit_count,
        prefix_counts=dict(sorted(prefix_counts.items())),
    )


def _relevant_item_ids(index: ReferenceQualityIndex) -> frozenset[str]:
    """Return every official and challenge item used by either comparison."""

    return frozenset(
        item_id
        for directions in index.directions_by_comparison.values()
        for direction in directions.values()
        for mirror in direction.mirrors
        for item_id in (mirror.official_item_id, mirror.challenge_item_id)
    )


def _load_relevant_predictions(
    descriptor: SubmissionDescriptor,
    relevant_ids: frozenset[str],
) -> dict[str, dict[str, Any]]:
    """Stream one entry and retain validated predictions for relevant item IDs."""

    predictions: dict[str, dict[str, Any]] = {}
    for line_number, record in read_jsonl(descriptor.predictions_path):
        context = f"{descriptor.predictions_path}:{line_number}"
        item = parse_item_id(record.get("item_id"), context)
        if item.raw not in relevant_ids:
            continue
        if item.raw in predictions:
            raise ValueError(f"{context}: duplicate relevant item_id {item.raw!r}")
        predictions[item.raw] = get_prediction_assessments(record, context)
    return predictions


def _coverage_status(
    direction: ReferenceQualityDirection,
    predictions: Mapping[str, Mapping[str, Any]],
    test_set: TestSetIndex,
) -> str:
    """Classify one comparison direction as complete, partial, or absent."""

    challenge_ids = {mirror.challenge_item_id for mirror in direction.mirrors}
    if not (challenge_ids & set(predictions)):
        return "absent"
    mirror_item_ids = {mirror.official_item_id for mirror in direction.mirrors}
    expected_official = frozenset(
        (item_id, system)
        for item_id, system in test_set.coverage_pairs_by_direction(
            reference_based=True
        )[direction.language_pair]
        if item_id in mirror_item_ids and system not in REFERENCE_CANDIDATE_SYSTEMS
    )
    expected_challenge = {
        (mirror.challenge_item_id, system)
        for mirror in direction.mirrors
        for system in mirror.common_systems
    }
    provided_official = {
        (item_id, system)
        for item_id, systems in predictions.items()
        for system in systems
        if (item_id, system) in expected_official
    }
    provided_challenge = {
        (item_id, system)
        for item_id, systems in predictions.items()
        for system in systems
        if (item_id, system) in expected_challenge
    }
    if (
        expected_official <= provided_official
        and expected_challenge <= provided_challenge
    ):
        return "complete"
    return "partial"


def _eligibility_for_entry(
    descriptor: SubmissionDescriptor,
    metadata: ModelCardMetadata,
    predictions: Mapping[str, Mapping[str, Any]],
    index: ReferenceQualityIndex,
    test_set: TestSetIndex,
    excluded_directions: tuple[str, ...] = (),
) -> dict[str, ReferenceQualityEligibility]:
    """Build strict per-comparison direction coverage for one reference-based entry."""

    eligibility: dict[str, ReferenceQualityEligibility] = {}
    for comparison in COMPARISONS:
        statuses = {
            direction: _coverage_status(definition, predictions, test_set)
            for direction, definition in index.directions_by_comparison[
                comparison
            ].items()
        }
        for direction in excluded_directions:
            if statuses.get(direction) == "complete":
                statuses[direction] = "partial"
        complete = tuple(
            sorted(
                direction
                for direction, status in statuses.items()
                if status == "complete"
            )
        )
        partial = tuple(
            sorted(
                direction
                for direction, status in statuses.items()
                if status == "partial"
            )
        )
        absent = tuple(
            sorted(
                direction
                for direction, status in statuses.items()
                if status == "absent"
            )
        )
        status = (
            "partial" if partial else ("candidate" if complete else "no_predictions")
        )
        eligibility[comparison] = ReferenceQualityEligibility(
            descriptor,
            metadata,
            comparison,
            status,
            complete,
            partial,
            absent,
        )
    return eligibility


def _entry_issue(
    descriptor: SubmissionDescriptor,
    stage: str,
    message: str,
) -> Issue:
    """Create a consistently identified entry-level analysis error."""

    return Issue(
        stage=stage,
        message=message,
        submission_id=descriptor.report_label,
        username=descriptor.username,
        path=str(descriptor.path),
    )


def _scorable_directions(
    index: ReferenceQualityIndex,
    ground_truth: GroundTruthSet,
    comparison: str,
) -> frozenset[str]:
    """Return comparison directions having at least one matched human-gold pair."""

    return frozenset(
        language_pair
        for language_pair, direction in index.directions_by_comparison[
            comparison
        ].items()
        if any(
            set(mirror.common_systems)
            & set(ground_truth.assessments.get(mirror.official_item_id, {}))
            for mirror in direction.mirrors
        )
    )


def _analysis_directions(
    index: ReferenceQualityIndex,
    ground_truth: GroundTruthSet,
) -> dict[str, tuple[str, ...]]:
    """Return the fixed, fully scorable direction set for each comparison."""

    return {
        comparison: tuple(
            sorted(
                set(index.directions_by_comparison[comparison])
                & set(_scorable_directions(index, ground_truth, comparison))
            )
        )
        for comparison in COMPARISONS
    }


def _prepare_entry_assessments(
    preparation: _EntryPreparation,
    index: ReferenceQualityIndex,
    ground_truth: GroundTruthSet,
    common_directions: Mapping[str, tuple[str, ...]],
    test_set: TestSetIndex,
) -> tuple[
    tuple[PreparedAssessment, ...],
    dict[tuple[str, str, str], tuple[tuple[str, str, int], ...]],
    int,
]:
    """Normalize and deduplicate all physical condition assessments for an entry."""

    prepared_by_key: dict[tuple[str, str, int], PreparedAssessment] = {}
    group_keys: dict[tuple[str, str, str], list[tuple[str, str, int]]] = {}
    normalized_predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    ignored_zero_lengths = 0
    for comparison in COMPARISONS:
        eligibility = preparation.eligibility[comparison]
        if eligibility.status != "eligible":
            continue
        for language_pair in common_directions[comparison]:
            direction = index.directions_by_comparison[comparison][language_pair]
            for side in ("left", "right"):
                group = group_keys.setdefault((comparison, language_pair, side), [])
                for mirror in direction.mirrors:
                    prediction_item_id = (
                        mirror.official_item_id
                        if side == "left"
                        else mirror.challenge_item_id
                    )
                    item_ground_truth = ground_truth.assessments.get(
                        mirror.official_item_id, {}
                    )
                    for system in sorted(
                        set(mirror.common_systems) & set(item_ground_truth)
                    ):
                        normalized_key = (prediction_item_id, system)
                        normalized = normalized_predictions.get(normalized_key)
                        if normalized is None:
                            normalized, ignored = normalize_assessment(
                                preparation.predictions[prediction_item_id][system],
                                test_set.translation_lengths[mirror.official_item_id][
                                    system
                                ],
                                f"prediction item {prediction_item_id!r}, system {system!r}",
                            )
                            normalized_predictions[normalized_key] = normalized
                            ignored_zero_lengths += ignored
                        for reference_index, reference in enumerate(
                            item_ground_truth[system]
                        ):
                            key = (prediction_item_id, system, reference_index)
                            if key not in prepared_by_key:
                                prepared_by_key[key] = PreparedAssessment(
                                    item_id=prediction_item_id,
                                    system=system,
                                    language_pair=language_pair,
                                    source_length=test_set.source_lengths.get(
                                        mirror.official_item_id
                                    ),
                                    target_length=test_set.translation_lengths[
                                        mirror.official_item_id
                                    ][system],
                                    prediction=normalized,
                                    reference=reference,
                                    reference_index=reference_index,
                                )
                            group.append(key)
    return (
        tuple(prepared_by_key[key] for key in sorted(prepared_by_key)),
        {key: tuple(values) for key, values in group_keys.items()},
        ignored_zero_lengths,
    )


def _condition_score(
    records: Iterable[ScoredAssessment],
) -> ReferenceConditionScore:
    """Aggregate one condition's raw components into reportable MPP values."""

    values = tuple(records)
    precision, recall, f1 = aggregate_mpp_components(values)
    return ReferenceConditionScore(precision, recall, f1, len(values))


def _macro_condition_score(
    scores: Iterable[ReferenceConditionScore],
) -> ReferenceConditionScore:
    """Macro-average condition metrics across a shared set of directions."""

    values = tuple(scores)
    return ReferenceConditionScore(
        precision=sum(score.precision for score in values) / len(values),
        recall=sum(score.recall for score in values) / len(values),
        f1=sum(score.f1 for score in values) / len(values),
        aligned_assessments=sum(score.aligned_assessments for score in values),
    )


def _score_entry(
    preparation: _EntryPreparation,
    index: ReferenceQualityIndex,
    ground_truth: GroundTruthSet,
    common_directions: Mapping[str, tuple[str, ...]],
    test_set: TestSetIndex,
    *,
    mpp_cache: MppCache | None,
    recompute_mpp: bool,
    matching_algorithm: str,
) -> dict[str, ReferenceQualityEntryScore]:
    """Score all eligible comparison conditions for one prepared entry."""

    assessments, group_keys, ignored = _prepare_entry_assessments(
        preparation, index, ground_truth, common_directions, test_set
    )
    if not assessments:
        return {}
    scored = score_prepared_assessments(
        preparation.descriptor,
        assessments,
        mpp_cache=mpp_cache,
        recompute_mpp=recompute_mpp,
        matching_algorithm=matching_algorithm,
    )
    scored_by_key = {
        (
            record.assessment.item_id,
            record.assessment.system,
            record.assessment.reference_index,
        ): record
        for record in scored
    }
    if ignored:
        logging.warning(
            "%s: ignored %d zero-length RefQuality prediction spans",
            preparation.descriptor.report_label,
            ignored,
        )

    output: dict[str, ReferenceQualityEntryScore] = {}
    for comparison in COMPARISONS:
        if preparation.eligibility[comparison].status != "eligible":
            continue
        direction_scores: list[ReferenceQualityDirectionScore] = []
        for language_pair in common_directions[comparison]:
            direction = index.directions_by_comparison[comparison][language_pair]
            left_keys = group_keys[(comparison, language_pair, "left")]
            right_keys = group_keys[(comparison, language_pair, "right")]
            left = _condition_score(scored_by_key[key] for key in left_keys)
            right = _condition_score(scored_by_key[key] for key in right_keys)
            scored_pair_count = len({(key[0], key[1]) for key in left_keys})
            direction_scores.append(
                ReferenceQualityDirectionScore(
                    language_pair=language_pair,
                    left_reference_type=direction.left_reference_type,
                    right_reference_type=direction.right_reference_type,
                    item_count=direction.item_count,
                    changed_reference_count=direction.changed_reference_count,
                    identical_reference_count=direction.identical_reference_count,
                    scored_pair_count=scored_pair_count,
                    left=left,
                    right=right,
                )
            )
        values = tuple(direction_scores)
        if not values:
            continue
        output[comparison] = ReferenceQualityEntryScore(
            descriptor=preparation.descriptor,
            model_card=preparation.model_card,
            comparison=comparison,
            directions=values,
            macro_left=_macro_condition_score(score.left for score in values),
            macro_right=_macro_condition_score(score.right for score in values),
        )
    return output


def analyze_reference_quality(
    entries: Iterable[SubmissionDescriptor],
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    model_card_template: ModelCardTemplate | None,
    index: ReferenceQualityIndex,
    *,
    mpp_cache: MppCache | None = None,
    recompute_mpp: bool = False,
    matching_algorithm: str = "greedy",
) -> ReferenceQualityResult:
    """Discover cohorts and score all matched reference-quality comparisons."""

    preparations: list[_EntryPreparation] = []
    eligibility: list[ReferenceQualityEligibility] = []
    issues: list[Issue] = []
    relevant_ids = _relevant_item_ids(index)
    for descriptor in entries:
        try:
            metadata = validate_model_card(
                descriptor.model_card_path, model_card_template
            )
        except (OSError, ValueError) as exc:
            issues.append(_entry_issue(descriptor, "model_card", str(exc)))
            eligibility.extend(
                ReferenceQualityEligibility(
                    descriptor, None, comparison, "invalid_model_card"
                )
                for comparison in COMPARISONS
            )
            continue
        reference_based = metadata.evaluator_characteristics.get("reference_use")
        if type(reference_based) is not bool:
            message = (
                f"{descriptor.model_card_path}: the model card must provide a "
                "valid required reference-use answer"
            )
            issues.append(_entry_issue(descriptor, "model_card", message))
            eligibility.extend(
                ReferenceQualityEligibility(
                    descriptor, metadata, comparison, "invalid_model_card"
                )
                for comparison in COMPARISONS
            )
            continue
        if not reference_based:
            eligibility.extend(
                ReferenceQualityEligibility(
                    descriptor, metadata, comparison, "reference_free"
                )
                for comparison in COMPARISONS
            )
            continue
        try:
            predictions = _load_relevant_predictions(descriptor, relevant_ids)
            termination = build_termination_coverage(
                test_set,
                ground_truth,
                read_gold_termination_statuses(
                    descriptor.predictions_path, test_set, ground_truth
                ),
                reference_based=True,
            )
            issues.extend(
                replace(
                    _entry_issue(descriptor, "termination", message), severity="warning"
                )
                for message in termination_coverage_messages(termination)
            )
            entry_eligibility = _eligibility_for_entry(
                descriptor,
                metadata,
                predictions,
                index,
                test_set,
                tuple(d for d, value in termination.items() if value.excluded),
            )
        except (OSError, ValueError) as exc:
            issues.append(_entry_issue(descriptor, "predictions", str(exc)))
            eligibility.extend(
                ReferenceQualityEligibility(
                    descriptor, metadata, comparison, "invalid_predictions"
                )
                for comparison in COMPARISONS
            )
            continue
        for comparison, status in entry_eligibility.items():
            structural_partial = tuple(
                direction
                for direction in status.partial_directions
                if _coverage_status(
                    index.directions_by_comparison[comparison][direction],
                    predictions,
                    test_set,
                )
                == "partial"
            )
            if structural_partial:
                message = (
                    f"{descriptor.predictions_path}: partial {comparison} "
                    "coverage is not allowed; directions: "
                    + ", ".join(structural_partial)
                )
                issues.append(
                    _entry_issue(descriptor, "reference_quality_coverage", message)
                )
        preparation = _EntryPreparation(
            descriptor, metadata, predictions, entry_eligibility
        )
        preparations.append(preparation)

    common = _analysis_directions(index, ground_truth)
    for preparation in preparations:
        for comparison in COMPARISONS:
            current = preparation.eligibility[comparison]
            if current.status == "candidate":
                current = replace(
                    current,
                    status=(
                        "no_scorable_directions"
                        if not common[comparison]
                        else "eligible"
                        if set(common[comparison]).issubset(current.complete_directions)
                        else "incomplete_directions"
                    ),
                )
                preparation.eligibility[comparison] = current
            eligibility.append(current)

    scores: dict[str, list[ReferenceQualityEntryScore]] = {
        comparison: [] for comparison in COMPARISONS
    }
    for preparation in preparations:
        if not any(
            preparation.eligibility[comparison].status == "eligible"
            for comparison in COMPARISONS
        ):
            continue
        try:
            entry_scores = _score_entry(
                preparation,
                index,
                ground_truth,
                common,
                test_set,
                mpp_cache=mpp_cache,
                recompute_mpp=recompute_mpp,
                matching_algorithm=matching_algorithm,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            issues.append(_entry_issue(preparation.descriptor, "scoring", str(exc)))
            for index_value, row in enumerate(eligibility):
                if (
                    row.descriptor == preparation.descriptor
                    and row.status == "eligible"
                ):
                    eligibility[index_value] = replace(row, status="scoring_failed")
            continue
        for comparison, score in entry_scores.items():
            scores[comparison].append(score)

    def score_sort_key(score: ReferenceQualityEntryScore) -> tuple[bool, int, str]:
        """Return the deterministic ordering shared by every cohort table."""

        return (
            score.descriptor.is_baseline,
            (
                int(score.descriptor.submission_id)
                if not score.descriptor.is_baseline
                else 0
            ),
            score.descriptor.baseline_name.casefold(),
        )

    for comparison in COMPARISONS:
        scores[comparison].sort(key=score_sort_key)

    scores_by_cohort = {
        ALL_REFERENCE_BASED: {
            comparison: tuple(values) for comparison, values in scores.items()
        },
        WITHOUT_XCOMET: {
            comparison: tuple(
                score
                for score in values
                if not is_xcomet_baseline(score.descriptor)
            )
            for comparison, values in scores.items()
        },
    }
    common_by_cohort = {cohort: common for cohort in COHORTS}
    cohort_membership = {
        cohort: tuple(
            sorted(
                {
                    score.descriptor.report_label
                    for values in comparison_scores.values()
                    for score in values
                }
            )
        )
        for cohort, comparison_scores in scores_by_cohort.items()
    }
    cohort_summaries: list[ReferenceQualityCohortSummary] = []
    for cohort in COHORTS:
        for comparison in COMPARISONS:
            values = scores_by_cohort[cohort][comparison]
            if values:
                cohort_summaries.append(
                    ReferenceQualityCohortSummary(
                        cohort=cohort,
                        comparison=comparison,
                        entry_count=len(values),
                        directions=common_by_cohort[cohort][comparison],
                        mean_delta_precision=sum(
                            score.delta_macro_precision for score in values
                        )
                        / len(values),
                        mean_delta_recall=sum(
                            score.delta_macro_recall for score in values
                        )
                        / len(values),
                        mean_delta_f1=sum(score.delta_macro_f1 for score in values)
                        / len(values),
                    )
                )
    eligibility.sort(
        key=lambda row: (
            row.comparison,
            row.descriptor.is_baseline,
            int(row.descriptor.submission_id) if not row.descriptor.is_baseline else 0,
            row.descriptor.baseline_name.casefold(),
        )
    )
    return ReferenceQualityResult(
        index=index,
        scores_by_comparison={
            comparison: tuple(values) for comparison, values in scores.items()
        },
        common_directions_by_comparison=common,
        eligibility=tuple(eligibility),
        issues=tuple(issues),
        matching_algorithm=matching_algorithm,
        scores_by_cohort_and_comparison={
            cohort: {
                comparison: tuple(values)
                for comparison, values in comparison_scores.items()
            }
            for cohort, comparison_scores in scores_by_cohort.items()
        },
        common_directions_by_cohort_and_comparison=common_by_cohort,
        cohort_membership=cohort_membership,
        cohort_summaries=tuple(cohort_summaries),
    )
