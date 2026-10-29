"""Strict per-submission and resilient batch MPP scoring."""

from __future__ import annotations

import logging
from time import perf_counter
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from span_mt_metrics_eval import MetricResult, compute

from wmt26_mteval_task1.constants import (
    MATCHING_ALGORITHMS,
    mpp_metric_options,
    OFFICIAL_SET_ID,
    PRIMARY_LANGUAGE_PAIRS,
)
from wmt26_mteval_task1.coverage import build_coverage_report
from wmt26_mteval_task1.io import (
    get_prediction_assessments,
    normalize_assessment,
    parse_item_id,
    read_jsonl,
    validate_model_card,
)
from wmt26_mteval_task1.models import (
    BatchScoreResult,
    MatchingComparisonResult,
    MatchingComparisonRow,
    MatchingComparisonTiming,
    CoverageReport,
    DirectionScore,
    Issue,
    ModelCardMetadata,
    ModelCardTemplate,
    GroundTruthSet,
    PermutationDirection,
    Span,
    SubmissionDescriptor,
    SubmissionScore,
    TestSetIndex,
)
from wmt26_mteval_task1.mpp_cache import MppCache
from wmt26_mteval_task1.termination import (
    build_termination_coverage,
    termination_status,
    termination_coverage_messages,
)


@dataclass(frozen=True)
class PreparedAssessment:
    """One prediction/reference comparison retained for MPP."""

    item_id: str
    system: str
    language_pair: str
    source_length: int | None
    target_length: int
    prediction: list[Span]
    reference: list[Span]
    reference_index: int


@dataclass(frozen=True)
class PreparedSubmission:
    """Validated aligned records plus the scorer's coverage information."""

    assessments: tuple[PreparedAssessment, ...]
    coverage: CoverageReport
    ignored_zero_length_spans: int
    issues: tuple[Issue, ...] = ()


@dataclass(frozen=True)
class MppComponents:
    """Raw side-specific components for one aligned MPP assessment."""

    precision_numerator: float
    precision_denominator: float
    recall_numerator: float
    recall_denominator: float


@dataclass(frozen=True)
class ScoredAssessment:
    """A prepared assessment plus its reusable raw MPP components."""

    assessment: PreparedAssessment
    segment_weight: float
    components: MppComponents


def _group_assessments_by_direction(
    assessments: Iterable[PreparedAssessment],
) -> dict[str, list[PreparedAssessment]]:
    """Group prepared assessments by translation direction."""

    grouped: dict[str, list[PreparedAssessment]] = {}
    for assessment in assessments:
        grouped.setdefault(assessment.language_pair, []).append(assessment)
    return grouped


def _segment_weights(
    assessments: Iterable[PreparedAssessment],
) -> list[float]:
    """Give every item-system pair total aggregation weight one."""

    records = list(assessments)
    annotation_counts = Counter(
        (assessment.item_id, assessment.system) for assessment in records
    )
    return [
        1.0 / annotation_counts[(assessment.item_id, assessment.system)]
        for assessment in records
    ]


def _group_metric_inputs(
    assessments: Iterable[PreparedAssessment],
) -> dict[str, tuple[list[list[Span]], list[list[Span]]]]:
    """Build per-direction prediction and reference inputs for metric calls."""

    return {
        language_pair: (
            [assessment.prediction for assessment in records],
            [assessment.reference for assessment in records],
        )
        for language_pair, records in _group_assessments_by_direction(
            assessments
        ).items()
    }


def _normalize_selected_directions(
    test_set: TestSetIndex,
    directions: Iterable[str] | None,
) -> tuple[str, ...] | None:
    """Validate and normalize an optional score-direction selection."""

    if directions is None:
        return None
    selected = tuple(sorted(set(directions)))
    unknown = sorted(set(selected) - test_set.official_directions)
    if unknown:
        raise ValueError("unknown translation direction(s): " + ", ".join(unknown))
    if not selected:
        raise ValueError("at least one translation direction must be selected")
    return selected


def _prepare_submission(
    submission: SubmissionDescriptor,
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    *,
    require_complete_predictions: bool,
    reference_based: bool = False,
    directions: Iterable[str] | None = None,
    allow_no_selected_predictions: bool = False,
) -> PreparedSubmission:
    """Validate/align predictions; optionally return empty coverage for cohort classification."""

    assessments_for_metric: list[PreparedAssessment] = []
    provided: dict[str, set[tuple[str, str]]] = {}
    observed_directions: set[str] = set()
    seen_official_items: set[str] = set()
    official_rows = 0
    challenge_rows = 0
    total_rows = 0
    ignored_zero_lengths = 0
    path = submission.predictions_path
    selected_directions = _normalize_selected_directions(test_set, directions)
    expected_by_direction = test_set.coverage_pairs_by_direction(
        reference_based=reference_based
    )
    if selected_directions is not None:
        selected_set = set(selected_directions)
        expected_by_direction = {
            direction: pairs
            for direction, pairs in expected_by_direction.items()
            if direction in selected_set
        }
    applicable_directions = set(expected_by_direction)
    discarded_rows_by_direction: Counter[str] = Counter()
    termination_statuses: dict[tuple[str, str], str] = {}

    for line_number, record in read_jsonl(path):
        total_rows += 1
        context = f"{path}:{line_number}"
        item = parse_item_id(record.get("item_id"), context)
        if item.set_id != OFFICIAL_SET_ID:
            challenge_rows += 1
            continue
        if (
            selected_directions is not None
            and item.language_pair not in selected_directions
        ):
            continue
        official_rows += 1
        if item.raw in seen_official_items:
            raise ValueError(f"{context}: duplicate item_id {item.raw!r}")
        seen_official_items.add(item.raw)
        item_lengths = test_set.translation_lengths.get(item.raw)
        if item_lengths is None:
            continue
        if reference_based and item.language_pair not in applicable_directions:
            discarded_rows_by_direction[item.language_pair] += 1
            continue
        prediction_assessments = get_prediction_assessments(record, context)

        observed_directions.add(item.language_pair)
        provided.setdefault(item.language_pair, set()).update(
            (item.raw, system)
            for system in prediction_assessments
            if system in item_lengths
        )
        item_ground_truth = ground_truth.assessments.get(item.raw)
        if item_ground_truth is None:
            continue
        retained_systems = sorted(set(prediction_assessments) & set(item_ground_truth))
        for system in retained_systems:
            termination_statuses[(item.raw, system)] = termination_status(
                record, system
            )
            normalized_prediction, ignored = normalize_assessment(
                prediction_assessments[system],
                item_lengths[system],
                f"prediction item {item.raw!r}, system {system!r}",
            )
            ignored_zero_lengths += ignored
            for reference_index, normalized_reference in enumerate(
                item_ground_truth[system]
            ):
                assessments_for_metric.append(
                    PreparedAssessment(
                        item_id=item.raw,
                        system=system,
                        language_pair=item.language_pair,
                        source_length=test_set.source_lengths.get(item.raw),
                        target_length=item_lengths[system],
                        prediction=normalized_prediction,
                        reference=normalized_reference,
                        reference_index=reference_index,
                    )
                )

    issues: list[Issue] = []
    termination_coverage = build_termination_coverage(
        test_set,
        ground_truth,
        termination_statuses,
        reference_based=reference_based,
        directions=selected_directions,
    )
    for termination_message in termination_coverage_messages(termination_coverage):
        logging.warning("%s: %s", submission.report_label, termination_message)
        issues.append(
            Issue(
                "termination",
                termination_message,
                submission.report_label,
                submission.username,
                str(path),
                severity="warning",
            )
        )
    for direction, discarded_count in sorted(discarded_rows_by_direction.items()):
        message = (
            f"{path}: discarded {discarded_count} official prediction row(s) "
            f"for reference-unavailable direction {direction!r}"
        )
        logging.warning(
            "%s: %s",
            submission.report_label,
            message,
        )
        issues.append(
            Issue(
                "predictions",
                message,
                submission.report_label,
                submission.username,
                str(path),
                severity="warning",
            )
        )
    available_primary = PRIMARY_LANGUAGE_PAIRS & applicable_directions
    incomplete_primary = sorted(
        direction
        for direction in available_primary
        if expected_by_direction[direction] - provided.get(direction, set())
    )
    if incomplete_primary:
        logging.warning(
            "%s: predictions do not completely cover the primary translation "
            "directions. Missing: %s",
            submission.report_label,
            ", ".join(incomplete_primary),
        )

    if require_complete_predictions:
        for direction in sorted(observed_directions):
            missing = expected_by_direction[direction] - provided.get(direction, set())
            if not missing:
                continue
            missing_items = len({item_id for item_id, _ in missing})
            examples = ", ".join(
                f"{item_id!r}/{system!r}" for item_id, system in sorted(missing)[:3]
            )
            raise ValueError(
                f"{path}: incomplete prediction coverage for {direction!r}; "
                f"missing {len(missing)} test-set item-system pair(s) across "
                f"{missing_items} item(s); examples: {examples}."
            )

    missing_directions = sorted(applicable_directions - observed_directions)
    if missing_directions:
        logging.debug(
            "%s: official translation directions missing from predictions: %s",
            submission.report_label,
            ", ".join(missing_directions),
        )
    if not observed_directions and not allow_no_selected_predictions:
        raise ValueError(f"{path}: no official test-set prediction items were provided")

    coverage = build_coverage_report(
        test_set,
        provided,
        official_rows=official_rows,
        challenge_rows=challenge_rows,
        total_rows=total_rows,
        termination_by_direction=termination_coverage,
        reference_based=reference_based,
        directions=selected_directions,
    )
    return PreparedSubmission(
        tuple(assessments_for_metric),
        coverage,
        ignored_zero_lengths,
        tuple(issues),
    )


def _prepare_metric_inputs(
    submission: SubmissionDescriptor,
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    *,
    require_complete_predictions: bool,
    reference_based: bool = False,
    directions: Iterable[str] | None = None,
) -> tuple[
    dict[str, tuple[list[list[Span]], list[list[Span]]]],
    CoverageReport,
    int,
]:
    """Compatibility wrapper around the richer shared preparation."""

    prepared = _prepare_submission(
        submission,
        test_set,
        ground_truth,
        require_complete_predictions=require_complete_predictions,
        reference_based=reference_based,
        directions=directions,
    )
    return (
        _group_metric_inputs(prepared.assessments),
        prepared.coverage,
        prepared.ignored_zero_length_spans,
    )


def _precision_recall_fscore(
    components: MppComponents,
) -> tuple[float, float, float]:
    """Convert aggregated raw MPP components to precision, recall, and F1."""

    precision = (
        components.precision_numerator / components.precision_denominator
        if components.precision_denominator > 0
        else 1.0
    )
    recall = (
        components.recall_numerator / components.recall_denominator
        if components.recall_denominator > 0
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return precision, recall, f1


def _sum_mpp_components(
    assessments: Iterable[ScoredAssessment],
) -> MppComponents:
    """Weight and sum reusable raw components over an arbitrary record subset."""

    precision_numerator = 0.0
    precision_denominator = 0.0
    recall_numerator = 0.0
    recall_denominator = 0.0
    for scored in assessments:
        weight = scored.segment_weight
        precision_numerator += scored.components.precision_numerator * weight
        precision_denominator += scored.components.precision_denominator * weight
        recall_numerator += scored.components.recall_numerator * weight
        recall_denominator += scored.components.recall_denominator * weight
    return MppComponents(
        precision_numerator,
        precision_denominator,
        recall_numerator,
        recall_denominator,
    )


def aggregate_mpp_components(
    assessments: Iterable[ScoredAssessment],
) -> tuple[float, float, float]:
    """Return micro-aggregated MPP precision, recall, and F1 for scored records.

    The input records already carry the reciprocal reference weights used by
    official scoring. The returned tuple therefore follows the exact Task 1
    aggregation policy for any caller-defined subset of aligned assessments.
    """

    return _precision_recall_fscore(_sum_mpp_components(assessments))


def scored_assessments_from_result(
    assessments: Iterable[PreparedAssessment],
    segment_weights: Iterable[float],
    result: MetricResult,
) -> tuple[ScoredAssessment, ...]:
    """Attach unweighted metric components and post-compute weights to assessments.

    Inputs are aligned prepared assessments, their aggregation weights, and an
    unweighted multi-segment MPP result. The output preserves that alignment and
    can be cached or passed to :func:`aggregate_mpp_components`.
    """

    records = tuple(assessments)
    weights = tuple(segment_weights)
    if len(weights) != len(records):
        raise ValueError("segment weights must align with prepared assessments")
    segment_components = getattr(
        getattr(result, "details", None), "segments_components", None
    )
    if segment_components is None or len(segment_components) != len(records):
        raise RuntimeError(
            "The MPP dependency did not return one raw component record "
            "per aligned assessment"
        )
    return tuple(
        ScoredAssessment(
            assessment,
            weight,
            MppComponents(
                float(components.precision_numerator),
                float(components.precision_denominator),
                float(components.recall_numerator),
                float(components.recall_denominator),
            ),
        )
        for assessment, weight, components in zip(
            records, weights, segment_components
        )
    )


def _compute_scored_assessments(
    submission: SubmissionDescriptor,
    assessments: tuple[PreparedAssessment, ...],
    *,
    matching_algorithm: str,
    segment_weights: tuple[float, ...] | None = None,
    timing_by_direction: dict[str, float] | None = None,
    computed_directions: set[str] | None = None,
) -> tuple[ScoredAssessment, ...]:
    """Compute raw MPP components once and align them with prepared records."""

    weights = (
        list(segment_weights)
        if segment_weights is not None
        else _segment_weights(assessments)
    )
    if len(weights) != len(assessments):
        raise ValueError("segment weights must align with prepared assessments")
    grouped: dict[str, list[tuple[int, PreparedAssessment, float]]] = {}
    for index, (assessment, weight) in enumerate(zip(assessments, weights)):
        grouped.setdefault(assessment.language_pair, []).append(
            (index, assessment, weight)
        )

    output: list[ScoredAssessment | None] = [None] * len(assessments)
    for language_pair in sorted(grouped):
        indexed = grouped[language_pair]
        records = [value[1] for value in indexed]
        logging.debug(
            "%s: scoring %s with %d aligned assessments",
            submission.report_label,
            language_pair,
            len(records),
        )
        started = perf_counter()
        result = compute(
            [record.prediction for record in records],
            [record.reference for record in records],
            **mpp_metric_options(matching_algorithm),
        )
        if timing_by_direction is not None:
            timing_by_direction[language_pair] = perf_counter() - started
        if computed_directions is not None:
            computed_directions.add(language_pair)
        scored_records = scored_assessments_from_result(
            records,
            (value[2] for value in indexed),
            result,
        )
        for (index, _, _), scored in zip(indexed, scored_records):
            output[index] = scored
    if any(value is None for value in output):
        raise RuntimeError("MPP component alignment failed")
    return tuple(value for value in output if value is not None)


def _cached_scored_assessments(
    submission: SubmissionDescriptor,
    assessments: tuple[PreparedAssessment, ...],
    segment_weights: tuple[float, ...],
    cache: MppCache,
    *,
    matching_algorithm: str,
) -> tuple[ScoredAssessment | None, ...] | None:
    """Rebuild every available scored record from a valid cache entry."""

    values = cache.load(submission, assessments, matching_algorithm=matching_algorithm)
    if values is None:
        return None
    output: list[ScoredAssessment | None] = []
    for assessment, expected_weight, value in zip(assessments, segment_weights, values):
        if value is None or value[0] != expected_weight:
            output.append(None)
            continue
        output.append(
            ScoredAssessment(
                assessment,
                expected_weight,
                MppComponents(*value[1:]),
            )
        )
    return tuple(output)


def score_prepared_assessments(
    submission: SubmissionDescriptor,
    assessments: Iterable[PreparedAssessment],
    *,
    mpp_cache: MppCache | None = None,
    recompute_mpp: bool = False,
    matching_algorithm: str = "greedy",
    timing_by_direction: dict[str, float] | None = None,
    computed_directions: set[str] | None = None,
) -> tuple[ScoredAssessment, ...]:
    """Score caller-prepared aligned assessments with official MPP semantics.

    Inputs must already be normalized and aligned to their gold spans. The
    output preserves input order and contains reusable raw MPP components.
    """

    if matching_algorithm not in MATCHING_ALGORITHMS:
        raise ValueError(f"unsupported MPP matching algorithm {matching_algorithm!r}")
    records = tuple(assessments)
    if not records:
        raise ValueError("no prepared assessments were provided for MPP scoring")
    segment_weights = tuple(_segment_weights(records))
    cached_assessments = (
        None
        if mpp_cache is None or recompute_mpp
        else _cached_scored_assessments(
            submission,
            records,
            segment_weights,
            mpp_cache,
            matching_algorithm=matching_algorithm,
        )
    )
    output: list[ScoredAssessment | None] = (
        [None] * len(records)
        if cached_assessments is None
        else list(cached_assessments)
    )
    missing_indices = [index for index, value in enumerate(output) if value is None]
    if missing_indices:
        computed = _compute_scored_assessments(
            submission,
            tuple(records[index] for index in missing_indices),
            matching_algorithm=matching_algorithm,
            segment_weights=tuple(segment_weights[index] for index in missing_indices),
            timing_by_direction=timing_by_direction,
            computed_directions=computed_directions,
        )
        for index, value in zip(missing_indices, computed):
            output[index] = value
        if mpp_cache is not None:
            mpp_cache.store(
                submission,
                computed,
                matching_algorithm=matching_algorithm,
            )
    if any(value is None for value in output):
        raise RuntimeError("MPP cache/computation alignment failed")
    return tuple(value for value in output if value is not None)


def permutation_inputs_from_assessments(
    assessments: Iterable[ScoredAssessment],
) -> dict[str, PermutationDirection]:
    """Collapse weighted human assessments into aligned item-system components."""
    grouped: dict[str, dict[tuple[str, str], list[ScoredAssessment]]] = {}
    for scored in assessments:
        record = scored.assessment
        grouped.setdefault(record.language_pair, {}).setdefault(
            (record.item_id, record.system), []
        ).append(scored)
    output = {}
    for direction, pairs in sorted(grouped.items()):
        support, components = [], []
        for (item, system), records in sorted(pairs.items()):
            records.sort(key=lambda value: value.assessment.reference_index)
            support.append((
                item,
                system,
                tuple((value.assessment.reference_index, value.segment_weight) for value in records),
            ))
            total = _sum_mpp_components(records)
            components.append((
                total.precision_numerator,
                total.precision_denominator,
                total.recall_numerator,
                total.recall_denominator,
            ))
        output[direction] = PermutationDirection(
            tuple(support), np.asarray(components, dtype=np.float64)
        )
    return output


def _direction_scores(
    scored_assessments: Iterable[ScoredAssessment],
) -> tuple[DirectionScore, ...]:
    """Aggregate scored assessments into official per-direction micro scores."""

    grouped: dict[str, list[ScoredAssessment]] = {}
    for scored in scored_assessments:
        grouped.setdefault(scored.assessment.language_pair, []).append(scored)
    direction_scores: list[DirectionScore] = []
    for language_pair in sorted(grouped):
        records = grouped[language_pair]
        precision, recall, f1 = aggregate_mpp_components(records)
        direction_scores.append(
            DirectionScore(language_pair, precision, recall, f1, len(records))
        )
    return tuple(direction_scores)


def _score_prepared_submission_with_components(
    submission: SubmissionDescriptor,
    metadata: ModelCardMetadata,
    prepared: PreparedSubmission,
    *,
    mpp_cache: MppCache | None = None,
    recompute_mpp: bool = False,
    matching_algorithm: str = "greedy",
    timing_by_direction: dict[str, float] | None = None,
    computed_directions: set[str] | None = None,
) -> tuple[SubmissionScore, tuple[ScoredAssessment, ...]]:
    """Return one official score together with its reusable segment records."""

    if not prepared.assessments:
        raise ValueError(
            f"{submission.predictions_path}: no direction had both supported "
            "predictions and available ground-truth annotations"
        )
    scored_assessments = score_prepared_assessments(
        submission,
        prepared.assessments,
        mpp_cache=mpp_cache,
        recompute_mpp=recompute_mpp,
        matching_algorithm=matching_algorithm,
        timing_by_direction=timing_by_direction,
        computed_directions=computed_directions,
    )
    if prepared.ignored_zero_length_spans:
        logging.warning(
            "%s: ignored %d zero-length prediction spans",
            submission.report_label,
            prepared.ignored_zero_length_spans,
        )
    return (
        SubmissionScore(
            submission,
            metadata,
            prepared.coverage,
            _direction_scores(scored_assessments),
            prepared.ignored_zero_length_spans,
            prepared.issues,
        ),
        scored_assessments,
    )


def _score_prepared_submission(
    submission: SubmissionDescriptor,
    metadata: ModelCardMetadata,
    prepared: PreparedSubmission,
    *,
    mpp_cache: MppCache | None = None,
    recompute_mpp: bool = False,
    matching_algorithm: str = "greedy",
) -> SubmissionScore:
    """Compatibility wrapper returning only the official submission score."""

    score, _ = _score_prepared_submission_with_components(
        submission,
        metadata,
        prepared,
        mpp_cache=mpp_cache,
        recompute_mpp=recompute_mpp,
        matching_algorithm=matching_algorithm,
    )
    return score


def _uses_references(metadata: ModelCardMetadata, path: object) -> bool:
    """Return the required reference-use flag from validated card metadata."""

    reference_use = metadata.evaluator_characteristics.get("reference_use")
    if type(reference_use) is not bool:
        raise ValueError(
            f"{path}: the model card must provide a valid required reference-use answer"
        )
    return reference_use


def score_submission(
    submission: SubmissionDescriptor,
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    model_card_template: ModelCardTemplate | None,
    *,
    require_complete_predictions: bool = True,
    mpp_cache: MppCache | None = None,
    recompute_mpp: bool = False,
    matching_algorithm: str = "greedy",
    directions: Iterable[str] | None = None,
    collect_significance_inputs: bool = False,
) -> SubmissionScore:
    """Validate and score one entry, optionally retaining permutation inputs.

    The returned score carries compact item-system components only when
    collect_significance_inputs is enabled; no permutations are run here.
    """

    metadata = validate_model_card(submission.model_card_path, model_card_template)
    prepared = _prepare_submission(
        submission,
        test_set,
        ground_truth,
        require_complete_predictions=require_complete_predictions,
        reference_based=_uses_references(metadata, submission.model_card_path),
        directions=directions,
    )
    score, components = _score_prepared_submission_with_components(
        submission,
        metadata,
        prepared,
        mpp_cache=mpp_cache,
        recompute_mpp=recompute_mpp,
        matching_algorithm=matching_algorithm,
    )
    if collect_significance_inputs:
        return replace(score, permutation_inputs=permutation_inputs_from_assessments(components))
    return score


def score_submissions(
    submissions: Iterable[SubmissionDescriptor],
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    model_card_template: ModelCardTemplate | None,
    *,
    require_complete_predictions: bool = True,
    mpp_cache: MppCache | None = None,
    recompute_mpp: bool = False,
    matching_algorithm: str = "greedy",
    directions: Iterable[str] | None = None,
    collect_significance_inputs: bool = False,
) -> BatchScoreResult:
    """Return scores and failures for a batch, with authoritative direction scope.

    Optional significance inputs are collected while scoring each entry, so
    later pairwise tests need neither revalidation nor repeated span matching.
    """

    directions = _normalize_selected_directions(test_set, directions)
    scores: list[SubmissionScore] = []
    issues: list[Issue] = []
    for submission in submissions:
        logging.debug(
            "Processing %s (%s)",
            submission.report_label,
            submission.username,
        )
        try:
            score = score_submission(
                submission,
                test_set,
                ground_truth,
                model_card_template,
                require_complete_predictions=require_complete_predictions,
                mpp_cache=mpp_cache,
                recompute_mpp=recompute_mpp,
                matching_algorithm=matching_algorithm,
                directions=directions,
                collect_significance_inputs=collect_significance_inputs,
            )
            scores.append(score)
            issues.extend(score.issues)
        except (OSError, ValueError, RuntimeError) as exc:
            logging.error("%s failed: %s", submission.report_label, exc)
            issues.append(
                Issue(
                    "scoring",
                    str(exc),
                    submission.report_label,
                    submission.username,
                    str(submission.path),
                )
            )
    selected_set = (
        set(directions) if directions is not None else set(test_set.official_directions)
    )
    directions_without_references = tuple(
        sorted(selected_set - test_set.reference_available_directions)
    )
    return BatchScoreResult(
        tuple(scores),
        tuple(issues),
        dict(test_set.reference_systems_by_direction),
        directions_without_references,
        matching_algorithm,
        (
            tuple(directions)
            if directions is not None
            else tuple(sorted(test_set.official_directions))
        ),
        tuple(sorted(
            direction for direction in selected_set
            if ground_truth.pair_counts_by_direction.get(direction, 0)
        )),
    )


def compare_submissions(
    submissions: Iterable[SubmissionDescriptor],
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    model_card_template: ModelCardTemplate | None,
    *,
    require_complete_predictions: bool = True,
    mpp_caches: Mapping[str, MppCache] | None = None,
    recompute_mpp: bool = False,
    directions: Iterable[str] | None = None,
    collect_significance_inputs: bool = False,
) -> tuple[BatchScoreResult, BatchScoreResult, MatchingComparisonResult]:
    """Score aligned entries using separate greedy and optimal caches.

    The returned batch results are suitable for the ordinary scoring reports.
    The comparison result contains paired direction metrics and timing data.
    With collect_significance_inputs, each batch also retains its own matcher-
    specific component arrays for subsequent evaluator significance tests.
    """

    selected_directions = _normalize_selected_directions(test_set, directions)
    algorithm_scores: dict[str, list[SubmissionScore]] = {
        algorithm: [] for algorithm in MATCHING_ALGORITHMS
    }
    algorithm_issues: dict[str, list[Issue]] = {
        algorithm: [] for algorithm in MATCHING_ALGORITHMS
    }
    comparison_rows: list[MatchingComparisonRow] = []
    comparison_timings: list[MatchingComparisonTiming] = []
    comparison_issues: list[Issue] = []

    for submission in submissions:
        try:
            metadata = validate_model_card(
                submission.model_card_path, model_card_template
            )
            prepared = _prepare_submission(
                submission,
                test_set,
                ground_truth,
                require_complete_predictions=require_complete_predictions,
                reference_based=_uses_references(metadata, submission.model_card_path),
                directions=selected_directions,
            )
            if not prepared.assessments:
                raise ValueError(
                    f"{submission.predictions_path}: no direction had both supported "
                    "predictions and available ground-truth annotations"
                )
        except (OSError, ValueError, RuntimeError) as exc:
            for algorithm in MATCHING_ALGORITHMS:
                issue = Issue(
                    "scoring",
                    str(exc),
                    submission.report_label,
                    submission.username,
                    str(submission.path),
                )
                algorithm_issues[algorithm].append(issue)
            comparison_issues.append(algorithm_issues["greedy"][-1])
            continue

        scores_by_algorithm: dict[str, SubmissionScore] = {}
        direction_components: dict[str, dict[str, DirectionScore]] = {}
        timing_by_algorithm: dict[str, dict[str, float]] = {}
        for algorithm in MATCHING_ALGORITHMS:
            direction_timings: dict[str, float] = {}
            computed_directions: set[str] = set()
            started = perf_counter()
            try:
                scored_assessments = score_prepared_assessments(
                    submission,
                    prepared.assessments,
                    mpp_cache=(mpp_caches or {}).get(algorithm),
                    recompute_mpp=recompute_mpp,
                    matching_algorithm=algorithm,
                    timing_by_direction=direction_timings,
                    computed_directions=computed_directions,
                )
                score = SubmissionScore(
                    submission,
                    metadata,
                    prepared.coverage,
                    _direction_scores(scored_assessments),
                    prepared.ignored_zero_length_spans,
                    prepared.issues,
                )
                if collect_significance_inputs:
                    score = replace(
                        score,
                        permutation_inputs=permutation_inputs_from_assessments(scored_assessments),
                    )
                algorithm_scores[algorithm].append(score)
                algorithm_issues[algorithm].extend(prepared.issues)
                scores_by_algorithm[algorithm] = score
                direction_components[algorithm] = {
                    row.language_pair: row for row in score.direction_scores
                }
                timing_by_algorithm[algorithm] = direction_timings
                comparison_timings.append(
                    MatchingComparisonTiming(
                        "baseline" if submission.is_baseline else "submission",
                        (
                            submission.baseline_name
                            if submission.is_baseline
                            else submission.submission_id
                        ),
                        submission.report_label,
                        algorithm,
                        (
                            sum(direction_timings.values())
                            if computed_directions
                            else None
                        ),
                        perf_counter() - started,
                        bool(computed_directions),
                    )
                )
            except (OSError, ValueError, RuntimeError) as exc:
                issue = Issue(
                    "scoring",
                    str(exc),
                    submission.report_label,
                    submission.username,
                    str(submission.path),
                )
                algorithm_issues[algorithm].append(issue)
                comparison_issues.append(issue)

        if len(scores_by_algorithm) != len(MATCHING_ALGORITHMS):
            continue

        common_directions = sorted(
            set(direction_components["greedy"]) & set(direction_components["optimal"])
        )
        for language_pair in common_directions:
            greedy = direction_components["greedy"][language_pair]
            optimal = direction_components["optimal"][language_pair]
            comparison_rows.append(
                MatchingComparisonRow(
                    "baseline" if submission.is_baseline else "submission",
                    (
                        submission.baseline_name
                        if submission.is_baseline
                        else submission.submission_id
                    ),
                    submission.report_label,
                    language_pair,
                    greedy.aligned_assessments,
                    greedy.precision,
                    greedy.recall,
                    greedy.f1,
                    optimal.precision,
                    optimal.recall,
                    optimal.f1,
                    optimal.precision - greedy.precision,
                    optimal.recall - greedy.recall,
                    optimal.f1 - greedy.f1,
                    timing_by_algorithm["greedy"].get(language_pair),
                    timing_by_algorithm["optimal"].get(language_pair),
                )
            )

    selected_tuple = (
        tuple(selected_directions)
        if selected_directions is not None
        else tuple(sorted(test_set.official_directions))
    )
    without_references = tuple(
        sorted(set(selected_tuple) - test_set.reference_available_directions)
    )
    batches = tuple(
        BatchScoreResult(
            tuple(algorithm_scores[algorithm]),
            tuple(algorithm_issues[algorithm]),
            dict(test_set.reference_systems_by_direction),
            without_references,
            algorithm,
            selected_tuple,
            tuple(sorted(
                direction for direction in selected_tuple
                if ground_truth.pair_counts_by_direction.get(direction, 0)
            )),
        )
        for algorithm in MATCHING_ALGORITHMS
    )
    comparison = MatchingComparisonResult(
        tuple(comparison_rows),
        tuple(comparison_timings),
        tuple(comparison_issues),
        selected_tuple,
    )
    return batches[0], batches[1], comparison
