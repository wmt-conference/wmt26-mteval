"""Matched reference-use and thinking effect analysis for Task 1."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Final

from wmt26_mteval_task1.constants import PRIMARY_LANGUAGE_PAIRS
from wmt26_mteval_task1.models import (
    BatchScoreResult,
    DirectionScore,
    Issue,
    SubmissionDescriptor,
    SubmissionScore,
    ModelCardTemplate,
    GroundTruthSet,
    TestSetIndex,
)
from wmt26_mteval_task1.mpp_cache import MppCache
from wmt26_mteval_task1.score_statistics import (
    SpanStatistics,
    statistics_for_records,
)
from wmt26_mteval_task1.scoring import _prepare_submission, score_submissions


PRIORITY_DIRECTIONS: Final[tuple[str, ...]] = tuple(sorted(PRIMARY_LANGUAGE_PAIRS))
METRICS: Final[tuple[str, ...]] = ("precision", "recall", "f1")
BEHAVIOR_METRICS: Final[tuple[str, ...]] = (
    "errors_per_prediction",
    "mean_span_length",
    "mean_severity_weight",
)
REFERENCE_EFFECT: Final[str] = "reference_effect"
THINKING_EFFECT: Final[str] = "thinking_effect"
BehaviorValues = tuple[float | None, float | None, float | None]


@dataclass(frozen=True)
class BaselinePairSpec:
    """Describe one stable matched baseline pair and its treatment direction."""

    effect: str
    pair_id: str
    family: str
    control_baseline: str
    treatment_baseline: str
    control_condition: str
    treatment_condition: str
    expected_control_reasoning: str
    expected_treatment_reasoning: str
    thinking_reference_based: bool = False


BASELINE_PAIR_SPECS: Final[tuple[BaselinePairSpec, ...]] = (
    BaselinePairSpec(
        REFERENCE_EFFECT,
        "gemma4_non_reasoning",
        "Gemma 4",
        "gemma4_no_thinking_task1",
        "gemma_ref_no_thinking_task1",
        "reference-free",
        "reference-based",
        "non_reasoning",
        "non_reasoning",
    ),
    BaselinePairSpec(
        REFERENCE_EFFECT,
        "qwen36_non_reasoning",
        "Qwen 3.6",
        "qwen36_no_thinking_task1",
        "qwen_ref_no_thinking_task1",
        "reference-free",
        "reference-based",
        "non_reasoning",
        "non_reasoning",
    ),
    BaselinePairSpec(
        THINKING_EFFECT,
        "gemma4_reference_based",
        "Gemma 4",
        "gemma_ref_no_thinking_task1",
        "gemma_ref_task1",
        "non-reasoning",
        "reasoning",
        "non_reasoning",
        "reasoning",
        thinking_reference_based=True,
    ),
    BaselinePairSpec(
        THINKING_EFFECT,
        "qwen36_reference_based",
        "Qwen 3.6",
        "qwen_ref_no_thinking_task1",
        "qwen_ref_task1",
        "non-reasoning",
        "reasoning",
        "non_reasoning",
        "reasoning",
        thinking_reference_based=True,
    ),
    BaselinePairSpec(
        REFERENCE_EFFECT,
        "gemma4_reasoning",
        "Gemma 4",
        "gemma_task1",
        "gemma_ref_task1",
        "reference-free",
        "reference-based",
        "reasoning",
        "reasoning",
    ),
    BaselinePairSpec(
        REFERENCE_EFFECT,
        "qwen36_reasoning",
        "Qwen 3.6",
        "qwen_task1",
        "qwen_ref_task1",
        "reference-free",
        "reference-based",
        "reasoning",
        "reasoning",
    ),
    BaselinePairSpec(
        REFERENCE_EFFECT,
        "xcomet_xl",
        "xCOMET XL",
        "xcomet_xl_no_ref_task1",
        "xcomet_xl_ref_task1",
        "reference-free",
        "reference-based",
        "non_reasoning",
        "non_reasoning",
    ),
    BaselinePairSpec(
        REFERENCE_EFFECT,
        "xcomet_xxl",
        "xCOMET XXL",
        "xcomet_xxl_no_ref_task1",
        "xcomet_xxl_ref_task1",
        "reference-free",
        "reference-based",
        "non_reasoning",
        "non_reasoning",
    ),
    BaselinePairSpec(
        THINKING_EFFECT,
        "gemma4_reference_free",
        "Gemma 4",
        "gemma4_no_thinking_task1",
        "gemma_task1",
        "non-reasoning",
        "reasoning",
        "non_reasoning",
        "reasoning",
    ),
    BaselinePairSpec(
        THINKING_EFFECT,
        "qwen36_reference_free",
        "Qwen 3.6",
        "qwen36_no_thinking_task1",
        "qwen_task1",
        "non-reasoning",
        "reasoning",
        "non_reasoning",
        "reasoning",
    ),
)
BASELINE_PAIR_NAMES: Final[frozenset[str]] = frozenset(
    baseline
    for spec in BASELINE_PAIR_SPECS
    for baseline in (spec.control_baseline, spec.treatment_baseline)
)


@dataclass(frozen=True)
class PairEffectRow:
    """Priority-aggregate score and behavior effects for one matched pair."""

    effect: str
    pair_id: str
    family: str
    control_baseline: str
    treatment_baseline: str
    control_entry: str
    treatment_entry: str
    control_condition: str
    treatment_condition: str
    control_reasoning: str
    treatment_reasoning: str
    control_values: tuple[float, float, float]
    treatment_values: tuple[float, float, float]
    delta_values: tuple[float, float, float]
    control_behavior: BehaviorValues
    treatment_behavior: BehaviorValues
    delta_behavior: BehaviorValues
    direction_count: int
    control_aligned_assessments: int
    treatment_aligned_assessments: int


@dataclass(frozen=True)
class DirectionEffectRow:
    """Score and behavior effects for one pair in one priority direction."""

    effect: str
    pair_id: str
    family: str
    language_pair: str
    control_baseline: str
    treatment_baseline: str
    control_entry: str
    treatment_entry: str
    control_values: tuple[float, float, float]
    treatment_values: tuple[float, float, float]
    delta_values: tuple[float, float, float]
    control_behavior: BehaviorValues
    treatment_behavior: BehaviorValues
    delta_behavior: BehaviorValues
    control_aligned_assessments: int
    treatment_aligned_assessments: int


@dataclass(frozen=True)
class OverallEffectRow:
    """Equal-pair aggregate score and behavior values for one effect."""

    effect: str
    pair_count: int
    control_values: tuple[float, float, float]
    treatment_values: tuple[float, float, float]
    delta_values: tuple[float, float, float]
    control_behavior: BehaviorValues
    treatment_behavior: BehaviorValues
    delta_behavior: BehaviorValues


@dataclass(frozen=True)
class BaselineManifestRow:
    """Reportable metadata and reasoning provenance for one matched baseline."""

    effect: str
    pair_id: str
    family: str
    role: str
    baseline: str
    entry: str
    condition: str
    reference_based: str
    reasoning: str
    reasoning_source: str
    status: str


@dataclass(frozen=True)
class MppEffectAnalysisResult:
    """Complete matched-effect data and issues for one matching algorithm."""

    pair_effects: tuple[PairEffectRow, ...]
    direction_effects: tuple[DirectionEffectRow, ...]
    overall_effects: tuple[OverallEffectRow, ...]
    manifest: tuple[BaselineManifestRow, ...]
    issues: tuple[Issue, ...]
    matching_algorithm: str
    directions: tuple[str, ...] = PRIORITY_DIRECTIONS

    @property
    def failed(self) -> bool:
        """Return whether any scoring or pairing error was recorded."""

        return any(issue.severity == "error" for issue in self.issues)


def infer_reasoning_label(
    description: str | None, baseline_name: str
) -> tuple[str, str]:
    """Infer reasoning status and provenance from a baseline model-card description."""

    text = (description or "").casefold()
    if re.search(
        r"\bwithout reasoning\b|\bno reasoning\b|\breasoning disabled\b", text
    ):
        return "non_reasoning", "model-card description"
    if re.search(r"\bwith reasoning\b|\breasoning enabled\b", text):
        return "reasoning", "model-card description"
    if baseline_name.startswith("xcomet_"):
        return "non_reasoning", "inferred dedicated metric model"
    raise ValueError(
        f"{baseline_name}: model-card description does not identify reasoning status"
    )


def _entry_key(descriptor: SubmissionDescriptor) -> str:
    """Return the stable baseline directory key for a scored entry."""

    return getattr(descriptor, "source_baseline_name", descriptor.path.name)


def _metric_values(score: SubmissionScore) -> tuple[float, float, float]:
    """Return priority-macro precision, recall, and F1 from a scored entry."""

    rows = [
        row
        for row in score.direction_scores
        if row.language_pair in PRIMARY_LANGUAGE_PAIRS
    ]
    if {row.language_pair for row in rows} != set(PRIORITY_DIRECTIONS):
        raise ValueError("a matched entry is missing one or more priority directions")
    return (
        fmean(row.precision for row in rows),
        fmean(row.recall for row in rows),
        fmean(row.f1 for row in rows),
    )


def _direction_values(row: DirectionScore) -> tuple[float, float, float]:
    """Return precision, recall, and F1 from one direction score."""

    return row.precision, row.recall, row.f1


def _behavior_values(statistics: SpanStatistics) -> BehaviorValues:
    """Return count, length, and severity summaries for unique predictions."""

    return (
        statistics.mean_total_errors_per_assessment,
        statistics.mean_text_span_length_characters,
        statistics.mean_severity_weight,
    )


def _mean_optional(values: Iterable[float | None]) -> float | None:
    """Return the arithmetic mean of available values, or none when empty."""

    available = [value for value in values if value is not None]
    return fmean(available) if available else None


def _mean_behavior(values: Iterable[BehaviorValues]) -> BehaviorValues:
    """Macro-average behavior values independently across rows."""

    rows = tuple(values)
    return (
        _mean_optional(row[0] for row in rows),
        _mean_optional(row[1] for row in rows),
        _mean_optional(row[2] for row in rows),
    )


def _subtract_behavior(
    treatment: BehaviorValues, control: BehaviorValues
) -> BehaviorValues:
    """Return treatment-minus-control behavior deltas when both values exist."""

    def subtract(index: int) -> float | None:
        """Subtract one optional behavior component."""

        if treatment[index] is None or control[index] is None:
            return None
        return treatment[index] - control[index]

    return subtract(0), subtract(1), subtract(2)


def collect_prediction_statistics(
    scores: Iterable[SubmissionScore],
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
) -> dict[str, dict[str, SpanStatistics]]:
    """Summarize each scored baseline's unique predictions by priority direction."""

    output: dict[str, dict[str, SpanStatistics]] = {}
    for score in scores:
        reference_based = (
            score.model_card.evaluator_characteristics.get("reference_use") is True
        )
        prepared = _prepare_submission(
            score.descriptor,
            test_set,
            ground_truth,
            require_complete_predictions=True,
            reference_based=reference_based,
            directions=PRIORITY_DIRECTIONS,
        )
        grouped = {
            direction: [
                record
                for record in prepared.assessments
                if record.language_pair == direction
            ]
            for direction in PRIORITY_DIRECTIONS
        }
        output[_entry_key(score.descriptor)] = {
            direction: statistics_for_records(records)[0]
            for direction, records in grouped.items()
        }
    return output


def _subtract_values(
    treatment: Sequence[float], control: Sequence[float]
) -> tuple[float, float, float]:
    """Return elementwise treatment-minus-control metric deltas."""

    return (
        treatment[0] - control[0],
        treatment[1] - control[1],
        treatment[2] - control[2],
    )


def _aggregate_behavior(
    values_by_direction: Mapping[str, SpanStatistics],
) -> BehaviorValues:
    """Macro-average prediction behavior over all priority directions."""

    missing = set(PRIORITY_DIRECTIONS) - set(values_by_direction)
    if missing:
        raise ValueError(
            "prediction behavior is missing priority directions: "
            + ", ".join(sorted(missing))
        )
    return _mean_behavior(
        _behavior_values(values_by_direction[direction])
        for direction in PRIORITY_DIRECTIONS
    )


def _validate_pair_card(
    spec: BaselinePairSpec,
    score: SubmissionScore,
    reasoning: str,
    role: str,
) -> str | None:
    """Return a model-card contract error for one pair endpoint, if present."""

    expected_reference_based = (
        spec.thinking_reference_based
        if spec.effect == THINKING_EFFECT
        else role == "treatment"
    )
    actual_reference_based = score.model_card.evaluator_characteristics.get(
        "reference_use"
    )
    if actual_reference_based is not expected_reference_based:
        return f"{score.descriptor.report_label}: expected reference_use={expected_reference_based}"
    expected_reasoning = (
        spec.expected_treatment_reasoning
        if role == "treatment"
        else spec.expected_control_reasoning
    )
    if reasoning != expected_reasoning:
        return (
            f"{score.descriptor.report_label}: expected reasoning={expected_reasoning}, "
            f"inferred {reasoning}"
        )
    return None


def _issue(message: str, *, severity: str = "error") -> Issue:
    """Build one analysis-level issue with a consistent stage label."""

    return Issue("reference_thinking_effects", message, severity=severity)


def _overall_effects(
    pair_effects: Iterable[PairEffectRow],
) -> tuple[OverallEffectRow, ...]:
    """Average complete pair values and deltas equally within each effect."""

    grouped: dict[str, list[PairEffectRow]] = {}
    for row in pair_effects:
        grouped.setdefault(row.effect, []).append(row)
    output: list[OverallEffectRow] = []
    for effect in sorted(grouped):
        rows = grouped[effect]
        control = (
            fmean(row.control_values[0] for row in rows),
            fmean(row.control_values[1] for row in rows),
            fmean(row.control_values[2] for row in rows),
        )
        treatment = (
            fmean(row.treatment_values[0] for row in rows),
            fmean(row.treatment_values[1] for row in rows),
            fmean(row.treatment_values[2] for row in rows),
        )
        delta = (
            fmean(row.delta_values[0] for row in rows),
            fmean(row.delta_values[1] for row in rows),
            fmean(row.delta_values[2] for row in rows),
        )
        output.append(
            OverallEffectRow(
                effect,
                len(rows),
                control,
                treatment,
                delta,
                _mean_behavior(row.control_behavior for row in rows),
                _mean_behavior(row.treatment_behavior for row in rows),
                _mean_behavior(row.delta_behavior for row in rows),
            )
        )
    return tuple(output)


def analyze_mpp_effects(
    baselines: Iterable[SubmissionDescriptor],
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    model_card_template: ModelCardTemplate | None,
    *,
    mpp_cache: MppCache | None = None,
    recompute_mpp: bool = False,
    matching_algorithm: str = "greedy",
) -> MppEffectAnalysisResult:
    """Score matched baselines and compute priority-direction MPP effects."""

    batch: BatchScoreResult = score_submissions(
        tuple(baselines),
        test_set,
        ground_truth,
        model_card_template,
        require_complete_predictions=True,
        mpp_cache=mpp_cache,
        recompute_mpp=recompute_mpp,
        matching_algorithm=matching_algorithm,
        directions=PRIORITY_DIRECTIONS,
    )
    issues = list(batch.issues)
    scores = {_entry_key(score.descriptor): score for score in batch.scores}
    prediction_statistics = collect_prediction_statistics(
        batch.scores, test_set, ground_truth
    )
    reasoning: dict[str, tuple[str, str]] = {}
    manifest: list[BaselineManifestRow] = []
    invalid_pairs: set[str] = set()
    for spec in BASELINE_PAIR_SPECS:
        for role, baseline in (
            ("control", spec.control_baseline),
            ("treatment", spec.treatment_baseline),
        ):
            score = scores.get(baseline)
            if score is None:
                manifest.append(
                    BaselineManifestRow(
                        spec.effect,
                        spec.pair_id,
                        spec.family,
                        role,
                        baseline,
                        "",
                        (
                            spec.control_condition
                            if role == "control"
                            else spec.treatment_condition
                        ),
                        "",
                        "unknown",
                        "",
                        "missing_score",
                    )
                )
                continue
            try:
                label, source = infer_reasoning_label(
                    score.model_card.submission_description, baseline
                )
            except ValueError as exc:
                issues.append(_issue(str(exc)))
                label, source = "unknown", "error"
            reasoning[baseline] = (label, source)
            contract_error = _validate_pair_card(spec, score, label, role)
            if contract_error is not None:
                issues.append(_issue(contract_error))
                invalid_pairs.add(spec.pair_id)
            reference_based = score.model_card.evaluator_characteristics.get(
                "reference_use"
            )
            manifest.append(
                BaselineManifestRow(
                    spec.effect,
                    spec.pair_id,
                    spec.family,
                    role,
                    baseline,
                    score.model_card.submission_name or score.descriptor.report_label,
                    (
                        spec.control_condition
                        if role == "control"
                        else spec.treatment_condition
                    ),
                    (
                        "true"
                        if reference_based is True
                        else "false"
                        if reference_based is False
                        else "unknown"
                    ),
                    label,
                    source,
                    "invalid_model_card" if contract_error is not None else "scored",
                )
            )

    pair_rows: list[PairEffectRow] = []
    direction_rows: list[DirectionEffectRow] = []
    for spec in BASELINE_PAIR_SPECS:
        if spec.pair_id in invalid_pairs:
            continue
        control = scores.get(spec.control_baseline)
        treatment = scores.get(spec.treatment_baseline)
        if control is None or treatment is None:
            issues.append(
                _issue(
                    f"{spec.pair_id}: both matched baselines must score successfully"
                )
            )
            continue
        if (
            not control.coverage.all_primary_complete
            or not treatment.coverage.all_primary_complete
        ):
            issues.append(
                _issue(
                    f"{spec.pair_id}: both endpoints must pass coverage on all eight priority directions",
                    severity="warning",
                )
            )
            continue
        control_directions = {row.language_pair for row in control.direction_scores}
        treatment_directions = {row.language_pair for row in treatment.direction_scores}
        if not set(PRIORITY_DIRECTIONS).issubset(control_directions) or not set(
            PRIORITY_DIRECTIONS
        ).issubset(treatment_directions):
            issues.append(
                _issue(
                    f"{spec.pair_id}: both endpoints must contain all eight priority directions"
                )
            )
            continue
        control_values = _metric_values(control)
        treatment_values = _metric_values(treatment)
        control_behavior = _aggregate_behavior(
            prediction_statistics[spec.control_baseline]
        )
        treatment_behavior = _aggregate_behavior(
            prediction_statistics[spec.treatment_baseline]
        )
        pair_rows.append(
            PairEffectRow(
                spec.effect,
                spec.pair_id,
                spec.family,
                spec.control_baseline,
                spec.treatment_baseline,
                control.descriptor.report_label,
                treatment.descriptor.report_label,
                spec.control_condition,
                spec.treatment_condition,
                reasoning.get(spec.control_baseline, ("unknown", ""))[0],
                reasoning.get(spec.treatment_baseline, ("unknown", ""))[0],
                control_values,
                treatment_values,
                _subtract_values(treatment_values, control_values),
                control_behavior,
                treatment_behavior,
                _subtract_behavior(treatment_behavior, control_behavior),
                len(PRIORITY_DIRECTIONS),
                sum(
                    row.aligned_assessments
                    for row in control.direction_scores
                    if row.language_pair in PRIORITY_DIRECTIONS
                ),
                sum(
                    row.aligned_assessments
                    for row in treatment.direction_scores
                    if row.language_pair in PRIORITY_DIRECTIONS
                ),
            )
        )
        control_by_direction = {
            row.language_pair: row for row in control.direction_scores
        }
        treatment_by_direction = {
            row.language_pair: row for row in treatment.direction_scores
        }
        for language_pair in PRIORITY_DIRECTIONS:
            control_direction = control_by_direction[language_pair]
            treatment_direction = treatment_by_direction[language_pair]
            control_direction_values = _direction_values(control_direction)
            treatment_direction_values = _direction_values(treatment_direction)
            control_direction_behavior = _behavior_values(
                prediction_statistics[spec.control_baseline][language_pair]
            )
            treatment_direction_behavior = _behavior_values(
                prediction_statistics[spec.treatment_baseline][language_pair]
            )
            direction_rows.append(
                DirectionEffectRow(
                    spec.effect,
                    spec.pair_id,
                    spec.family,
                    language_pair,
                    spec.control_baseline,
                    spec.treatment_baseline,
                    control.descriptor.report_label,
                    treatment.descriptor.report_label,
                    control_direction_values,
                    treatment_direction_values,
                    _subtract_values(
                        treatment_direction_values, control_direction_values
                    ),
                    control_direction_behavior,
                    treatment_direction_behavior,
                    _subtract_behavior(
                        treatment_direction_behavior, control_direction_behavior
                    ),
                    control_direction.aligned_assessments,
                    treatment_direction.aligned_assessments,
                )
            )

    return MppEffectAnalysisResult(
        tuple(pair_rows),
        tuple(direction_rows),
        _overall_effects(pair_rows),
        tuple(manifest),
        tuple(issues),
        matching_algorithm,
    )
