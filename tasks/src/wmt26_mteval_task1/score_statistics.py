"""Focused gold-aligned MPP and annotation-behavior analysis."""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, quantiles
from typing import Any

from wmt26_mteval_task1.constants import (
    EVALUATOR_CHARACTERISTIC_QUESTIONS,
    OFFICIAL_SET_ID,
    PRIMARY_LANGUAGE_PAIRS,
    SEVERITY_WEIGHTS,
)
from wmt26_mteval_task1.io import parse_item_id, read_jsonl, validate_model_card
from wmt26_mteval_task1.mpp_cache import MppCache
from wmt26_mteval_task1.models import (
    Issue,
    ModelCardMetadata,
    ModelCardTemplate,
    GroundTruthSet,
    Span,
    SubmissionDescriptor,
    SubmissionScore,
    TestSetIndex,
)
from wmt26_mteval_task1.reporting_policy import reported_fine_tuned
from wmt26_mteval_task1.scoring import (
    PreparedAssessment,
    ScoredAssessment,
    _precision_recall_fscore,
    _prepare_submission,
    _score_prepared_submission_with_components,
    _sum_mpp_components,
    _uses_references,
)


COHORTS = ("priority", "all_official")
LENGTH_BASES = ("source_tokens", "target_tokens", "source_target_tokens")
LENGTH_BUCKET_SCOPES = ("within_direction", "global_priority")
LENGTH_BUCKETS = ("Q1", "Q2", "Q3", "Q4")
REASONING_VALUES = frozenset({"reasoning", "non_reasoning", "unknown"})
CORE_SPAN_METRICS = (
    "mean_text_span_length_characters",
    "mean_total_errors_per_assessment",
    "mean_severity_weight",
)
XLM_ROBERTA_TOKENIZER = "FacebookAI/xlm-roberta-base"
XLM_ROBERTA_REVISION = "a8c9047dc3eab38d118ef000fbbcd91042f0e9f6"
GLOBAL_PRIORITY_LABEL = "all_priority"


@dataclass(frozen=True)
class SpanStatistics:
    """Counts from which focused annotation-behavior statistics are derived."""

    pair_count: int
    assessment_count: int
    text_span_count: int
    text_span_character_count: int
    omission_count: int
    minor_error_count: int
    major_error_count: int
    minor_omission_count: int
    major_omission_count: int

    @property
    def total_error_count(self) -> int:
        """Return text spans plus omission annotations."""
        return self.text_span_count + self.omission_count

    @property
    def mean_text_span_length_characters(self) -> float | None:
        """Return mean non-omission span length in Unicode code points."""
        if not self.text_span_count:
            return None
        return self.text_span_character_count / self.text_span_count

    @property
    def mean_text_spans_per_assessment(self) -> float | None:
        """Return mean non-omission spans per assessment."""
        if not self.assessment_count:
            return None
        return self.text_span_count / self.assessment_count

    @property
    def mean_total_errors_per_assessment(self) -> float | None:
        """Return mean text spans plus omissions per assessment."""
        if not self.assessment_count:
            return None
        return self.total_error_count / self.assessment_count

    @property
    def mean_severity_weight(self) -> float | None:
        """Return mean MPP severity weight across spans and omissions."""

        if not self.total_error_count:
            return None
        weighted = (
            SEVERITY_WEIGHTS["minor"] * self.minor_error_count
            + SEVERITY_WEIGHTS["major"] * self.major_error_count
        )
        return weighted / self.total_error_count

    @property
    def omission_rate(self) -> float | None:
        """Return omission annotations per assessment."""
        if not self.assessment_count:
            return None
        return self.omission_count / self.assessment_count

    @property
    def minor_omission_rate(self) -> float | None:
        """Return minor omissions per assessment."""
        if not self.assessment_count:
            return None
        return self.minor_omission_count / self.assessment_count

    @property
    def major_omission_rate(self) -> float | None:
        """Return major omissions per assessment."""
        if not self.assessment_count:
            return None
        return self.major_omission_count / self.assessment_count


@dataclass(frozen=True)
class EvaluatorCharacteristics:
    """Explicit reasoning labels keyed by entry type and stable entry ID."""

    path: Path
    reasoning_by_entry: Mapping[tuple[str, str], str]


@dataclass(frozen=True)
class TokenLengthIndex:
    """Pinned-tokenizer lengths for human-annotated sources and hypotheses."""

    tokenizer_name: str
    tokenizer_revision: str
    source_lengths: Mapping[str, int]
    target_lengths: Mapping[tuple[str, str], int]


@dataclass(frozen=True)
class LengthBucketDefinition:
    """Quartile boundaries for one token-length basis and scope."""

    bucket_scope: str
    length_basis: str
    language_pair: str
    first_quartile: float
    median: float
    third_quartile: float
    minimum: int
    maximum: int
    unit_count: int


@dataclass(frozen=True)
class EntryStatisticsRow:
    """Focused MPP and span statistics for one evaluator cohort."""

    descriptor: SubmissionDescriptor
    model_card: ModelCardMetadata
    cohort: str
    directions: tuple[str, ...]
    f1: float
    aligned_assessments: int
    prediction_statistics: SpanStatistics
    gold_statistics: SpanStatistics
    characteristics: Mapping[str, str]


@dataclass(frozen=True)
class GoldStatisticsRow:
    """Focused human-gold statistics for one fixed cohort."""

    cohort: str
    directions: tuple[str, ...]
    statistics: SpanStatistics


@dataclass(frozen=True)
class DirectionStatisticsRow:
    """One evaluator-direction observation used by Spearman analysis."""

    descriptor: SubmissionDescriptor
    language_pair: str
    f1: float
    prediction_statistics: SpanStatistics


@dataclass(frozen=True)
class LengthStatisticsRow:
    """Direction-macro MPP and prediction behavior for one priority bucket."""

    descriptor: SubmissionDescriptor
    model_card: ModelCardMetadata
    bucket_scope: str
    length_basis: str
    length_bucket: str
    direction_count: int
    f1: float | None
    aligned_assessments: int
    mean_total_errors_per_assessment: float | None = None
    mean_text_span_length_characters: float | None = None
    mean_severity_weight: float | None = None


@dataclass(frozen=True)
class LengthSummaryRow:
    """Median and interquartile evaluator F1 for one length bucket."""

    bucket_scope: str
    length_basis: str
    length_bucket: str
    observation_count: int
    median_f1: float | None
    first_quartile_f1: float | None
    third_quartile_f1: float | None


@dataclass(frozen=True)
class CorrelationRow:
    """One per-direction across-evaluator Spearman result."""

    language_pair: str
    predictor: str
    observation_count: int
    spearman_correlation: float | None
    status: str
    undefined_evaluators: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorrelationSummaryRow:
    """Direction-equal distribution summary of Spearman coefficients."""

    predictor: str
    defined_direction_count: int
    total_direction_count: int
    median_correlation: float | None
    first_quartile_correlation: float | None
    third_quartile_correlation: float | None


@dataclass(frozen=True)
class GroundTruthRedundancyStatistics:
    """Global counts describing repeated human annotations."""

    total_pair_count: int
    total_assessment_count: int
    multiple_annotation_pair_count: int
    multiple_distinct_annotator_pair_count: int
    same_annotator_repeated_pair_count: int
    maximum_assessments_per_pair: int

    @property
    def extra_assessment_count(self) -> int:
        """Return human assessments beyond one per item/system pair."""
        return self.total_assessment_count - self.total_pair_count

    @property
    def multiple_annotation_pair_percentage(self) -> float:
        """Return the percentage of pairs with multiple assessments."""
        return 100.0 * self.multiple_annotation_pair_count / self.total_pair_count

    @property
    def multiple_distinct_annotator_pair_percentage(self) -> float:
        """Return the percentage of pairs covered by multiple annotators."""
        return (
            100.0 * self.multiple_distinct_annotator_pair_count / self.total_pair_count
        )

    @property
    def same_annotator_repeated_pair_percentage(self) -> float:
        """Return the percentage of pairs repeating an annotator ID."""
        return 100.0 * self.same_annotator_repeated_pair_count / self.total_pair_count


@dataclass(frozen=True)
class BatchStatisticsResult:
    """All focused score-statistics rows and reportable issues."""

    scores: tuple[SubmissionScore, ...]
    entry_statistics: tuple[EntryStatisticsRow, ...]
    gold_statistics: tuple[GoldStatisticsRow, ...]
    direction_statistics: tuple[DirectionStatisticsRow, ...]
    length_statistics: tuple[LengthStatisticsRow, ...]
    length_summaries: tuple[LengthSummaryRow, ...]
    length_buckets: tuple[LengthBucketDefinition, ...]
    correlations: tuple[CorrelationRow, ...]
    correlation_summaries: tuple[CorrelationSummaryRow, ...]
    ground_truth_redundancy: GroundTruthRedundancyStatistics
    token_lengths: TokenLengthIndex | None
    issues: tuple[Issue, ...]
    reference_systems_by_direction: Mapping[str, str]
    directions_without_references: tuple[str, ...]
    matching_algorithm: str = "greedy"

    @property
    def failed(self) -> bool:
        """Return whether any entry-level error was recorded."""
        return any(issue.severity == "error" for issue in self.issues)


def load_evaluator_characteristics(path: Path) -> EvaluatorCharacteristics:
    """Load a strict entry_type,entry_id,reasoning CSV mapping."""

    required_fields = ["entry_type", "entry_id", "reasoning"]
    try:
        stream = path.open(encoding="utf-8", newline="")
    except FileNotFoundError as exc:
        raise ValueError(f"{path}: evaluator-characteristics CSV is missing") from exc
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != required_fields:
            raise ValueError(
                f"{path}: evaluator-characteristics CSV header must be exactly "
                + ",".join(required_fields)
            )
        labels: dict[tuple[str, str], str] = {}
        for line_number, row in enumerate(reader, 2):
            entry_type = row["entry_type"].strip()
            entry_id = row["entry_id"].strip()
            reasoning = row["reasoning"].strip().casefold()
            if entry_type not in {"submission", "baseline"}:
                raise ValueError(
                    f"{path}:{line_number}: entry_type must be submission or baseline"
                )
            if not entry_id:
                raise ValueError(f"{path}:{line_number}: entry_id must not be blank")
            if reasoning not in REASONING_VALUES:
                raise ValueError(
                    f"{path}:{line_number}: reasoning must be reasoning, "
                    "non_reasoning, or unknown"
                )
            key = (entry_type, entry_id)
            if key in labels:
                raise ValueError(
                    f"{path}:{line_number}: duplicate evaluator-characteristics "
                    f"entry {entry_type}/{entry_id}"
                )
            labels[key] = reasoning
    return EvaluatorCharacteristics(path, labels)


def _load_xlm_roberta_tokenizer() -> Any:
    """Lazy-load the pinned fast tokenizer without loading model weights."""

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Source-length analysis requires the 'tokenizers' package. "
            "Install the project with 'python -m pip install -e .'."
        ) from exc
    try:
        return Tokenizer.from_pretrained(
            XLM_ROBERTA_TOKENIZER, revision=XLM_ROBERTA_REVISION
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Could not load the pinned XLM-R tokenizer from the local Hugging "
            "Face cache or network. Connect to the network or pre-cache it."
        ) from exc


def load_token_lengths(
    test_set_path: Path,
    ground_truth: GroundTruthSet,
    *,
    tokenizer: Any | None = None,
    batch_size: int = 1024,
) -> TokenLengthIndex:
    """Return complete, untruncated token counts for gold-backed text strings."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    active_tokenizer = (
        tokenizer if tokenizer is not None else _load_xlm_roberta_tokenizer()
    )
    active_tokenizer.no_truncation()
    source_lengths: dict[str, int] = {}
    target_lengths: dict[tuple[str, str], int] = {}
    texts: list[str] = []
    keys: list[tuple[str, str, str]] = []

    def flush() -> None:
        """Encode the pending batch without truncation or special tokens."""

        if not texts:
            return
        encodings = active_tokenizer.encode_batch(texts, add_special_tokens=False)
        if len(encodings) != len(keys):
            raise RuntimeError("Tokenizer returned an unexpected batch size")
        for key, encoding in zip(keys, encodings):
            kind, item_id, system = key
            length = len(encoding.ids)
            if kind == "source":
                source_lengths[item_id] = length
            else:
                target_lengths[(item_id, system)] = length
        texts.clear()
        keys.clear()

    for line_number, record in read_jsonl(test_set_path):
        context = f"{test_set_path}:{line_number}"
        item = parse_item_id(record.get("item_id"), context)
        if item.set_id != OFFICIAL_SET_ID or item.raw not in ground_truth.assessments:
            continue
        source = record.get("src")
        if not isinstance(source, str):
            raise ValueError(
                f"{context}: token-length analysis requires a string src field"
            )
        texts.append(source)
        keys.append(("source", item.raw, ""))
        hypotheses = record.get("hyps")
        if not isinstance(hypotheses, dict):
            raise ValueError(f"{context}: hyps must be an object")
        for system in sorted(ground_truth.assessments[item.raw]):
            translation = hypotheses.get(system)
            if not isinstance(translation, str):
                raise ValueError(
                    f"{context}: missing string hypothesis for human-annotated "
                    f"system {system!r}"
                )
            texts.append(translation)
            keys.append(("target", item.raw, system))
            if len(texts) >= batch_size:
                flush()
    flush()

    expected_sources = set(ground_truth.assessments)
    expected_targets = {
        (item_id, system)
        for item_id, systems in ground_truth.assessments.items()
        for system in systems
    }
    if (
        set(source_lengths) != expected_sources
        or set(target_lengths) != expected_targets
    ):
        raise ValueError(
            "Token-length analysis could not align every human-annotated source "
            "and hypothesis with the test set"
        )
    return TokenLengthIndex(
        XLM_ROBERTA_TOKENIZER,
        XLM_ROBERTA_REVISION,
        source_lengths,
        target_lengths,
    )


def _entry_key(descriptor: SubmissionDescriptor) -> tuple[str, str]:
    """Return the stable reasoning-metadata key for an entry."""

    if descriptor.is_baseline:
        return ("baseline", descriptor.baseline_name)
    return ("submission", descriptor.submission_id)


def _characteristic_values(
    descriptor: SubmissionDescriptor,
    metadata: ModelCardMetadata,
    evaluator_characteristics: EvaluatorCharacteristics | None,
    issues: list[Issue],
) -> dict[str, str]:
    """Normalize evaluator characteristics for descriptive entry metadata."""

    values: dict[str, str] = {}
    for key, question in EVALUATOR_CHARACTERISTIC_QUESTIONS.items():
        value = metadata.evaluator_characteristics.get(key)
        if key == "supervised":
            value = reported_fine_tuned(descriptor, value)
        if value is None:
            values[key] = "unknown"
            issues.append(
                Issue(
                    "characteristics",
                    f"{descriptor.model_card_path}: answer for {question!r} "
                    "cannot be normalized",
                    descriptor.report_label,
                    descriptor.username,
                    str(descriptor.model_card_path),
                    severity="warning",
                )
            )
        elif key == "reference_use":
            values[key] = "reference_based" if value else "reference_free"
        else:
            values[key] = "true" if value else "false"
    if evaluator_characteristics is None:
        values["reasoning"] = "unknown"
    else:
        key = _entry_key(descriptor)
        reasoning = evaluator_characteristics.reasoning_by_entry.get(key)
        if reasoning is None:
            reasoning = "unknown"
            issues.append(
                Issue(
                    "characteristics",
                    "No explicit reasoning label was provided",
                    descriptor.report_label,
                    descriptor.username,
                    str(evaluator_characteristics.path),
                    severity="warning",
                )
            )
        values["reasoning"] = reasoning
    return values


def _ground_truth_records(
    test_set: TestSetIndex, ground_truth: GroundTruthSet
) -> tuple[PreparedAssessment, ...]:
    """Build one validated prepared record per human assessment."""

    records: list[PreparedAssessment] = []
    for item_id, systems in ground_truth.assessments.items():
        direction = test_set.item_directions[item_id]
        for system, assessments in systems.items():
            target_length = test_set.translation_lengths[item_id][system]
            for reference_index, reference in enumerate(assessments):
                records.append(
                    PreparedAssessment(
                        item_id=item_id,
                        system=system,
                        language_pair=direction,
                        source_length=test_set.source_lengths.get(item_id),
                        target_length=target_length,
                        prediction=[],
                        reference=reference,
                        reference_index=reference_index,
                    )
                )
    return tuple(records)


def _ground_truth_redundancy_statistics(
    ground_truth: GroundTruthSet,
) -> GroundTruthRedundancyStatistics:
    """Summarize how often item/system pairs have multiple human assessments."""

    total_pair_count = 0
    total_assessment_count = 0
    multiple_annotation_pair_count = 0
    multiple_distinct_annotator_pair_count = 0
    same_annotator_repeated_pair_count = 0
    maximum_assessments_per_pair = 0
    for item_id, systems in ground_truth.assessments.items():
        for system, assessments in systems.items():
            annotators = [
                annotator
                for annotator in ground_truth.annotator_ids.get(item_id, {}).get(
                    system, ()
                )
                if annotator is not None
            ]
            assessment_count = len(assessments)
            total_pair_count += 1
            total_assessment_count += assessment_count
            maximum_assessments_per_pair = max(
                maximum_assessments_per_pair, assessment_count
            )
            if assessment_count > 1:
                multiple_annotation_pair_count += 1
            if len(set(annotators)) > 1:
                multiple_distinct_annotator_pair_count += 1
            if len(annotators) != len(set(annotators)):
                same_annotator_repeated_pair_count += 1
    return GroundTruthRedundancyStatistics(
        total_pair_count,
        total_assessment_count,
        multiple_annotation_pair_count,
        multiple_distinct_annotator_pair_count,
        same_annotator_repeated_pair_count,
        maximum_assessments_per_pair,
    )


def _quartile_values(values: Sequence[int]) -> tuple[float, float, float]:
    """Return inclusive quartile boundaries for a non-empty integer sequence."""

    if not values:
        raise ValueError("Cannot construct length buckets from an empty sequence")
    if len(values) == 1:
        value = float(values[0])
        return value, value, value
    first, second, third = quantiles(values, n=4, method="inclusive")
    return float(first), float(second), float(third)


def _build_length_buckets(
    records: Sequence[PreparedAssessment],
    token_lengths: TokenLengthIndex,
    directions: set[str],
) -> tuple[LengthBucketDefinition, ...]:
    """Build relative and global token quartiles for priority directions."""

    source_units: dict[str, dict[str, int]] = defaultdict(dict)
    target_units: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    combined_units: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for record in records:
        if record.language_pair not in directions:
            continue
        pair = (record.item_id, record.system)
        source_units[record.language_pair].setdefault(
            record.item_id, token_lengths.source_lengths[record.item_id]
        )
        target_units[record.language_pair].setdefault(
            pair,
            token_lengths.target_lengths[pair],
        )
        combined_units[record.language_pair].setdefault(
            pair,
            token_lengths.source_lengths[record.item_id]
            + token_lengths.target_lengths[pair],
        )
    definitions: list[LengthBucketDefinition] = []
    for basis, grouped in (
        ("source_tokens", source_units),
        ("target_tokens", target_units),
        ("source_target_tokens", combined_units),
    ):
        for language_pair, units in sorted(grouped.items()):
            values = list(units.values())
            first, second, third = _quartile_values(values)
            definitions.append(
                LengthBucketDefinition(
                    "within_direction",
                    basis,
                    language_pair,
                    first,
                    second,
                    third,
                    min(values),
                    max(values),
                    len(values),
                )
            )
        global_values = [
            value for units in grouped.values() for value in units.values()
        ]
        first, second, third = _quartile_values(global_values)
        definitions.append(
            LengthBucketDefinition(
                "global_priority",
                basis,
                GLOBAL_PRIORITY_LABEL,
                first,
                second,
                third,
                min(global_values),
                max(global_values),
                len(global_values),
            )
        )
    return tuple(definitions)


def _bucket_lookup(
    definitions: Sequence[LengthBucketDefinition],
) -> dict[tuple[str, str, str], LengthBucketDefinition]:
    """Index length definitions by scope, basis, and direction label."""

    return {
        (row.bucket_scope, row.length_basis, row.language_pair): row
        for row in definitions
    }


def _record_token_length(
    record: PreparedAssessment, basis: str, token_lengths: TokenLengthIndex
) -> int:
    """Return the requested source, target, or combined token length."""

    if basis == "source_tokens":
        return token_lengths.source_lengths[record.item_id]
    target_length = token_lengths.target_lengths[(record.item_id, record.system)]
    if basis == "target_tokens":
        return target_length
    if basis == "source_target_tokens":
        return token_lengths.source_lengths[record.item_id] + target_length
    raise ValueError(f"Unknown token-length basis: {basis!r}")


def _length_bucket(
    record: PreparedAssessment,
    scope: str,
    basis: str,
    definitions: Mapping[tuple[str, str, str], LengthBucketDefinition],
    token_lengths: TokenLengthIndex,
) -> str:
    """Assign one prepared assessment to a token-length quartile."""

    label = (
        record.language_pair if scope == "within_direction" else GLOBAL_PRIORITY_LABEL
    )
    definition = definitions[(scope, basis, label)]
    value = _record_token_length(record, basis, token_lengths)
    if value <= definition.first_quartile:
        return "Q1"
    if value <= definition.median:
        return "Q2"
    if value <= definition.third_quartile:
        return "Q3"
    return "Q4"


def _summarize_span_lists(
    values: Sequence[tuple[tuple[str, str], int, list[Span]]],
) -> SpanStatistics:
    """Count text spans, omissions, and severities in assessment lists."""

    pairs: set[tuple[str, str]] = set()
    text_span_count = 0
    text_span_character_count = 0
    omission_count = 0
    minor_error_count = 0
    major_error_count = 0
    minor_omission_count = 0
    major_omission_count = 0
    for pair, target_length, spans in values:
        pairs.add(pair)
        for span in spans:
            severity = span["severity"]
            is_omission = (
                span["start"] == target_length and span["end"] == target_length + 1
            )
            if severity == "minor":
                minor_error_count += 1
            else:
                major_error_count += 1
            if is_omission:
                omission_count += 1
                if severity == "minor":
                    minor_omission_count += 1
                else:
                    major_omission_count += 1
            else:
                text_span_count += 1
                text_span_character_count += span["end"] - span["start"]
    return SpanStatistics(
        len(pairs),
        len(values),
        text_span_count,
        text_span_character_count,
        omission_count,
        minor_error_count,
        major_error_count,
        minor_omission_count,
        major_omission_count,
    )


def statistics_for_records(
    records: Sequence[PreparedAssessment],
) -> tuple[SpanStatistics, SpanStatistics]:
    """Count unique evaluator predictions and every aligned human assessment."""

    unique_predictions: dict[
        tuple[str, str], tuple[tuple[str, str], int, list[Span]]
    ] = {}
    gold_values: list[tuple[tuple[str, str], int, list[Span]]] = []
    for record in records:
        pair = (record.item_id, record.system)
        unique_predictions.setdefault(
            pair, (pair, record.target_length, record.prediction)
        )
        gold_values.append((pair, record.target_length, record.reference))
    return (
        _summarize_span_lists(list(unique_predictions.values())),
        _summarize_span_lists(gold_values),
    )


def core_span_metric_values(
    statistics: SpanStatistics,
) -> dict[str, float | None]:
    """Return the three non-redundant span-behavior predictors."""

    return {
        "mean_text_span_length_characters": statistics.mean_text_span_length_characters,
        "mean_total_errors_per_assessment": statistics.mean_total_errors_per_assessment,
        "mean_severity_weight": statistics.mean_severity_weight,
    }


def _length_behavior_values(
    rows: Sequence[ScoredAssessment],
) -> dict[str, float | None]:
    """Return direction-equal behavior means from unique bucket predictions.

    Reuse the reference/thinking analysis counting policy, including omission
    errors and excluding omissions from text-span length. Undefined direction
    values are omitted independently for each measure; empty buckets stay null.
    """

    grouped: dict[str, list[PreparedAssessment]] = defaultdict(list)
    for row in rows:
        grouped[row.assessment.language_pair].append(row.assessment)
    statistics = [statistics_for_records(records)[0] for records in grouped.values()]
    result: dict[str, float | None] = {}
    for metric in CORE_SPAN_METRICS:
        values = [getattr(value, metric) for value in statistics]
        defined = [value for value in values if value is not None]
        result[metric] = fmean(defined) if defined else None
    return result


def _macro_f1(score: SubmissionScore, directions: set[str]) -> tuple[float, int]:
    """Macro-average official direction F1 over a non-empty direction set."""

    retained = [
        value for value in score.direction_scores if value.language_pair in directions
    ]
    if not retained:
        raise ValueError(
            "No human-annotation-backed directions were available for this cohort"
        )
    return (
        fmean(value.f1 for value in retained),
        sum(value.aligned_assessments for value in retained),
    )


def _compute_record_mpp(
    records: Sequence[ScoredAssessment],
) -> tuple[float | None, int, int]:
    """Regroup cached MPP components and macro-average direction F1."""

    grouped: dict[str, list[ScoredAssessment]] = defaultdict(list)
    for record in records:
        grouped[record.assessment.language_pair].append(record)
    if not grouped:
        return None, 0, 0
    f1: list[float] = []
    aligned = 0
    for language_pair in sorted(grouped):
        direction_records = grouped[language_pair]
        _, _, direction_f1 = _precision_recall_fscore(
            _sum_mpp_components(direction_records)
        )
        f1.append(direction_f1)
        aligned += len(direction_records)
    return fmean(f1), len(grouped), aligned


def _cohort_directions(cohort: str, records: Sequence[PreparedAssessment]) -> set[str]:
    """Return human-annotation-backed directions for an eligibility cohort."""

    available = {record.language_pair for record in records}
    if cohort == "priority":
        return available & PRIMARY_LANGUAGE_PAIRS
    return available


def _eligible(score: SubmissionScore, cohort: str) -> bool:
    """Return whether an evaluator meets a fixed cohort coverage gate."""

    if cohort == "priority":
        return score.coverage.all_primary_complete
    return score.coverage.all_official_complete


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Rank values from one, assigning the average rank to exact ties."""

    ordered = sorted(enumerate(values), key=lambda value: value[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def _spearman(
    x_values: Sequence[float], y_values: Sequence[float]
) -> tuple[float | None, str]:
    """Compute Spearman as Pearson correlation over average ranks."""

    if len(x_values) < 3:
        return None, "insufficient_observations"
    if len(set(x_values)) < 2 or len(set(y_values)) < 2:
        return None, "constant_value"
    x_ranks = _average_ranks(x_values)
    y_ranks = _average_ranks(y_values)
    x_mean = fmean(x_ranks)
    y_mean = fmean(y_ranks)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_ranks, y_ranks)
    )
    denominator = math.sqrt(
        sum((value - x_mean) ** 2 for value in x_ranks)
        * sum((value - y_mean) ** 2 for value in y_ranks)
    )
    if denominator == 0:
        return None, "constant_value"
    return numerator / denominator, "ok"


def _quartile_band(values: Sequence[float]) -> tuple[float, float]:
    """Return inclusive first and third quartiles for a non-empty sequence."""

    if len(values) == 1:
        return values[0], values[0]
    first, _, third = quantiles(values, n=4, method="inclusive")
    return float(first), float(third)


def _build_correlations(
    rows: Sequence[DirectionStatisticsRow], directions: Sequence[str]
) -> tuple[tuple[CorrelationRow, ...], tuple[CorrelationSummaryRow, ...]]:
    """Compute per-direction Spearman values and direction-equal summaries."""

    output: list[CorrelationRow] = []
    for language_pair in directions:
        selected = [row for row in rows if row.language_pair == language_pair]
        for predictor in CORE_SPAN_METRICS:
            pairs: list[tuple[float, float]] = []
            undefined: list[str] = []
            for row in selected:
                if not math.isfinite(row.f1):
                    raise ValueError(
                        f"Spearman correlation requires finite F1; "
                        f"{row.descriptor.report_label} has non-finite F1 for {language_pair}"
                    )
                predictor_value = core_span_metric_values(row.prediction_statistics)[
                    predictor
                ]
                if predictor_value is None:
                    undefined.append(row.descriptor.report_label)
                    continue
                if not math.isfinite(float(predictor_value)):
                    raise ValueError(
                        "Spearman correlation requires every complete "
                        "evaluator-direction observation to have a finite "
                        f"{predictor} value; {row.descriptor.report_label} has "
                        f"no finite value for {language_pair}"
                    )
                pairs.append((float(predictor_value), row.f1))
            coefficient, status = (
                (None, "undefined_predictor")
                if undefined
                else _spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])
            )
            output.append(
                CorrelationRow(
                    language_pair,
                    predictor,
                    len(selected),
                    coefficient,
                    status,
                    tuple(sorted(undefined)),
                )
            )
    summaries: list[CorrelationSummaryRow] = []
    for predictor in CORE_SPAN_METRICS:
        values = [
            row.spearman_correlation
            for row in output
            if row.predictor == predictor
            and row.status == "ok"
            and row.spearman_correlation is not None
        ]
        if values:
            first, third = _quartile_band(values)
            median_value: float | None = median(values)
        else:
            first = third = median_value = None
        summaries.append(
            CorrelationSummaryRow(
                predictor, len(values), len(directions), median_value, first, third
            )
        )
    return tuple(output), tuple(summaries)


def _build_length_summaries(
    rows: Sequence[LengthStatisticsRow],
) -> tuple[LengthSummaryRow, ...]:
    """Summarize priority-evaluator F1 distributions for each length bucket."""

    summaries: list[LengthSummaryRow] = []
    for scope in LENGTH_BUCKET_SCOPES:
        for basis in LENGTH_BASES:
            for bucket in LENGTH_BUCKETS:
                values = [
                    float(row.f1)
                    for row in rows
                    if row.bucket_scope == scope
                    and row.length_basis == basis
                    and row.length_bucket == bucket
                    and row.f1 is not None
                    and math.isfinite(float(row.f1))
                ]
                if values:
                    first, third = _quartile_band(values)
                    median_value: float | None = median(values)
                else:
                    first = third = median_value = None
                summaries.append(
                    LengthSummaryRow(
                        scope, basis, bucket, len(values), median_value, first, third
                    )
                )
    return tuple(summaries)


def analyze_score_statistics(
    submissions: Iterable[SubmissionDescriptor],
    test_set: TestSetIndex,
    ground_truth: GroundTruthSet,
    model_card_template: ModelCardTemplate | None,
    *,
    evaluator_characteristics: EvaluatorCharacteristics | None = None,
    token_lengths: TokenLengthIndex | None = None,
    require_complete_predictions: bool = True,
    mpp_cache: MppCache | None = None,
    recompute_mpp: bool = False,
    matching_algorithm: str = "greedy",
) -> BatchStatisticsResult:
    """Score entries and compute focused gold-aligned exploratory statistics."""

    gold_records = _ground_truth_records(test_set, ground_truth)
    ground_truth_redundancy = _ground_truth_redundancy_statistics(ground_truth)
    priority_directions = _cohort_directions("priority", gold_records)
    bucket_definitions = (
        _build_length_buckets(gold_records, token_lengths, priority_directions)
        if token_lengths is not None
        else ()
    )
    bucket_lookup = _bucket_lookup(bucket_definitions)
    issues: list[Issue] = []
    if evaluator_characteristics is None:
        issues.append(
            Issue(
                "characteristics",
                "No evaluator-characteristics CSV was supplied; reasoning metadata "
                "is unknown",
                severity="warning",
            )
        )
    gold_rows: list[GoldStatisticsRow] = []
    for cohort in COHORTS:
        directions = _cohort_directions(cohort, gold_records)
        scoped_gold = [
            record for record in gold_records if record.language_pair in directions
        ]
        gold_rows.append(
            GoldStatisticsRow(
                cohort,
                tuple(sorted(directions)),
                _summarize_span_lists(
                    [
                        (
                            (record.item_id, record.system),
                            record.target_length,
                            record.reference,
                        )
                        for record in scoped_gold
                    ]
                ),
            )
        )
    scores: list[SubmissionScore] = []
    entry_rows: list[EntryStatisticsRow] = []
    direction_rows: list[DirectionStatisticsRow] = []
    length_rows: list[LengthStatisticsRow] = []
    for submission in submissions:
        logging.debug(
            "Processing score statistics for %s (%s)",
            submission.report_label,
            submission.username,
        )
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
            )
            score, scored_assessments = _score_prepared_submission_with_components(
                submission,
                metadata,
                prepared,
                mpp_cache=mpp_cache,
                recompute_mpp=recompute_mpp,
                matching_algorithm=matching_algorithm,
            )
            characteristics = _characteristic_values(
                submission, metadata, evaluator_characteristics, issues
            )
            issues.extend(score.issues)
            scores.append(score)
            available_directions = {
                record.language_pair for record in prepared.assessments
            }
            available_prediction, available_gold = statistics_for_records(
                prepared.assessments
            )
            available_f1, available_aligned = _macro_f1(score, available_directions)
            entry_rows.append(
                EntryStatisticsRow(
                    submission,
                    metadata,
                    "available",
                    tuple(sorted(available_directions)),
                    available_f1,
                    available_aligned,
                    available_prediction,
                    available_gold,
                    characteristics,
                )
            )
            direction_score_lookup = {
                row.language_pair: row for row in score.direction_scores
            }
            for direction in sorted(score.coverage.complete_directions):
                direction_score = direction_score_lookup.get(direction)
                if direction_score is None:
                    continue
                records = [
                    record
                    for record in prepared.assessments
                    if record.language_pair == direction
                ]
                if not records:
                    continue
                prediction_statistics, _ = statistics_for_records(records)
                direction_rows.append(
                    DirectionStatisticsRow(
                        submission,
                        direction,
                        direction_score.f1,
                        prediction_statistics,
                    )
                )
            for cohort in COHORTS:
                if not _eligible(score, cohort):
                    continue
                directions = _cohort_directions(cohort, prepared.assessments)
                if not directions:
                    continue
                cohort_records = [
                    record
                    for record in prepared.assessments
                    if record.language_pair in directions
                ]
                prediction_statistics, gold_statistics = statistics_for_records(
                    cohort_records
                )
                cohort_f1, aligned = _macro_f1(score, directions)
                entry_rows.append(
                    EntryStatisticsRow(
                        submission,
                        metadata,
                        cohort,
                        tuple(sorted(directions)),
                        cohort_f1,
                        aligned,
                        prediction_statistics,
                        gold_statistics,
                        characteristics,
                    )
                )
            if token_lengths is not None and score.coverage.all_primary_complete:
                priority_scored = [
                    row
                    for row in scored_assessments
                    if row.assessment.language_pair in priority_directions
                ]
                for scope in LENGTH_BUCKET_SCOPES:
                    for basis in LENGTH_BASES:
                        for bucket in LENGTH_BUCKETS:
                            selected = [
                                row
                                for row in priority_scored
                                if _length_bucket(
                                    row.assessment,
                                    scope,
                                    basis,
                                    bucket_lookup,
                                    token_lengths,
                                )
                                == bucket
                            ]
                            f1, direction_count, aligned = _compute_record_mpp(selected)
                            length_rows.append(
                                LengthStatisticsRow(
                                    submission,
                                    metadata,
                                    scope,
                                    basis,
                                    bucket,
                                    direction_count,
                                    f1,
                                    aligned,
                                    **_length_behavior_values(selected),
                                )
                            )
        except (OSError, ValueError, RuntimeError) as exc:
            logging.error("%s failed: %s", submission.report_label, exc)
            issues.append(
                Issue(
                    "score_statistics",
                    str(exc),
                    submission.report_label,
                    submission.username,
                    str(submission.path),
                )
            )
    correlation_directions = sorted({record.language_pair for record in gold_records})
    correlations, correlation_summaries = _build_correlations(
        direction_rows, correlation_directions
    )
    length_summaries = _build_length_summaries(length_rows)
    return BatchStatisticsResult(
        tuple(scores),
        tuple(entry_rows),
        tuple(gold_rows),
        tuple(direction_rows),
        tuple(length_rows),
        length_summaries,
        tuple(bucket_definitions),
        correlations,
        correlation_summaries,
        ground_truth_redundancy,
        token_lengths,
        tuple(issues),
        dict(test_set.reference_systems_by_direction),
        tuple(
            sorted(
                test_set.official_directions - test_set.reference_available_directions
            )
        ),
        matching_algorithm,
    )
