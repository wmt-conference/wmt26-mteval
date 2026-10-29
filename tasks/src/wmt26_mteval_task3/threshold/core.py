"""Threshold optimization and validated Task 2 score loading."""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Iterable

from ..io import DataError, iter_jsonl, parse_item_id
from ..models import HumanReference, OfficialInventory
from ..task2_conversion import ConversionDecision, _numeric_scores

NEAR_OPTIMAL_TOLERANCE = 0.005
CV_FOLDS = 5


@dataclass(frozen=True)
class Observation:
    """Store typed observation state shared by the original analysis routines."""
    score: float
    gold: bool
    language_pair: str
    item_id: str
    system: str


@dataclass(frozen=True)
class ScoreData:
    """Store typed score data state shared by the original analysis routines."""
    observations: tuple[Observation, ...]
    attempted_pairs: tuple[str, ...]
    official_labels: int
    source_score_min: float
    source_score_max: float


@dataclass(frozen=True)
class CurvePoint:
    """Store typed curve point state shared by the original analysis routines."""
    threshold: float
    mcc: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        """Compute precision from the supplied analysis state and return its representation."""
        return precision_from_counts(self.true_positive, self.false_positive)

    @property
    def recall(self) -> float:
        """Compute recall from the supplied analysis state and return its representation."""
        return recall_from_counts(self.true_positive, self.false_negative)


@dataclass(frozen=True)
class SweepResult:
    """Store typed sweep result state shared by the original analysis routines."""
    points: tuple[CurvePoint, ...]
    best_index: int
    exact_plateau_start: int
    exact_plateau_end: int
    near_optimal_start: int
    near_optimal_end: int

    @property
    def best(self) -> CurvePoint:
        """Compute best from the supplied analysis state and return its representation."""
        return self.points[self.best_index]


@dataclass(frozen=True)
class LanguageAnalysis:
    """Store typed language analysis state shared by the original analysis routines."""
    row: dict[str, Any]
    plot_points: tuple[CurvePoint, ...]


@dataclass(frozen=True)
class AnalysisResult:
    """Store typed analysis result state shared by the original analysis routines."""
    row: dict[str, Any]
    plot_points: tuple[CurvePoint, ...]
    declared_pair_counts: dict[str, tuple[int, int, int, int]]
    languages: tuple[LanguageAnalysis, ...]

    def language(self, language_pair: str) -> LanguageAnalysis | None:
        """Compute language from the supplied analysis state and return its representation."""
        return next(
            (
                analysis
                for analysis in self.languages
                if analysis.row["language_pair"] == language_pair
            ),
            None,
        )


def mcc_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    """Return binary MCC, using zero when the denominator is degenerate."""
    numerator = tp * tn - fp * fn
    denominator_squared = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator_squared == 0:
        return 0.0
    return numerator / math.sqrt(denominator_squared)


def precision_from_counts(tp: int, fp: int) -> float:
    """Return error-free precision with a zero-division value of zero."""
    return tp / (tp + fp) if tp + fp else 0.0


def recall_from_counts(tp: int, fn: int) -> float:
    """Return error-free recall with a zero-division value of zero."""
    return tp / (tp + fn) if tp + fn else 0.0


def confusion(
    observations: Iterable[Observation], predicate: Callable[[float], bool]
) -> tuple[int, int, int, int]:
    """Return TP/FP/FN/TN for supplied scored observations and a threshold predicate."""
    tp = fp = fn = tn = 0
    for observation in observations:
        prediction = predicate(observation.score)
        if observation.gold and prediction:
            tp += 1
        elif prediction:
            fp += 1
        elif observation.gold:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def representative_threshold(scores: list[float], cut: int) -> float:
    """Choose the original representative boundary without splitting equal scores."""
    if cut == 0:
        return math.nextafter(scores[0], math.inf)
    upper = scores[cut - 1]
    if cut == len(scores):
        return upper
    lower = scores[cut]
    midpoint = lower + (upper - lower) / 2
    return midpoint if lower < midpoint <= upper else upper


def optimize_threshold(
    observations: Iterable[Observation], anchor_threshold: float
) -> SweepResult:
    """Sweep all realizable score boundaries and return the original tie-broken optimum."""
    ordered = sorted(observations, key=lambda observation: observation.score, reverse=True)
    if not ordered:
        raise ValueError("Cannot optimize a threshold without observations")

    grouped: list[tuple[float, int, int]] = []
    for score, members in groupby(ordered, key=lambda observation: observation.score):
        group = list(members)
        positives = sum(observation.gold for observation in group)
        grouped.append((score, positives, len(group) - positives))

    scores = [score for score, _, _ in grouped]
    total_positive = sum(observation.gold for observation in ordered)
    total_negative = len(ordered) - total_positive
    tp = fp = 0
    points = [
        CurvePoint(
            representative_threshold(scores, 0),
            0.0,
            0,
            0,
            total_positive,
            total_negative,
        )
    ]
    for cut, (_, group_positive, group_negative) in enumerate(grouped, start=1):
        tp += group_positive
        fp += group_negative
        fn = total_positive - tp
        tn = total_negative - fp
        points.append(
            CurvePoint(
                representative_threshold(scores, cut),
                mcc_from_counts(tp, fp, fn, tn),
                tp,
                fp,
                fn,
                tn,
            )
        )

    best_mcc = max(point.mcc for point in points)
    best_indexes = [
        index
        for index, point in enumerate(points)
        if math.isclose(point.mcc, best_mcc, rel_tol=0.0, abs_tol=1e-15)
    ]
    runs: list[list[int]] = []
    for index in best_indexes:
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    longest = max(len(run) for run in runs)
    candidate_runs = [run for run in runs if len(run) == longest]
    selected_run = min(
        candidate_runs,
        key=lambda run: abs(points[run[len(run) // 2]].threshold - anchor_threshold),
    )
    best_index = selected_run[len(selected_run) // 2]

    near_limit = best_mcc - NEAR_OPTIMAL_TOLERANCE
    near_start = best_index
    near_end = best_index
    while near_start > 0 and points[near_start - 1].mcc >= near_limit:
        near_start -= 1
    while near_end + 1 < len(points) and points[near_end + 1].mcc >= near_limit:
        near_end += 1

    return SweepResult(
        points=tuple(points),
        best_index=best_index,
        exact_plateau_start=selected_run[0],
        exact_plateau_end=selected_run[-1],
        near_optimal_start=near_start,
        near_optimal_end=near_end,
    )


def load_score_data(
    decision: ConversionDecision,
    task2_dir: Path,
    inventory: OfficialInventory,
    human_reference: HumanReference,
    language_pairs: frozenset[str] | None = None,
) -> ScoreData:
    """Validate complete coverage and expand scores against all retained human judgments."""
    segments_path = task2_dir / decision.directory_name / decision.segments_file
    submitted_ids: set[str] = set()
    submitted_by_pair: defaultdict[str, set[str]] = defaultdict(set)
    observations: list[Observation] = []
    score_min = math.inf
    score_max = -math.inf
    official_labels = 0

    for line_number, record in iter_jsonl(segments_path):
        source = f"{segments_path}:{line_number}"
        if "item_id" not in record:
            raise DataError(f"{source}: missing item_id")
        item_id = record["item_id"]
        parsed_id = parse_item_id(item_id, source)
        if parsed_id.set_id != "official":
            continue
        if "task2_pred" not in record:
            raise DataError(f"{source}: official record is missing task2_pred")
        scores = _numeric_scores(record["task2_pred"], source)
        if item_id in submitted_ids:
            raise DataError(f"{source}: duplicate official item_id {item_id!r}")
        submitted_ids.add(item_id)

        test_item = inventory.items.get(item_id)
        if test_item is None:
            raise DataError(f"{source}: unknown official item_id {item_id!r}")
        if set(scores) != set(test_item.hypotheses):
            raise DataError(f"{source}: official system coverage mismatch")
        pair = parsed_id.language_pair
        submitted_by_pair[pair].add(item_id)
        if language_pairs is not None and pair not in language_pairs:
            continue
        for system, score in scores.items():
            score_min = min(score_min, score)
            score_max = max(score_max, score)
            official_labels += 1
            observations.extend(
                Observation(score, human.error_free, pair, item_id, system)
                for human in human_reference.judgments_for((item_id, system))
            )

    if not submitted_by_pair:
        raise DataError(f"{segments_path}: no official predictions")
    for pair, actual_ids in submitted_by_pair.items():
        expected_ids = inventory.item_ids_by_pair.get(pair)
        if expected_ids is None or actual_ids != expected_ids:
            expected_count = 0 if expected_ids is None else len(expected_ids)
            raise DataError(
                f"{segments_path}: incomplete {pair} coverage; expected "
                f"{expected_count} items, found {len(actual_ids)}"
            )
    submitted_pairs = set(submitted_by_pair)
    if language_pairs is not None and not language_pairs <= submitted_pairs:
        missing = ", ".join(sorted(language_pairs - submitted_pairs))
        raise DataError(f"{segments_path}: missing required language pairs: {missing}")
    selected_pairs = submitted_pairs if language_pairs is None else set(language_pairs)
    observed_pairs = {observation.language_pair for observation in observations}
    if observed_pairs != selected_pairs:
        raise DataError(f"{segments_path}: at least one submitted pair has no human labels")

    return ScoreData(
        observations=tuple(observations),
        attempted_pairs=tuple(sorted(selected_pairs)),
        official_labels=official_labels,
        source_score_min=score_min,
        source_score_max=score_max,
    )


def pair_metrics(
    observations: Iterable[Observation], predicate: Callable[[float], bool]
) -> tuple[dict[str, tuple[int, int, int, int]], float]:
    """Return per-pair confusion counts and macro MCC for a score decision predicate."""
    by_pair: defaultdict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_pair[observation.language_pair].append(observation)
    counts = {
        pair: confusion(pair_observations, predicate)
        for pair, pair_observations in sorted(by_pair.items())
    }
    return counts, statistics.fmean(
        mcc_from_counts(*value) for value in counts.values()
    )


def downsample_curve(
    points: tuple[CurvePoint, ...], preserve_indexes: set[int], limit: int = 800
) -> tuple[CurvePoint, ...]:
    """Select a bounded display curve while preserving marked/optimal thresholds."""
    if len(points) <= limit:
        return points
    selected = {0, len(points) - 1, *preserve_indexes}
    metrics = (
        lambda point: point.mcc,
        lambda point: point.precision,
        lambda point: point.recall,
    )
    bucket_count = max(1, (limit - len(selected)) // (2 * len(metrics)))
    bucket_size = math.ceil(len(points) / bucket_count)
    for start in range(0, len(points), bucket_size):
        indexes = range(start, min(start + bucket_size, len(points)))
        for metric in metrics:
            selected.add(min(indexes, key=lambda index: metric(points[index])))
            selected.add(max(indexes, key=lambda index: metric(points[index])))
    return tuple(points[index] for index in sorted(selected))


def analyze_per_language(
    observations: Iterable[Observation],
    anchor_threshold: float,
    declared_predicate: Callable[[float], bool],
    pooled_optimal_threshold: float,
) -> tuple[LanguageAnalysis, ...]:
    """Compute declared, pooled and locally optimized metrics for each language pair."""
    by_pair: defaultdict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_pair[observation.language_pair].append(observation)

    analyses = []
    for pair, pair_observations in sorted(by_pair.items()):
        declared_counts = confusion(pair_observations, declared_predicate)
        pooled_counts = confusion(
            pair_observations,
            lambda score: score >= pooled_optimal_threshold,
        )
        sweep = optimize_threshold(pair_observations, anchor_threshold)
        optimal = sweep.best
        nearest_declared = min(
            range(len(sweep.points)),
            key=lambda index: abs(
                sweep.points[index].threshold - anchor_threshold
            ),
        )
        declared_mcc = mcc_from_counts(*declared_counts)
        pooled_mcc = mcc_from_counts(*pooled_counts)
        analyses.append(
            LanguageAnalysis(
                row={
                    "language_pair": pair,
                    "human_assessments": len(pair_observations),
                    "source_score_min": min(value.score for value in pair_observations),
                    "source_score_max": max(value.score for value in pair_observations),
                    "observed_unique_scores": len(
                        {value.score for value in pair_observations}
                    ),
                    "declared_mcc": declared_mcc,
                    "pooled_optimal_threshold_mcc": pooled_mcc,
                    "language_optimal_threshold": optimal.threshold,
                    "language_optimal_mcc": optimal.mcc,
                    "language_optimal_mcc_delta_vs_declared": (
                        optimal.mcc - declared_mcc
                    ),
                    "language_optimal_mcc_delta_vs_pooled_optimum": (
                        optimal.mcc - pooled_mcc
                    ),
                    "true_positive_language_optimal": optimal.true_positive,
                    "false_positive_language_optimal": optimal.false_positive,
                    "false_negative_language_optimal": optimal.false_negative,
                    "true_negative_language_optimal": optimal.true_negative,
                },
                plot_points=downsample_curve(
                    sweep.points,
                    {sweep.best_index, nearest_declared},
                ),
            )
        )
    return tuple(analyses)


def optimize_per_language(
    observations: Iterable[Observation],
    anchor_threshold: float,
    declared_predicate: Callable[[float], bool],
    pooled_optimal_threshold: float,
) -> tuple[dict[str, Any], ...]:
    """Return per-language exact optima and their unweighted macro MCC."""
    return tuple(
        analysis.row
        for analysis in analyze_per_language(
            observations,
            anchor_threshold,
            declared_predicate,
            pooled_optimal_threshold,
        )
    )


def item_fold(item_id: str, folds: int) -> int:
    """Assign a complete item ID deterministically to one cross-validation fold."""
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def cross_validated_mcc(
    observations: tuple[Observation, ...], anchor_threshold: float, folds: int
) -> tuple[float, list[float]]:
    """Fit pooled thresholds on training items and score held-out item-grouped folds."""
    fold_ids = [item_fold(observation.item_id, folds) for observation in observations]
    aggregate = [0, 0, 0, 0]
    thresholds = []
    for held_out in range(folds):
        training = [
            observation
            for observation, fold in zip(observations, fold_ids)
            if fold != held_out
        ]
        test = [
            observation
            for observation, fold in zip(observations, fold_ids)
            if fold == held_out
        ]
        if not training or not test:
            raise ValueError(f"Fold {held_out} has no training or test observations")
        threshold = optimize_threshold(training, anchor_threshold).best.threshold
        thresholds.append(threshold)
        counts = confusion(test, lambda score, value=threshold: score >= value)
        for index, count in enumerate(counts):
            aggregate[index] += count
    return mcc_from_counts(*aggregate), thresholds


def analyze_system(
    track: str,
    decision: ConversionDecision,
    display_name: str,
    task2_dir: Path,
    inventory: OfficialInventory,
    human_reference: HumanReference,
    language_pairs: frozenset[str] | None = None,
) -> AnalysisResult:
    """Return the original pooled, per-language and cross-validated sensitivity results."""
    if decision.status != "converted" or decision.threshold is None:
        raise ValueError(f"{decision.submission_id} is not an eligible conversion")
    if decision.inclusive is None:
        raise ValueError(f"{decision.submission_id} has no comparison rule")

    data = load_score_data(
        decision, task2_dir, inventory, human_reference, language_pairs
    )
    declared_predicate = (
        (lambda score: score >= decision.threshold)
        if decision.inclusive
        else (lambda score: score > decision.threshold)
    )
    declared_counts = confusion(data.observations, declared_predicate)
    declared_pair_counts, declared_macro = pair_metrics(
        data.observations, declared_predicate
    )
    sweep = optimize_threshold(data.observations, decision.threshold)
    optimal = sweep.best
    _, optimal_macro = pair_metrics(
        data.observations,
        lambda score: score >= optimal.threshold,
    )
    languages = analyze_per_language(
        data.observations,
        decision.threshold,
        declared_predicate,
        optimal.threshold,
    )
    language_optimal_macro = statistics.fmean(
        language.row["language_optimal_mcc"] for language in languages
    )
    cv_mcc, cv_thresholds = cross_validated_mcc(
        data.observations, decision.threshold, CV_FOLDS
    )

    near_thresholds = [
        point.threshold
        for point in sweep.points[
            sweep.near_optimal_start : sweep.near_optimal_end + 1
        ]
    ]
    item_system_keys = {
        (observation.item_id, observation.system) for observation in data.observations
    }
    declared_mcc = mcc_from_counts(*declared_counts)
    nearest_declared = min(
        range(len(sweep.points)),
        key=lambda index: abs(sweep.points[index].threshold - decision.threshold),
    )
    row = {
        "track": track,
        "entry_id": decision.submission_id,
        "display_name": display_name,
        "organization_name": decision.organization_name,
        "submission_name": decision.submission_name,
        "base_model": decision.base_model,
        "uses_reference": "" if decision.uses_reference is None else decision.uses_reference,
        "language_pairs": len(data.attempted_pairs),
        "official_labels": data.official_labels,
        "annotated_items": len({observation.item_id for observation in data.observations}),
        "annotated_item_systems": len(item_system_keys),
        "human_assessments": len(data.observations),
        "gold_error_free_rate": sum(observation.gold for observation in data.observations)
        / len(data.observations),
        "source_score_min": data.source_score_min,
        "source_score_max": data.source_score_max,
        "observed_unique_scores": len({observation.score for observation in data.observations}),
        "declared_comparison": decision.comparison,
        "declared_threshold": decision.threshold,
        "declared_pooled_mcc": declared_mcc,
        "declared_macro_language_mcc": declared_macro,
        "optimal_comparison": ">=",
        "optimal_threshold": optimal.threshold,
        "optimal_pooled_mcc": optimal.mcc,
        "optimal_macro_language_mcc": optimal_macro,
        "language_optimal_macro_mcc": language_optimal_macro,
        "apparent_pooled_mcc_delta": optimal.mcc - declared_mcc,
        "pooled_optimum_macro_language_mcc_delta": optimal_macro - declared_macro,
        "language_optimal_macro_mcc_delta_vs_declared": (
            language_optimal_macro - declared_macro
        ),
        "language_optimal_macro_mcc_delta_vs_pooled_optimum": (
            language_optimal_macro - optimal_macro
        ),
        "optimal_exact_plateau_cutpoints": (
            sweep.exact_plateau_end - sweep.exact_plateau_start + 1
        ),
        "near_optimal_tolerance": NEAR_OPTIMAL_TOLERANCE,
        "near_optimal_threshold_min": min(near_thresholds),
        "near_optimal_threshold_max": max(near_thresholds),
        "near_optimal_cutpoints": (
            sweep.near_optimal_end - sweep.near_optimal_start + 1
        ),
        "cv_folds": CV_FOLDS,
        "cv_item_grouped_pooled_mcc": cv_mcc,
        "cv_pooled_mcc_delta_vs_declared": cv_mcc - declared_mcc,
        "cv_selected_threshold_median": statistics.median(cv_thresholds),
        "cv_selected_threshold_min": min(cv_thresholds),
        "cv_selected_threshold_max": max(cv_thresholds),
        "true_positive_declared": declared_counts[0],
        "false_positive_declared": declared_counts[1],
        "false_negative_declared": declared_counts[2],
        "true_negative_declared": declared_counts[3],
        "true_positive_optimal": optimal.true_positive,
        "false_positive_optimal": optimal.false_positive,
        "false_negative_optimal": optimal.false_negative,
        "true_negative_optimal": optimal.true_negative,
    }
    language_rows = tuple(
        LanguageAnalysis(
            row={
                "track": track,
                "entry_id": decision.submission_id,
                "submission_name": decision.submission_name,
                "base_model": decision.base_model,
                **language.row,
            },
            plot_points=language.plot_points,
        )
        for language in languages
    )
    return AnalysisResult(
        row=row,
        plot_points=downsample_curve(
            sweep.points,
            {sweep.best_index, nearest_declared},
        ),
        declared_pair_counts=declared_pair_counts,
        languages=language_rows,
    )


def load_official_metrics(path: Path, id_column: str) -> dict[str, list[dict[str, str]]]:
    """Group official result CSV rows by stable system identity."""
    rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row[id_column]].append(row)
    return dict(rows)


def pooled_metrics_from_official_rows(
    rows: list[dict[str, str]],
) -> dict[str, float]:
    """Pool official confusion counts using the original metric convention."""
    counts = [
        sum(int(row[field]) for row in rows)
        for field in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        )
    ]
    tp, fp, fn, tn = counts
    return {
        "mcc": mcc_from_counts(tp, fp, fn, tn),
        "precision": precision_from_counts(tp, fp),
        "recall": recall_from_counts(tp, fn),
    }


def verify_declared_result(
    result: AnalysisResult, official_rows: list[dict[str, str]]
) -> None:
    """Reject any mismatch between threshold analysis and official declared-rule counts."""
    expected_pairs = set(result.declared_pair_counts)
    actual_pairs = {row["language_pair"] for row in official_rows}
    if expected_pairs != actual_pairs:
        raise DataError(
            f"{result.row['entry_id']}: official metric language pairs do not match"
        )
    for row in official_rows:
        actual = result.declared_pair_counts[row["language_pair"]]
        expected = tuple(
            int(row[field])
            for field in (
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
            )
        )
        if actual != expected:
            raise DataError(
                f"{result.row['entry_id']} {row['language_pair']}: "
                f"declared confusion counts differ: {actual} != {expected}"
            )
