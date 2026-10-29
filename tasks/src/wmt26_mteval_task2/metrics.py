"""PDP and SPA implementations for WMT26 contrastive human evidence."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from wmt26_mteval_task2.constants import (
    DEFAULT_SPA_PERMUTATIONS,
    DEFAULT_SPA_SEED,
    GROUP_BY_SCREEN_PDP,
    GROUP_BY_SEGMENT_PDP,
    SCREEN_BASED_SPA,
    SEGMENT_BASED_SPA,
)
from wmt26_mteval_task2.models import HumanEvidence


Predictions = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class ScreenSpaReferencePair:
    """Precomputed human SPA evidence for one co-screened system pair."""

    system_a: str
    system_b: str
    item_ids: tuple[str, ...]
    signs: np.ndarray
    human_pvalue: float


@dataclass(frozen=True)
class SegmentSpaReferencePair:
    """Human SPA evidence for one pair on its common annotated items."""

    system_a: str
    system_b: str
    item_ids: tuple[str, ...]
    signs: np.ndarray
    human_pvalue: float


@dataclass(frozen=True)
class SpaReference:
    """Reusable human-side permutation results for one direction.

    Task 2 ranks many evaluation systems against identical human evidence.
    Precomputing this object avoids repeating the intrinsic human permutation
    tests once per submission while preserving the exact upstream algorithm.
    """

    systems: tuple[str, ...]
    items: tuple[str, ...]
    segment_signs: np.ndarray
    segment_human_pvalues: Mapping[tuple[int, int], float]
    screen_pairs: tuple[ScreenSpaReferencePair, ...]
    segment_pairs: tuple[SegmentSpaReferencePair, ...] = ()


def pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Pearson's correlation, or zero for insufficient/constant input."""

    if len(first) != len(second):
        raise ValueError("Pearson vectors must have equal lengths")
    if len(first) < 2:
        return 0.0
    mean_first = math.fsum(first) / len(first)
    mean_second = math.fsum(second) / len(second)
    centered_first = [value - mean_first for value in first]
    centered_second = [value - mean_second for value in second]
    covariance = math.fsum(
        left * right for left, right in zip(centered_first, centered_second)
    )
    first_variance = math.fsum(value * value for value in centered_first)
    second_variance = math.fsum(value * value for value in centered_second)
    denominator = math.sqrt(first_variance * second_variance)
    return covariance / denominator if denominator else 0.0


def _append_ordered_differences(
    human: list[float],
    metric: list[float],
    human_a: float,
    human_b: float,
    metric_a: float,
    metric_b: float,
) -> None:
    """Append both orientations of one non-self within-item comparison."""

    human_difference = human_a - human_b
    metric_difference = metric_a - metric_b
    human.extend((human_difference, -human_difference))
    metric.extend((metric_difference, -metric_difference))


def score_pdp(
    evidence: HumanEvidence,
    predictions: Predictions,
    *,
    screen_based: bool,
) -> tuple[float, int]:
    """Compute group-by-screen or group-by-segment PDP.

    The returned count is the number of ordered non-self differences supplied
    to the single global Pearson correlation.
    """

    human_differences: list[float] = []
    metric_differences: list[float] = []
    if screen_based:
        for (item_id, system_a, system_b), (
            human_a,
            human_b,
        ) in sorted(evidence.co_screen_scores.items()):
            _append_ordered_differences(
                human_differences,
                metric_differences,
                human_a,
                human_b,
                predictions[item_id][system_a],
                predictions[item_id][system_b],
            )
    else:
        for item_id, human_scores in sorted(evidence.segment_scores.items()):
            for system_a, system_b in itertools.combinations(sorted(human_scores), 2):
                _append_ordered_differences(
                    human_differences,
                    metric_differences,
                    human_scores[system_a],
                    human_scores[system_b],
                    predictions[item_id][system_a],
                    predictions[item_id][system_b],
                )
    return (
        pearson_correlation(human_differences, metric_differences),
        len(human_differences),
    )


def _permutation_signs(
    segment_count: int,
    num_permutations: int,
    seed: int,
) -> np.ndarray:
    """Reproduce mt-metrics-eval's deterministic float32 swap matrix."""

    if segment_count <= 0:
        raise ValueError("SPA system comparisons require at least one segment")
    if num_permutations <= 0:
        raise ValueError("SPA num_permutations must be positive")
    rng = np.random.default_rng(seed)
    signs = rng.random(
        size=(num_permutations, segment_count), dtype=np.float32
    )
    np.rint(signs, out=signs, casting="same_kind")
    signs *= 2.0
    signs -= 1.0
    return signs


def _pairwise_p_values(
    score_matrix: np.ndarray,
    signs: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Port mt-metrics-eval PCE pairwise p-values for one dense matrix."""

    scores = np.asarray(score_matrix, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("SPA score input must be a systems-by-segments matrix")
    if signs.shape[1] != scores.shape[1]:
        raise ValueError("SPA permutation width differs from segment count")
    system_scores = np.sum(scores, axis=1)
    partial = np.matmul(signs, scores.T)
    values: dict[tuple[int, int], float] = {}
    for first in range(scores.shape[0]):
        for second in range(first + 1, scores.shape[0]):
            null_difference = partial[:, first] - partial[:, second]
            test_difference = system_scores[first] - system_scores[second]
            values[(first, second)] = float(
                np.sum(null_difference >= test_difference) / signs.shape[0]
            )
    return values


def prepare_spa_reference(
    evidence: HumanEvidence,
    num_permutations: int = DEFAULT_SPA_PERMUTATIONS,
    seed: int = DEFAULT_SPA_SEED,
) -> SpaReference:
    """Precompute all human-side SPA permutation tests for one direction."""

    # Segment-based SPA uses the common annotated-item support separately for
    # every system pair. This remains well-defined when human coverage is
    # incomplete, while retaining all items available to that pair.
    segment_signs = _permutation_signs(len(evidence.items), num_permutations, seed)
    segment_human_pvalues: dict[tuple[int, int], float] = {}
    segment_pairs: list[SegmentSpaReferencePair] = []
    signs_by_width: dict[int, np.ndarray] = {}
    for first_index, (system_a, system_b) in enumerate(
        itertools.combinations(evidence.systems, 2)
    ):
        item_ids = tuple(
            item
            for item in evidence.items
            if system_a in evidence.segment_scores.get(item, {})
            and system_b in evidence.segment_scores.get(item, {})
        )
        if not item_ids:
            continue
        width = len(item_ids)
        signs = signs_by_width.get(width)
        if signs is None:
            signs = _permutation_signs(width, num_permutations, seed)
            signs_by_width[width] = signs
        pair_human = np.asarray(
            [
                [evidence.segment_scores[item][system_a] for item in item_ids],
                [evidence.segment_scores[item][system_b] for item in item_ids],
            ]
        )
        segment_pairs.append(
            SegmentSpaReferencePair(
                system_a,
                system_b,
                item_ids,
                signs,
                _pairwise_p_values(pair_human, signs)[(0, 1)],
            )
        )

    by_system_pair: dict[
        tuple[str, str], list[tuple[str, float, float]]
    ] = defaultdict(list)
    for (item_id, system_a, system_b), (
        score_a,
        score_b,
    ) in evidence.co_screen_scores.items():
        by_system_pair[(system_a, system_b)].append((item_id, score_a, score_b))

    signs_by_width = {}
    screen_pairs: list[ScreenSpaReferencePair] = []
    for system_a, system_b in itertools.combinations(evidence.systems, 2):
        observations = sorted(by_system_pair.get((system_a, system_b), ()))
        if not observations:
            continue
        width = len(observations)
        signs = signs_by_width.get(width)
        if signs is None:
            signs = _permutation_signs(width, num_permutations, seed)
            signs_by_width[width] = signs
        pair_human = np.asarray(
            [
                [observation[1] for observation in observations],
                [observation[2] for observation in observations],
            ]
        )
        screen_pairs.append(
            ScreenSpaReferencePair(
                system_a,
                system_b,
                tuple(observation[0] for observation in observations),
                signs,
                _pairwise_p_values(pair_human, signs)[(0, 1)],
            )
        )
    if not screen_pairs and not segment_pairs:
        raise ValueError("No system pair has co-screen human evidence")
    return SpaReference(
        evidence.systems,
        evidence.items,
        segment_signs,
        segment_human_pvalues,
        tuple(screen_pairs),
        tuple(segment_pairs),
    )


def _segment_spa(
    evidence: HumanEvidence,
    predictions: Predictions,
    num_permutations: int,
    seed: int,
    reference: SpaReference | None = None,
) -> tuple[float, int]:
    """Compute standard segment-based SPA from all item-system means."""

    reference = reference or prepare_spa_reference(
        evidence, num_permutations, seed
    )
    agreements = []
    for pair in reference.segment_pairs:
        metric = np.asarray(
            [
                [predictions[item][pair.system_a] for item in pair.item_ids],
                [predictions[item][pair.system_b] for item in pair.item_ids],
            ]
        )
        metric_pvalue = _pairwise_p_values(metric, pair.signs)[(0, 1)]
        agreements.append(1.0 - abs(pair.human_pvalue - metric_pvalue))
    if not agreements:
        raise ValueError("No system pair has common segment human evidence")
    return math.fsum(agreements) / len(agreements), len(agreements)


def _screen_spa(
    evidence: HumanEvidence,
    predictions: Predictions,
    num_permutations: int,
    seed: int,
    reference: SpaReference | None = None,
) -> tuple[float, int]:
    """Compute SPA using only pair-specific co-screen item evidence."""

    reference = reference or prepare_spa_reference(
        evidence, num_permutations, seed
    )
    agreements: list[float] = []
    for pair in reference.screen_pairs:
        metric = np.asarray(
            [
                [predictions[item][pair.system_a] for item in pair.item_ids],
                [predictions[item][pair.system_b] for item in pair.item_ids],
            ]
        )
        metric_pvalue = _pairwise_p_values(metric, pair.signs)[(0, 1)]
        agreements.append(1.0 - abs(pair.human_pvalue - metric_pvalue))
    if not agreements:
        raise ValueError("No system pair has co-screen human evidence")
    return math.fsum(agreements) / len(agreements), len(agreements)


def score_measure(
    evidence: HumanEvidence,
    predictions: Predictions,
    measure: str,
    *,
    spa_permutations: int = DEFAULT_SPA_PERMUTATIONS,
    spa_seed: int = DEFAULT_SPA_SEED,
    spa_reference: SpaReference | None = None,
) -> tuple[float, int]:
    """Dispatch one of the four named WMT26 Task 2 measures."""

    if measure == GROUP_BY_SCREEN_PDP:
        return score_pdp(evidence, predictions, screen_based=True)
    if measure == GROUP_BY_SEGMENT_PDP:
        return score_pdp(evidence, predictions, screen_based=False)
    if measure == SCREEN_BASED_SPA:
        return _screen_spa(
            evidence,
            predictions,
            spa_permutations,
            spa_seed,
            spa_reference,
        )
    if measure == SEGMENT_BASED_SPA:
        return _segment_spa(
            evidence,
            predictions,
            spa_permutations,
            spa_seed,
            spa_reference,
        )
    raise ValueError(f"Unknown Task 2 measure: {measure!r}")
