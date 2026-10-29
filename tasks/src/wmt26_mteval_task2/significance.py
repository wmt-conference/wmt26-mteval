"""Joint source-block Task 2 permutation statistics with stable float64 SPA.

All supported observations stay fixed. A source's exchange bit is shared by
every MT translation, direction occurrence, screen comparison and meta-metric.
This module does not alter the legacy scorer or publish paper results.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from wmt26_mteval_task2.metrics import Predictions, _permutation_signs
from wmt26_mteval_task2.models import HumanEvidence
from wmt26_mteval_task2.significance_numerics import (
    permutation_counts, standardize_predictions, tie_bound,
)

METHOD_VERSION = "joint-source-float64-v1"


def source_key(item: str) -> tuple[str, ...]:
    """Identify an official source occurrence without merging equal text."""
    fields = item.split("_###_")
    if len(fields) != 6 or fields[0] != "official" or not all(fields):
        raise ValueError(f"Invalid official item identity: {item!r}")
    return tuple(fields[:2] + fields[3:])


def readonly(value: np.ndarray) -> np.ndarray:
    """Protect prepared arrays from accidental mutation inside workers."""
    value.flags.writeable = False
    return value


@dataclass(frozen=True)
class PairWidth:
    """MT pairs sharing a support width, not necessarily the same item set."""

    sources: np.ndarray  # common-item position, MT pair -> global source index
    differences: np.ndarray  # evaluator, common-item position, MT pair
    flipped: np.ndarray  # inner permutation, common-item position
    human_counts: np.ndarray
    metric_counts: np.ndarray


@dataclass(frozen=True)
class PreparedDirection:
    """Immutable exact-support inputs for one direction and every evaluator."""

    name: str
    sources: np.ndarray
    pdp_numerators: np.ndarray  # evaluator, item
    pdp_norms: np.ndarray
    human_norm: float
    widths: tuple[PairWidth, ...]
    spa_permutations: int
    prediction_means: np.ndarray
    prediction_sds: np.ndarray


def prepare_direction(
    evidence: HumanEvidence,
    predictions: list[Predictions],
    source_lookup: dict[tuple[str, ...], int],
    *,
    spa_permutations: int = 1000,
    spa_seed: int = 4,
    ordered_pdp_differences: bool = False,
) -> PreparedDirection:
    """Prepare global PDP sufficient sums and same-width SPA pair matrices.

Differences are computed on raw scores then divided by one evaluator/direction
standard deviation. This is algebraically identical to differencing z-scores,
without unnecessary centering cancellation. No per-source normalization occurs.
"""
    # The blindset can also contain items with no retained human observations.
    # Only human-supported items enter the statistic or its exchange blocks.
    items = tuple(sorted(evidence.segment_scores))
    item_lookup = {item: i for i, item in enumerate(items)}
    sources = np.array([source_lookup[source_key(item)] for item in items], dtype=np.int64)
    scales = []
    means = []
    for prediction in predictions:
        if set(prediction) != set(evidence.segment_scores) or any(
            set(prediction[i]) != set(evidence.segment_scores[i]) for i in items
        ):
            raise ValueError("Prediction support must equal retained human support")
        _, mean, sd = standardize_predictions(prediction)
        means.append(mean)
        scales.append(sd)
    scales = np.array(scales)
    numerators = np.zeros((len(predictions), len(items)))
    norms = np.zeros_like(numerators)
    human_norm = []
    for (item, a, b), (ha, hb) in sorted(evidence.co_screen_scores.items()):
        # PDP's Pearson input contains both ordered non-self differences.  The
        # primary production checkpoint predates this explicit option and is
        # kept byte-for-byte compatible by retaining the historical default.
        orientations = ((a, b, ha, hb), (b, a, hb, ha)) if ordered_pdp_differences else ((a, b, ha, hb),)
        for first, second, first_human, second_human in orientations:
            h = first_human - second_human
            m = np.array([p[item][first] - p[item][second] for p in predictions]) / scales
            numerators[:, item_lookup[item]] += h * m
            norms[:, item_lookup[item]] += m * m
            human_norm.append(h * h)
    pairs_by_width = defaultdict(list)
    for a, b in itertools.combinations(evidence.systems, 2):
        common = tuple(i for i in items if a in evidence.segment_scores[i] and b in evidence.segment_scores[i])
        if common:
            pairs_by_width[len(common)].append((a, b, common))
    if not human_norm or not pairs_by_width:
        raise ValueError("Both screen PDP and source SPA require nonempty support")
    widths = []
    for width, pairs in sorted(pairs_by_width.items()):
        flipped = np.asarray((1 - _permutation_signs(width, spa_permutations, spa_seed)) / 2,
                             dtype=np.float64)
        sources_by_pair = np.array([[source_lookup[source_key(i)] for i in common]
                                    for _, _, common in pairs], dtype=np.int64).T
        human = np.array([[evidence.segment_scores[i][a] - evidence.segment_scores[i][b]
                           for i in common] for a, b, common in pairs], dtype=np.float64).T
        differences = np.array([
            [[(p[i][a] - p[i][b]) / scale for i in common] for a, b, common in pairs]
            for p, scale in zip(predictions, scales)
        ], dtype=np.float64).transpose(0, 2, 1).copy()
        metric_counts = np.array([permutation_counts(d, flipped) for d in differences])
        widths.append(PairWidth(readonly(sources_by_pair), readonly(differences), readonly(flipped),
                                readonly(permutation_counts(human, flipped)), readonly(metric_counts)))
    return PreparedDirection(evidence.direction, readonly(sources), readonly(numerators),
        readonly(norms), math.fsum(human_norm), tuple(widths), spa_permutations,
        readonly(np.array(means)), readonly(scales))


def pdp_from_sums(numerators: np.ndarray, norms: np.ndarray, human_norm: float) -> np.ndarray:
    """Return the global ordered-difference Pearson, including constant=zero."""
    if np.any(norms < 0) or human_norm < 0:
        raise ValueError("Negative squared PDP norm")
    denominator = np.sqrt(norms * human_norm)
    result = np.divide(numerators, denominator, out=np.zeros_like(numerators), where=denominator > 0)
    if np.any(np.abs(result) > 1 + 1e-12):
        raise ValueError("Invalid PDP correlation")
    return np.clip(result, -1, 1)


def observed_components(directions: tuple[PreparedDirection, ...]) -> np.ndarray:
    """Return evaluator-by-direction-by-component (PDP, SPA) point estimates."""
    results = []
    for direction in directions:
        pdp = pdp_from_sums(direction.pdp_numerators.sum(axis=1), direction.pdp_norms.sum(axis=1),
                            direction.human_norm)
        error = sum(np.abs(w.metric_counts - w.human_counts).sum(axis=1) for w in direction.widths)
        pairs = sum(len(w.human_counts) for w in direction.widths)
        spa = 1 - error / (direction.spa_permutations * pairs)
        results.append(np.stack((pdp, spa), axis=-1))
    return np.stack(results, axis=1)


def permutation_masks(seed: int, pair: tuple[str, str], start: int, count: int, sources: int) -> np.ndarray:
    """Generate draw-indexed masks independent of chunk size and worker count."""
    if seed < 0 or start < 0 or count < 1 or sources < 1 or pair[0] >= pair[1]:
        raise ValueError("Invalid deterministic permutation request")
    base = hashlib.sha256(f"{METHOD_VERSION}\0{seed}\0{pair[0]}\0{pair[1]}".encode()).digest()
    rows = []
    for draw in range(start, start + count):
        entropy = int.from_bytes(hashlib.sha256(base + draw.to_bytes(8, "little")).digest(), "little")
        rows.append(np.random.default_rng(entropy).integers(0, 2, size=sources, dtype=np.int8).astype(bool))
    return np.array(rows)


def evaluate_exchanges(
    directions: tuple[PreparedDirection, ...], a: int, b: int, masks: np.ndarray,
) -> np.ndarray:
    """Return draw/hybrid/component means, preserving the original arithmetic."""
    return aggregate_exchanges(evaluate_exchanges_by_direction(directions, a, b, masks))


def aggregate_exchanges(values: np.ndarray) -> np.ndarray:
    """Average the direction axis in the same order as the original engine."""
    values = np.asarray(values)
    if values.ndim != 4 or values.shape[1] != 2 or values.shape[3] != 2 or not values.shape[2]:
        raise ValueError("Expected draw/hybrid/direction/component statistics")
    output = np.zeros((len(values), 2, 2), dtype=np.float64)
    for index in range(values.shape[2]):
        output += values[:, :, index, :] / values.shape[2]
    return output


def evaluate_exchanges_by_direction(
    directions: tuple[PreparedDirection, ...], a: int, b: int, masks: np.ndarray,
) -> np.ndarray:
    """Score once, retaining draw/hybrid/direction/component statistics.

Each inner matrix product evaluates all MT pairs with equal support width.
Their source-index matrix still selects each pair's distinct actual support.
The same masks apply to every direction and component. No pair weighting changes.
"""
    masks = np.asarray(masks)
    if not directions or masks.ndim != 2 or not masks.shape[0] or not np.isin(masks, (0, 1)).all():
        raise ValueError("Expected a nonempty binary draw-by-source matrix")
    masks = masks.astype(bool)
    output = np.zeros((len(masks), 2, len(directions), 2), dtype=np.float64)
    for index, direction in enumerate(directions):
        switch = masks[:, direction.sources]
        for side, (first, second) in enumerate(((a, b), (b, a))):
            num = np.where(switch, direction.pdp_numerators[second], direction.pdp_numerators[first]).sum(axis=1)
            norm = np.where(switch, direction.pdp_norms[second], direction.pdp_norms[first]).sum(axis=1)
            output[:, side, index, 0] = pdp_from_sums(num, norm, direction.human_norm)
        errors = np.zeros((len(masks), 2), dtype=np.int64)
        pairs = 0
        for group in direction.widths:
            switch = masks[:, group.sources]  # draw, common item position, pair
            pair_count = len(group.human_counts)
            width = group.differences.shape[1]
            for side, (first, second) in enumerate(((a, b), (b, a))):
                hybrid = np.where(switch, group.differences[second], group.differences[first])
                columns = hybrid.transpose(1, 0, 2).reshape(width, -1)
                # Masks were validated at preparation, avoiding a large repeated scan.
                null_sums = group.flipped @ columns
                counts = np.count_nonzero(null_sums <= tie_bound(columns), axis=0).reshape(len(masks), pair_count)
                errors[:, side] += np.abs(counts - group.human_counts).sum(axis=1)
            pairs += pair_count
        output[:, :, index, 1] = 1 - errors / (direction.spa_permutations * pairs)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite permuted statistic")
    return output


def two_sided_pvalue(null_differences: np.ndarray, observed_difference: float) -> tuple[float, int]:
    """Use the inclusive absolute tail with the Monte Carlo +1 correction."""
    values = np.asarray(null_differences, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or not math.isfinite(observed_difference):
        raise ValueError("Finite observed and null differences required")
    magnitude = abs(observed_difference)
    absolute = np.abs(values)
    extreme = int(np.count_nonzero((absolute >= magnitude) | np.isclose(absolute, magnitude, rtol=1e-12, atol=1e-15)))
    return (extreme + 1) / (len(values) + 1), extreme


def one_sided_pvalue(null_differences: np.ndarray, observed_difference: float) -> tuple[float, int]:
    """Count the upper tail for a specified evaluator order, with +1 correction.

When ordered by the observed ranking, this is a directional sensitivity result,
not evidence that the comparison direction was specified independently of data.
"""
    values = np.asarray(null_differences, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or not math.isfinite(observed_difference):
        raise ValueError("Finite observed and null differences required")
    extreme = int(np.count_nonzero((values >= observed_difference) |
        np.isclose(values, observed_difference, rtol=1e-12, atol=1e-15)))
    return (extreme + 1) / (len(values) + 1), extreme


def assign_clusters(pvalues: np.ndarray, alpha: float = .05) -> list[int]:
    """Reproduce WMT's greedy contiguous clusters on a score-ordered matrix."""
    pvalues = np.asarray(pvalues)
    if pvalues.ndim != 2 or pvalues.shape[0] != pvalues.shape[1] or not 0 < alpha < 1:
        raise ValueError("Invalid p-value matrix or alpha")
    upper = pvalues[np.triu_indices(len(pvalues), 1)]
    if not np.isfinite(upper).all() or np.any((upper < 0) | (upper > 1)):
        raise ValueError("Incomplete or invalid pairwise p-value matrix")
    start, cluster, labels = 0, 1, []
    for index in range(len(pvalues)):
        if np.any(pvalues[start:index, index] <= alpha):
            start, cluster = index, cluster + 1
        labels.append(cluster)
    return labels
