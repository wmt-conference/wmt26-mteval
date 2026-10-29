"""Stable numerical primitives for the Task 2 significance audit.

The published float32 scorer is deliberately unchanged. These functions use
the same inner permutation signs, but sum paired differences in float64 and
include numerically tied permutations. Their effect must be audited first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from wmt26_mteval_task2.metrics import Predictions, SpaReference


def tie_bound(differences: np.ndarray) -> np.ndarray:
    """Bound float64 reduction roundoff relative to each column's L1 norm.

The width-dependent bound is eight times a first-order summation error bound.
It has no fixed absolute floor, so changing the score unit scales the bound
with the statistic. It handles arithmetic roundoff, not substantive score ties.
"""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not np.isfinite(values).all():
        raise ValueError("Differences must be a finite nonempty items-by-pairs matrix")
    return 8 * np.finfo(np.float64).eps * (values.shape[0] + 2) * np.abs(values).sum(axis=0)


def permutation_counts(differences: np.ndarray, flipped: np.ndarray) -> np.ndarray:
    """Count inner permutations at least as favorable as the observed scores.

For d = score_A - score_B, null_sum >= observed_sum iff the sum of d on
flipped items is <= 0. This avoids subtracting two large system-score sums.
Columns may represent several MT pairs with the same support width. Each
column's rows must retain that pair's sorted common-item order.
"""
    differences = np.asarray(differences, dtype=np.float64)
    flipped = np.asarray(flipped, dtype=np.float64)
    bound = tie_bound(differences)
    if flipped.ndim != 2 or not flipped.shape[0] or flipped.shape[1] != differences.shape[0]:
        raise ValueError("Permutation shape does not match paired support")
    if not np.isin(flipped, (0, 1)).all():
        raise ValueError("Permutation mask must contain only zero or one")
    sums = flipped @ differences
    return np.count_nonzero(sums <= bound, axis=0)


def stable_spa_counts(reference: SpaReference, predictions: Predictions) -> np.ndarray:
    """Return float64 inner p-value numerators for the existing SPA pair order."""
    counts = []
    for pair in reference.segment_pairs:
        differences = np.array([
            predictions[item][pair.system_a] - predictions[item][pair.system_b]
            for item in pair.item_ids
        ], dtype=np.float64)[:, None]
        counts.append(int(permutation_counts(differences, (1.0 - pair.signs) / 2.0)[0]))
    if not counts:
        raise ValueError("No SPA pairs with common support")
    return np.asarray(counts, dtype=np.int64)


def spa_from_counts(human: np.ndarray, metric: np.ndarray, permutations: int) -> float:
    """Average SPA contributions using exact integer p-value numerators."""
    human, metric = np.asarray(human), np.asarray(metric)
    if human.shape != metric.shape or human.ndim != 1 or not human.size or permutations < 1:
        raise ValueError("Invalid SPA count shapes or permutation count")
    if any(np.any((values < 0) | (values > permutations)) for values in (human, metric)):
        raise ValueError("SPA counts outside permutation range")
    return 1.0 - int(np.abs(human - metric).sum()) / (permutations * len(human))


def standardize_predictions(predictions: Predictions) -> tuple[dict[str, dict[str, float]], float, float]:
    """Standardize once over distinct supported translations, never per source."""
    values = [predictions[item][system] for item in sorted(predictions)
              for system in sorted(predictions[item])]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Nonempty finite predictions required")
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
    if not math.isfinite(sd) or sd <= 0:
        raise ValueError("Constant or numerically degenerate evaluator predictions")
    return ({item: {system: (value - mean) / sd for system, value in scores.items()}
             for item, scores in predictions.items()}, mean, sd)
