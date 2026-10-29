"""PERM-BOTH adapted to MPP and greedy significance clusters.

The swap unit is one item-system output, including all its human assessments.
The test and clustering follow google-research/mt-metrics-eval's stats.py and
AssignRanks in data.py; no correlation-specific score standardization is used.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Iterator, Mapping

import numpy as np

from wmt26_mteval_task1.constants import mpp_metric_options
from wmt26_mteval_task1.models import (
    BatchScoreResult,
    PermutationDirection,
    SubmissionScore,
)
from wmt26_mteval_task1.mpp_cache import _file_sha256
from wmt26_mteval_task1.rankings import (
    EntryKey,
    RankingScope,
    entry_key,
    ranking_scopes,
    scope_score,
)
from wmt26_mteval_task1.significance_cache import CHUNK_SIZE, SignificanceCache, fingerprint
from wmt26_mteval_task1.significance_parallel import DirectionWork, completed_pair_work
from wmt26_mteval_task1.parquet_source import current_source

METHOD_VERSION = 2


def _significance_input_sha256(path: Path) -> str:
    """Preserve data-equivalent permutation seeds across JSONL and parquet."""

    source = current_source()
    if source is not None and source.contains(path):
        return source.logical_digest(path)
    return _file_sha256(path)


@dataclass(frozen=True)
class SignificanceConfig:
    """Fixed-run permutation settings; alpha only affects reporting decisions."""

    permutations: int = 10000
    seed: int = 4
    alpha: float = 0.05
    workers: int = 1
    alternative: str = "two-sided"

    def __post_init__(self) -> None:
        """Reject invalid alternatives, worker counts, draws, seeds, and thresholds."""
        if self.alternative not in {"one-sided", "two-sided"}:
            raise ValueError("significance alternative must be one-sided or two-sided")
        if not isinstance(self.workers, int) or isinstance(self.workers, bool) or self.workers < 1:
            raise ValueError("significance workers must be a positive integer")
        if self.permutations <= 0 or self.seed < 0 or not 0 < self.alpha < 1:
            raise ValueError(
                "significance requires positive draws, nonnegative seed, and 0 < alpha < 1"
            )


@dataclass(frozen=True)
class PairSignificance:
    """One score-oriented evaluator comparison within an explicit ranking scope."""

    scope: str
    better: EntryKey
    worse: EntryKey
    observed_difference: float
    p_value: float
    draws: int
    significant: bool


@dataclass(frozen=True)
class SignificanceResult:
    """Complete pairwise tests, scope-specific cluster labels, and run provenance."""

    pairs: tuple[PairSignificance, ...]
    clusters: Mapping[str, Mapping[EntryKey, int]]
    manifest: dict


def f1_from_totals(totals: np.ndarray) -> np.ndarray:
    """Convert arrays of four MPP totals to F1 using official empty-side rules."""
    totals = np.asarray(totals, dtype=np.float64)
    precision = np.divide(
        totals[..., 0], totals[..., 1],
        out=np.ones(totals.shape[:-1]), where=totals[..., 1] > 0,
    )
    recall = np.divide(
        totals[..., 2], totals[..., 3],
        out=np.ones(totals.shape[:-1]), where=totals[..., 3] > 0,
    )
    return np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=precision + recall > 0,
    )


def permuted_differences(a: np.ndarray, b: np.ndarray, swaps: np.ndarray) -> np.ndarray:
    """Return MPP F1(A)-F1(B) for each supplied item-system swap mask.

    A and B contain weighted four-component rows on identical support. Each
    mask moves an entire row; references remain attached to their item-system.
    """
    sum_a, sum_b = a.sum(axis=0), b.sum(axis=0)
    lower_bounds = np.minimum(a, b).sum(axis=0)
    error_margin = 8 * len(a) * np.finfo(np.float64).eps * (sum_a + sum_b)
    if np.all(lower_bounds > error_margin):
        # All possible mixtures have positive totals, safely away from zero.
        # Moving a row transfers B-A to A and its opposite to B. This uses one
        # matrix product instead of four while preserving the MPP statistic.
        transfer = swaps @ (b - a)
        total_a = sum_a + transfer
        total_b = sum_b - transfer
    else:
        # Direct nonnegative sums preserve exact zeros and the empty-side rules
        # when subtraction could otherwise introduce cancellation residuals.
        keep = 1 - swaps
        total_a = keep @ a + swaps @ b
        total_b = swaps @ a + keep @ b
    return f1_from_totals(total_a) - f1_from_totals(total_b)


def permutation_p_value(
    null_differences: np.ndarray, observed: float, alternative: str = "two-sided",
) -> float:
    """Return the inclusive signed or absolute Monte Carlo tail proportion.

    A tolerance of 1e-12 relative / 1e-15 absolute treats numerical roundoff
    around the observed statistic as equality, including identity permutations.
    """
    if alternative not in {"one-sided", "two-sided"}:
        raise ValueError("significance alternative must be one-sided or two-sided")
    if (
        not len(null_differences)
        or not np.isfinite(null_differences).all()
        or not np.isfinite(observed)
    ):
        raise ValueError("significance requires finite observed and permuted differences")
    if alternative == "two-sided":
        null_differences, observed = np.abs(null_differences), abs(observed)
    equal = np.isclose(null_differences, observed, rtol=1e-12, atol=1e-15)
    return float(np.mean((null_differences >= observed) | equal))


def assign_clusters(p_values: np.ndarray, alpha: float) -> list[int]:
    """Assign contiguous clusters from a score-ordered upper-triangle p matrix."""
    if p_values.ndim != 2 or p_values.shape[0] != p_values.shape[1]:
        raise ValueError("cluster assignment requires a square p-value matrix")
    start, cluster, labels = 0, 1, []
    for index in range(len(p_values)):
        values = p_values[:index, index]
        if not np.isfinite(values).all():
            raise ValueError("cannot assign clusters from incomplete pairwise tests")
        if np.any(values[start:] <= alpha):
            cluster += 1
            start = index
        labels.append(cluster)
    return labels


def validate_support(a: PermutationDirection, b: PermutationDirection) -> None:
    """Require identical nonempty human support and finite four-component rows."""
    if not a.support or a.support != b.support:
        raise ValueError("evaluators have different item-system/ground-truth support")
    for value in (a, b):
        if (
            value.components.shape != (len(value.support), 4)
            or not np.isfinite(value.components).all()
            or np.any(value.components < 0)
        ):
            raise ValueError("invalid MPP permutation components")


def _register_direction(
    a: PermutationDirection, b: PermutationDirection, config: SignificanceConfig,
    cache: SignificanceCache, metadata: dict, *, recompute: bool,
) -> str:
    """Validate aligned inputs and identify a stream independently of worker count."""
    validate_support(a, b)
    component_hashes = [
        hashlib.sha256(np.asarray(value.components, dtype="<f8").tobytes()).hexdigest()
        for value in (a, b)
    ]
    return cache.register(
        {
            **metadata,
            "method_version": METHOD_VERSION,
            "seed": config.seed,
            "rng": "PCG64",
            "numpy_version": np.__version__,
            "support_sha256": fingerprint(a.support),
            "components_sha256": component_hashes,
        },
        recompute=recompute,
    )


def permutation_chunk(a: np.ndarray, b: np.ndarray, key: str, index: int) -> np.ndarray:
    """Return one deterministically seeded chunk using only component matrices."""
    chunk_seed = int(fingerprint([key, index]), 16)
    generator = np.random.Generator(np.random.PCG64(chunk_seed))
    swaps = generator.integers(
        0, 2, size=(CHUNK_SIZE, len(a)), dtype=np.int8,
    ).astype(np.float64)
    return permuted_differences(a, b, swaps)


def direction_draws(
    a: PermutationDirection,
    b: PermutationDirection,
    config: SignificanceConfig,
    cache: SignificanceCache,
    metadata: dict,
    *,
    recompute: bool = False,
) -> tuple[np.ndarray, str]:
    """Load or extend a deterministic pair-direction stream and return its prefix."""
    key = _register_direction(a, b, config, cache, metadata, recompute=recompute)
    chunks = []
    for index in range((config.permutations + CHUNK_SIZE - 1) // CHUNK_SIZE):
        draws = cache.load(key, index)
        if draws is None:
            draws = permutation_chunk(a.components, b.components, key, index)
            cache.store(key, index, draws)
        chunks.append(draws)
    return np.concatenate(chunks)[:config.permutations], key


@dataclass
class PreparedPair:
    """Parent-owned scopes, stream metadata, and cached chunks for one pair."""

    a: EntryKey
    b: EntryKey
    scopes: list[RankingScope]
    streams: list[dict]
    chunks: dict[str, dict[int, np.ndarray]]


def _prepare_pair_work(
    scores: Mapping[EntryKey, SubmissionScore],
    pairs_to_scopes: Mapping[tuple[EntryKey, EntryKey], list[RankingScope]],
    config: SignificanceConfig,
    cache: SignificanceCache,
    inputs: dict,
    predictions: Mapping[EntryKey, str],
    matching_algorithm: str,
    *,
    recompute: bool,
) -> Iterator[tuple[tuple[DirectionWork, ...], PreparedPair]]:
    """Lazily prepare bounded pair jobs; all validation and cache reads stay here."""
    chunk_count = (config.permutations + CHUNK_SIZE - 1) // CHUNK_SIZE
    for (key_a, key_b), pair_scopes in sorted(pairs_to_scopes.items()):
        a, b = scores[key_a], scores[key_b]
        directions = sorted({direction for scope in pair_scopes for direction in scope.directions})
        streams, jobs, cached = [], [], {}
        for direction in directions:
            if direction not in a.permutation_inputs or direction not in b.permutation_inputs:
                raise ValueError(f"missing significance inputs for {key_a}/{key_b}/{direction}")
            array_a, array_b = a.permutation_inputs[direction], b.permutation_inputs[direction]
            validate_support(array_a, array_b)
            for score, array in ((a, array_a), (b, array_b)):
                reconstructed = float(f1_from_totals(array.components.sum(axis=0)))
                if not np.isclose(
                    reconstructed, scope_score(score, (direction,)), rtol=1e-12, atol=1e-15,
                ):
                    raise ValueError("permutation components disagree with reported MPP F1")
            key = _register_direction(array_a, array_b, config, cache, {
                "evaluators": [key_a, key_b], "direction": direction,
                "test_set_sha256": inputs["test_set"]["sha256"],
                "references_sha256": inputs["references"]["sha256"],
                "predictions_sha256": [predictions[key_a], predictions[key_b]],
                "metric_options": mpp_metric_options(matching_algorithm),
            }, recompute=recompute)
            chunks, missing = {}, []
            for index in range(chunk_count):
                draws = cache.load(key, index)
                if draws is None:
                    missing.append(index)
                else:
                    chunks[index] = draws
            cached[key] = chunks
            if missing:
                jobs.append(DirectionWork(key, array_a.components, array_b.components, tuple(missing)))
            streams.append({
                "evaluators": [key_a, key_b], "direction": direction, "fingerprint": key,
                "observed_difference": scope_score(a, (direction,)) - scope_score(b, (direction,)),
            })
        yield tuple(jobs), PreparedPair(key_a, key_b, pair_scopes, streams, cached)


def compute_significance(
    result: BatchScoreResult,
    config: SignificanceConfig,
    cache_path: Path,
    *,
    test_set_path: Path,
    ground_truth_path: Path,
    recompute: bool = False,
    paper_only: bool = False,
) -> SignificanceResult:
    """Compute ranking tests with bounded pair workers and parent-owned caching.

    Each worker computes pure permutation chunks. Only the parent validates,
    reads/writes SQLite, and aggregates results. Worker count affects neither
    fingerprints nor seeds; every scope equally weights its direction draws.
    ``paper_only`` omits the unpublished all-direction aggregate and returns
    tests for the priority aggregate and individual directions only.
    """
    scopes = tuple(
        scope for scope in ranking_scopes(result)
        if not paper_only or scope.name != "all_directions"
    )
    scores = {entry_key(score.descriptor): score for score in result.scores}
    if len(scores) != len(result.scores):
        raise ValueError("duplicate evaluator identities in significance cohort")
    inputs = {
        "test_set": {"path": str(test_set_path), "sha256": _significance_input_sha256(test_set_path)},
        "references": {
            "path": str(ground_truth_path), "sha256": _significance_input_sha256(ground_truth_path),
        },
    }
    predictions = {
        key: _significance_input_sha256(score.descriptor.predictions_path)
        for key, score in scores.items()
    }
    pairs_to_scopes: dict[tuple[EntryKey, EntryKey], list[RankingScope]] = {}
    matrices = {
        scope.name: np.full((len(scope.entries), len(scope.entries)), np.nan)
        for scope in scopes
    }
    positions = {
        scope.name: {
            entry_key(score.descriptor): i for i, score in enumerate(scope.entries)
        }
        for scope in scopes
    }
    for scope in scopes:
        for score_a, score_b in itertools.combinations(scope.entries, 2):
            pair = tuple(sorted((entry_key(score_a.descriptor), entry_key(score_b.descriptor))))
            pairs_to_scopes.setdefault(pair, []).append(scope)

    comparisons, streams = [], []
    logging.info(
        "Significance: %d evaluator pairs, %d draws per direction, %d workers, %s test",
        len(pairs_to_scopes), config.permutations, config.workers, config.alternative,
    )
    last_log = monotonic()
    source = current_source()
    with SignificanceCache(
        cache_path,
        data_fingerprint=source.parquet_sha256 if source is not None else None,
    ) as cache:
        jobs = _prepare_pair_work(
            scores, pairs_to_scopes, config, cache, inputs, predictions,
            result.matching_algorithm, recompute=recompute,
        )
        for number, (pair, computed) in enumerate(
            completed_pair_work(jobs, config.workers), 1,
        ):
            for key, index, draws in computed:
                cache.store(key, index, draws)
                pair.chunks[key][index] = draws
            draws_by_direction = {
                stream["direction"]: np.concatenate([
                    draws for _, draws in sorted(pair.chunks[stream["fingerprint"]].items())
                ])[:config.permutations]
                for stream in pair.streams
            }
            streams.extend(pair.streams)
            key_a, key_b = pair.a, pair.b
            a, b = scores[key_a], scores[key_b]
            pair_scopes = pair.scopes
            for scope in pair_scopes:
                i, j = positions[scope.name][key_a], positions[scope.name][key_b]
                sign = 1 if i < j else -1
                delta = sign * (
                    scope_score(a, scope.directions) - scope_score(b, scope.directions)
                )
                null = sign * np.mean(
                    [draws_by_direction[direction] for direction in scope.directions],
                    axis=0,
                )
                p_value = permutation_p_value(null, delta, config.alternative)
                matrices[scope.name][min(i, j), max(i, j)] = p_value
                comparisons.append(PairSignificance(
                    scope.name,
                    key_a if sign == 1 else key_b,
                    key_b if sign == 1 else key_a,
                    delta, p_value, config.permutations, p_value <= config.alpha,
                ))
            if monotonic() - last_log >= 30 or number == len(pairs_to_scopes):
                logging.info(
                    "Significance: completed %d/%d evaluator pairs",
                    number, len(pairs_to_scopes),
                )
                last_log = monotonic()
        logging.info(cache.summary())
        cache_statistics = {
            "reused_chunks": cache.reused_chunks,
            "computed_chunks": cache.computed_chunks,
        }

    clusters = {
        scope.name: dict(zip(
            (entry_key(score.descriptor) for score in scope.entries),
            assign_clusters(matrices[scope.name], config.alpha),
        ))
        for scope in scopes
    }
    comparisons.sort(key=lambda pair: (
        pair.scope, positions[pair.scope][pair.better], positions[pair.scope][pair.worse],
    ))
    streams.sort(key=lambda stream: (stream["evaluators"], stream["direction"]))
    manifest = {
        "enabled": True,
        "status": "complete",
        "method": "perm-both-mpp",
        "method_version": METHOD_VERSION,
        "swap_unit": "item-system",
        "permutations": config.permutations,
        "workers": config.workers,
        "seed": config.seed,
        "alpha": config.alpha,
        "alternative": config.alternative,
        "p_value_formula": (
            "count(abs(null_difference) >= abs(observed_difference)) / draws"
            if config.alternative == "two-sided"
            else "count(null_difference >= observed_difference) / draws"
        ),
        "equality_tolerance": {"rtol": 1e-12, "atol": 1e-15},
        "early_stopping": False,
        "multiple_testing_correction": None,
        "matching_algorithm": result.matching_algorithm,
        "inputs": inputs,
        "predictions": [
            {
                "entry": key,
                "path": str(scores[key].descriptor.predictions_path),
                "sha256": value,
            }
            for key, value in sorted(predictions.items())
        ],
        "scopes": [
            {
                "name": scope.name,
                "directions": scope.directions,
                "entries": [entry_key(score.descriptor) for score in scope.entries],
            }
            for scope in scopes
        ],
        "cache": {"path": str(cache_path), **cache_statistics},
        "streams": streams,
        "pairwise_test_count": len(comparisons),
    }
    return SignificanceResult(tuple(comparisons), clusters, manifest)
