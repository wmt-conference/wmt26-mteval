"""Small process-pool tests: three evaluators, two directions, 32 draws."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from scripts.score_task1_submissions import build_parser, main
from test_task1_significance import _batch
from wmt26_mteval_task1.models import DirectionScore, PermutationDirection
from wmt26_mteval_task1.significance import (
    SignificanceConfig, compute_significance, f1_from_totals,
)
from wmt26_mteval_task1.significance_parallel import DirectionWork, completed_pair_work


def _three_evaluator_batch(tmp_path):
    """Return a tiny fully supported cohort with nondegenerate component matrices."""
    batch, test_set, references, _ = _batch(tmp_path)
    directions = ("eng_Latn-deu_Latn", "eng_Latn-ind_Latn")
    support = (("first", "s", ((0, 1.),)), ("second", "s", ((0, 1.),)))
    scores = []
    for index in range(3):
        base = batch.scores[index % 2]
        inputs, direction_scores = {}, []
        for number, direction in enumerate(directions):
            components = np.array([
                [.2 + .1 * index, 1., .4, 1.],
                [.5, 1., .2 + .1 * number + .1 * index, 1.],
            ])
            totals = components.sum(axis=0)
            direction_scores.append(DirectionScore(
                direction, totals[0] / totals[1], totals[2] / totals[3],
                float(f1_from_totals(totals)), 2,
            ))
            inputs[direction] = PermutationDirection(support, components)
        scores.append(replace(
            base, descriptor=replace(base.descriptor, submission_id=str(index + 1)),
            direction_scores=tuple(direction_scores), permutation_inputs=inputs,
            coverage=replace(base.coverage, complete_directions=directions),
        ))
    return replace(batch, scores=tuple(scores), selected_directions=directions,
                   gold_directions=directions), test_set, references


def _cache_chunks(path):
    """Read cached chunk bytes in stable order for exact reproducibility checks."""
    with sqlite3.connect(path) as connection:
        return connection.execute(
            "SELECT fingerprint, chunk_index, draws FROM permutation_chunks "
            "ORDER BY fingerprint, chunk_index"
        ).fetchall()


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_matches_serial_and_reuses_cache(tmp_path, workers, monkeypatch):
    """Cold parallel draws match serial bytes; worker changes reuse every chunk."""
    import wmt26_mteval_task1.significance_parallel as parallel
    batch, test_set, references = _three_evaluator_batch(tmp_path)
    kwargs = {"test_set_path": test_set.path, "ground_truth_path": references.path}
    serial_path, parallel_path = tmp_path / "serial.sqlite3", tmp_path / "parallel.sqlite3"
    serial = compute_significance(batch, SignificanceConfig(32), serial_path, **kwargs)
    config = SignificanceConfig(32, workers=workers)
    result = compute_significance(batch, config, parallel_path, **kwargs)
    assert result.pairs == serial.pairs and result.clusters == serial.clusters
    assert result.manifest["streams"] == serial.manifest["streams"]
    assert result.manifest["workers"] == workers
    assert _cache_chunks(parallel_path) == _cache_chunks(serial_path)
    assert result.manifest["cache"]["computed_chunks"] == 6

    def unexpected_submit(*args, **kwargs):
        """Fail if a warm run submits any new permutation work."""
        raise AssertionError("warm cache should not submit worker jobs")

    monkeypatch.setattr(parallel.ProcessPoolExecutor, "submit", unexpected_submit)
    warm = compute_significance(batch, config, serial_path, **kwargs)
    assert warm.pairs == serial.pairs
    assert warm.manifest["cache"]["computed_chunks"] == 0
    assert warm.manifest["cache"]["reused_chunks"] == 6


def test_parallel_resumes_a_partially_cached_pair(tmp_path):
    """Only the missing direction chunk is recomputed when switching workers."""
    batch, test_set, references = _three_evaluator_batch(tmp_path)
    kwargs = {"test_set_path": test_set.path, "ground_truth_path": references.path}
    path = tmp_path / "cache.sqlite3"
    serial = compute_significance(batch, SignificanceConfig(32), path, **kwargs)
    original = _cache_chunks(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM permutation_chunks WHERE fingerprint=?", (original[0][0],))
    resumed = compute_significance(batch, SignificanceConfig(32, workers=4), path, **kwargs)
    assert resumed.pairs == serial.pairs
    assert _cache_chunks(path) == original
    assert resumed.manifest["cache"]["computed_chunks"] == 1
    assert resumed.manifest["cache"]["reused_chunks"] == 5


def _worker_probe(work):
    """Report the actual child PID and native thread environment to the parent."""
    return os.getpid(), {name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )}


def test_process_workers_limit_native_threads_and_restore_environment(monkeypatch):
    """Worker processes get one native thread regardless of the parent's settings."""
    import wmt26_mteval_task1.significance_parallel as parallel
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.setattr(parallel, "compute_pair_chunks", _worker_probe)
    work = (DirectionWork("probe", np.ones((2, 4)), np.ones((2, 4)), (0,)),)
    results = list(completed_pair_work(((work, index) for index in range(3)), 4))
    assert len(results) == 3
    for _, (pid, environment) in results:
        assert pid != os.getpid()
        assert set(environment.values()) == {"1"}
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"


def test_worker_failure_propagates_without_hanging():
    """Invalid worker input returns an exception instead of partial significance."""
    work = (DirectionWork("bad", np.ones((2, 4)), np.ones((3, 4)), (0,)),)
    with pytest.raises(ValueError):
        list(completed_pair_work([(work, None)], 2))


def test_worker_cli_defaults_and_invalid_values():
    """Expose opt-in process counts and reject invalid values before loading data."""
    assert build_parser().parse_args([]).significance_workers == 1
    assert build_parser().parse_args(["--significance-workers", "4"]).significance_workers == 4
    for value in (0, -1):
        with pytest.raises(ValueError, match="workers"):
            SignificanceConfig(32, workers=value)
        assert main(["--significance-workers", str(value)]) == 2
