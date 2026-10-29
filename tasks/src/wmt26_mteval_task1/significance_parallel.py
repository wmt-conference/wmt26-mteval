"""Bounded process scheduling for pure Task 1 permutation computations."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, TypeVar

import numpy as np

Context = TypeVar("Context")


@dataclass(frozen=True)
class DirectionWork:
    """Matrices and missing chunk indices for one fingerprinted direction stream."""

    key: str
    a: np.ndarray
    b: np.ndarray
    indices: tuple[int, ...]


ChunkResults = list[tuple[str, int, np.ndarray]]


def compute_pair_chunks(work: tuple[DirectionWork, ...]) -> ChunkResults:
    """Compute missing chunks for one evaluator pair without touching the cache."""
    # Local import keeps the scheduler independent of the statistical module's
    # initialization, including when multiprocessing imports it in a new process.
    from wmt26_mteval_task1.significance import permutation_chunk

    return [
        (direction.key, index, permutation_chunk(
            direction.a, direction.b, direction.key, index,
        ))
        for direction in work
        for index in direction.indices
    ]


@contextmanager
def single_threaded_worker_environment() -> Iterator[None]:
    """Set native thread limits before spawning Python, then restore the parent env.

    Spawned workers import NumPy afresh, so these limits apply before numerical
    libraries initialize. No SQLite connection or parent numerical state is forked.
    """
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def completed_pair_work(
    jobs: Iterable[tuple[tuple[DirectionWork, ...], Context]], workers: int,
) -> Iterator[tuple[Context, ChunkResults]]:
    """Yield pair results with at most workers jobs submitted or held in memory.

    Cache-only jobs bypass the pool. The caller retains each job's context and
    performs all cache I/O. Worker failures propagate; queued work is cancelled
    and running work is joined before returning control to the caller.
    """
    if workers == 1:
        for work, context in jobs:
            yield context, compute_pair_chunks(work)
        return

    with single_threaded_worker_environment():
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        )
        pending = {}
        source = iter(jobs)
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < workers:
                    try:
                        work, context = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    if work:
                        pending[pool.submit(compute_pair_chunks, work)] = context
                    else:
                        yield context, []
                if pending:
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        context = pending.pop(future)
                        yield context, future.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
