"""Bounded, resumable Task 2 permutations; only the parent writes checkpoints."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import signal
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np

from wmt26_mteval_task2.significance import evaluate_exchanges_by_direction, permutation_masks
from wmt26_mteval_task2.significance_progress import PermutationProgress
from wmt26_mteval_task2.significance_storage import exclusive_lock, numerical_environment, prepared_hash

_WORK = None


def initialize_worker(directions, ids, sources, seed):
    """Install immutable inputs; let the parent handle terminal interrupts."""
    global _WORK
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORK = (directions, ids, sources, seed)


def compute_chunk(task):
    """Compute a deterministic draw chunk without writing any external state."""
    a, b, start, count = task
    directions, ids, sources, seed = _WORK
    masks = permutation_masks(seed, (ids[a], ids[b]), start, count, sources)
    return task, evaluate_exchanges_by_direction(directions, a, b, masks)


def run_permutations(
    directions, ids, sources, database: Path, input_fingerprint: str, *,
    permutations=1000, seed=98001, workers=1, chunk_size=16,
    pairs=None, stop_after_chunks=None, start_method="fork",
    progress="auto", progress_interval=1.0, progress_path=None, require_complete=False,
):
    """Compute or resume fixed-count draws and return arrays plus run status.

Each pair maps to a float64 array with axes draw/hybrid/direction/component.
Direction order is recorded in the checkpoint; components are screen PDP and
source SPA. Schema 3 rejects older checkpoints that retained only aggregates.

SQLite FULL synchronous commits protect completed chunks across process or
machine restarts, subject to the filesystem honoring durable writes. At most
twice the worker count is submitted concurrently. A first Ctrl-C stops new
submissions and drains already running chunks; unfinished work remains safely
recomputable. Only the parent displays progress or touches SQLite.
"""
    if min(permutations, workers, chunk_size, sources) < 1 or seed < 0:
        raise ValueError("Positive dimensions/workers and a nonnegative seed required")
    if list(ids) != sorted(set(ids)):
        raise ValueError("Evaluator identities must be unique and sorted")
    if stop_after_chunks is not None and stop_after_chunks < 1:
        raise ValueError("stop_after_chunks must be positive")
    if start_method not in mp.get_all_start_methods():
        raise ValueError(f"Unavailable multiprocessing start method: {start_method}")
    if progress not in ("auto", "plain", "off") or not np.isfinite(progress_interval) or progress_interval <= 0:
        raise ValueError("Invalid progress options")
    pairs = tuple(pairs if pairs is not None else ((a, b) for a in range(len(ids)) for b in range(a+1, len(ids))))
    if not pairs or len(set(pairs)) != len(pairs) or any(not 0 <= a < b < len(ids) for a, b in pairs):
        raise ValueError("Invalid evaluator pair list")
    names = [direction.name for direction in directions]
    if not names or len(set(names)) != len(names):
        raise ValueError("Prepared directions must be nonempty and unique")
    shape = (2, len(directions), 2)
    config = json.dumps(dict(version=3, input_fingerprint=input_fingerprint,
                             directions=names, payload_axes=["draw", "hybrid", "direction", "component"],
                             prepared_sha256=prepared_hash(directions, sources),
                             numerical_environment=numerical_environment(),
                             ids=list(ids), sources=sources, permutations=permutations,
                             seed=seed, chunk_size=chunk_size), sort_keys=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    arrays = {pair: np.full((permutations, *shape), np.nan) for pair in pairs}
    present = set()
    completed_by_pair = dict.fromkeys(pairs, 0)
    computed = 0
    interrupted = False
    started = time.perf_counter()
    with exclusive_lock(database.with_suffix(database.suffix + ".lock")), sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, config TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS chunks (a INTEGER, b INTEGER, start INTEGER, n INTEGER, payload BLOB, checksum TEXT, PRIMARY KEY(a,b,start))")
        previous = connection.execute("SELECT config FROM metadata WHERE id=1").fetchone()
        if previous and previous[0] != config:
            raise ValueError("Checkpoint configuration/input fingerprint or numerical environment differs; use a new output directory")
        if not previous:
            connection.execute("INSERT INTO metadata VALUES (1, ?)", (config,))
            connection.commit()
        for a, b, start, count, payload, checksum in connection.execute("SELECT * FROM chunks ORDER BY a,b,start"):
            if not 0 <= a < b < len(ids):
                raise ValueError("Malformed checkpoint evaluator pair")
            if start % chunk_size or not 0 <= start < permutations or count != min(chunk_size, permutations-start):
                raise ValueError("Malformed checkpoint draw interval")
            if len(payload) != count * 4 * len(directions) * 8 or hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError("Corrupted permutation checkpoint")
            values = np.frombuffer(payload, dtype="<f8").reshape(count, *shape)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite checkpoint statistics")
            if (a, b) not in arrays:
                continue
            arrays[a, b][start:start+count] = values
            completed_by_pair[a, b] += count
            present.add((a, b, start))
        cached = len(present)
        tasks = [(a, b, start, min(chunk_size, permutations-start)) for a, b in pairs
                 for start in range(0, permutations, chunk_size) if (a, b, start) not in present]
        if require_complete and tasks:
            raise ValueError("Cannot render results: checkpoint is incomplete")
        if stop_after_chunks is not None:
            tasks = tasks[:stop_after_chunks]
        display = PermutationProgress(total_draws=len(pairs)*permutations,
            cached_draws=sum(completed_by_pair.values()), total_pairs=len(pairs),
            completed_pairs=sum(n == permutations for n in completed_by_pair.values()),
            total_chunks=len(pairs)*((permutations+chunk_size-1)//chunk_size), cached_chunks=cached,
            workers=workers, mode=progress, interval=progress_interval, path=progress_path)
        display.refresh(force=True)

        def save(expected_task, result):
            """Validate worker identity and transactionally save its exact draw range."""
            nonlocal computed
            task, values = result
            if task != expected_task:
                raise ValueError("Worker returned the wrong draw range")
            a, b, start, count = task
            if values.shape != (count, *shape) or not np.isfinite(values).all():
                raise ValueError("Malformed worker result")
            payload = np.asarray(values, dtype="<f8").tobytes()
            connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                               (a, b, start, count, payload, hashlib.sha256(payload).hexdigest()))
            connection.commit()
            arrays[a, b][start:start+count] = values
            computed += 1
            completed_by_pair[a, b] += count
            display.advance(count, pair_completed=completed_by_pair[a, b] == permutations)

        def request_stop(signum, frame):
            """Stop submitting work; keep already completed checkpoints intact."""
            nonlocal interrupted
            interrupted = True

        previous_handler = signal.signal(signal.SIGINT, request_stop)
        try:
            if tasks:
                with ProcessPoolExecutor(workers, mp_context=mp.get_context(start_method),
                                         initializer=initialize_worker,
                                         initargs=(directions, ids, sources, seed)) as executor:
                    iterator = iter(tasks)
                    pending = {}
                    for _ in range(min(2*workers, len(tasks))):
                        task = next(iterator)
                        pending[executor.submit(compute_chunk, task)] = task
                    while pending:
                        done, _ = wait(pending, timeout=progress_interval, return_when=FIRST_COMPLETED)
                        for future in done:
                            task = pending.pop(future)
                            if not future.cancelled():
                                save(task, future.result())
                            next_task = None if interrupted else next(iterator, None)
                            if next_task is not None:
                                pending[executor.submit(compute_chunk, next_task)] = next_task
                        if interrupted:
                            for future in pending:
                                future.cancel()
                        display.refresh(in_flight=len(pending), status="stopping" if interrupted else "running")
        except BaseException:
            display.refresh(status="failed; saved work retained", force=True)
            raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        complete = all(np.isfinite(array).all() for array in arrays.values())
        state = display.refresh(status="complete" if complete else "interrupted" if interrupted else "partial pilot", force=True)
    return arrays, dict(complete=complete, interrupted=interrupted,
                        cached_chunks=cached, computed_chunks=computed,
                        cached_draws=state["cached_draws"], new_draws=state["new_draws"],
                        elapsed_seconds=time.perf_counter()-started, workers=workers,
                        start_method=start_method)
