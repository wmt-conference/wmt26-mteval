"""Exclusive output ownership and provenance for significance runs."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """Hold a POSIX lock released on exit or reboot, not a stale PID marker.

The small lock file remains on disk; its existence is not a lock. Never unlink
it, because concurrent processes must lock the same inode.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns this run: {path}") from error
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def numerical_environment() -> dict:
    """Capture versions and numerical-library configuration used for resumption."""
    return dict(python=platform.python_version(), numpy=np.__version__,
                machine=platform.machine(), system=platform.system(),
                numpy_config=np.show_config(mode="dicts"),
                threads={key: os.environ.get(key) for key in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})


def canonical_hash(value: dict) -> str:
    """Hash stable JSON metadata, independent of dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_run_manifest(previous: dict, current: dict) -> None:
    """Reject changed run identity without treating JSON key coercion as a change."""
    immutable = ("input_fingerprint", "permutations", "spa_permutations", "spa_seed", "seed",
                 "chunk_size", "evaluator_ids", "source_support", "purpose", "directions", "checkpoint_schema")
    # Integer mapping keys become strings on disk; normalize before comparison.
    current = json.loads(json.dumps(current, allow_nan=False))
    changed = [key for key in immutable if previous.get(key) != current.get(key)]
    if changed:
        raise ValueError("Existing run inputs/configuration differ (" + ", ".join(changed)
                         + "); preserved previous outputs. Use a new output directory")


def prepared_hash(directions, sources: int) -> str:
    """Fingerprint actual prepared arrays, including support, order and scaling."""
    digest = hashlib.sha256(str(sources).encode())
    for direction in directions:
        digest.update(direction.name.encode())
        digest.update(repr((direction.human_norm, direction.spa_permutations)).encode())
        arrays = [direction.sources, direction.pdp_numerators, direction.pdp_norms,
                  direction.prediction_means, direction.prediction_sds]
        for group in direction.widths:
            arrays.extend((group.sources, group.differences, group.flipped,
                           group.human_counts, group.metric_counts))
        for array in arrays:
            digest.update(repr((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
