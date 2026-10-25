"""Discovery of the evaluator runs held in the grand file.

A run is one participant submission or one organiser baseline.  Each run owns one canonical column
of the grand file; ``list_runs`` turns the manifest into one ``RunSpec`` per table to publish.
Baselines come in two snapshots of the same run, ``official`` (the deduplicated predictions, or
the only ones there are) and ``alt`` (the original LLM predictions from before deduplication).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from wmt26_mteval.grand import Grand

# Baseline folders are named ``<run>_task1`` plus, for the later snapshots, one of these suffixes.
_SNAPSHOT_SUFFIXES = ("__deduplicated", "__pre_dedup_pre_enhy_fix")
_TASK1_SUFFIX = "_task1"

# Baseline namespaces that give a table, with the task they belong to.  Task 1 baselines also
# carry Task 2 scores, but they are Task 1 runs.  The pre-deduplication snapshot is left out.
_BASELINE_TASKS = {"task1": "task1", "task1_deduplicated": "task1", "task2": "task2"}
_SKIPPED_BASELINE_NAMESPACES = frozenset({"task1_prededup_and_enhy_fix"})
_DEDUPLICATED = "task1_deduplicated"
_ORIGINAL = "task1"


@dataclass(frozen=True)
class RunSpec:
    """One table to publish: where its records live in the grand file and how it is labelled."""

    run_id: str
    kind: Literal["submission", "baseline"]
    task: str
    variant: Literal["official", "alt"]
    column: str
    source_path: str


def baseline_run_id(folder: str) -> str:
    """Run id of a baseline from its folder name.

    The snapshot suffix goes first and the ``_task1`` suffix second, so that
    ``gemma_task1__deduplicated`` and ``gemma_task1`` are both ``gemma``.  Numeric Task 2
    folders such as ``872013`` have neither suffix and are returned as they are.
    """
    for suffix in _SNAPSHOT_SUFFIXES:
        if folder.endswith(suffix):
            folder = folder[: -len(suffix)]
            break
    if folder.endswith(_TASK1_SUFFIX):
        folder = folder[: -len(_TASK1_SUFFIX)]
    return folder


def _baseline_folder(path: str) -> str:
    """``<folder>`` of ``baselines/<namespace>/<folder>/predictions.jsonl``."""
    parts = PurePosixPath(path).parts
    if len(parts) != 4 or parts[0] != "baselines":
        raise ValueError(f"unexpected baseline path {path!r}")
    return parts[2]


def list_runs(grand: Grand) -> list[RunSpec]:
    """Every run to publish, sorted by ``(kind, run_id, variant)``.

    Submissions use the manifest ``run_id``.  Baselines use ``baseline_run_id`` of their folder.
    A deduplicated Task 1 baseline is ``official``; its original is ``alt`` when a deduplicated
    snapshot exists, and ``official`` when it does not (xCOMET: the files are identical, so the
    manifest keeps only the original).  Numeric Task 2 baselines are ``official``.  The
    pre-deduplication snapshot is skipped, and so are the ground truth and the test set.
    An unexpected baseline namespace or a repeated run raises ``ValueError``.
    """
    baselines = []
    runs = []
    for entry in grand.canonical_entries():
        if entry.kind == "submission":
            runs.append(
                RunSpec(
                    run_id=entry.run_id,
                    kind="submission",
                    task=entry.task_namespace,
                    variant="official",
                    column=entry.column,
                    source_path=entry.path,
                )
            )
        elif entry.kind == "baseline":
            if entry.task_namespace in _SKIPPED_BASELINE_NAMESPACES:
                continue
            if entry.task_namespace not in _BASELINE_TASKS:
                raise ValueError(
                    f"{entry.path}: unexpected baseline namespace {entry.task_namespace!r}"
                )
            baselines.append((entry, baseline_run_id(_baseline_folder(entry.path))))

    deduplicated = {run_id for e, run_id in baselines if e.task_namespace == _DEDUPLICATED}
    for entry, run_id in baselines:
        original_with_snapshot = entry.task_namespace == _ORIGINAL and run_id in deduplicated
        runs.append(
            RunSpec(
                run_id=run_id,
                kind="baseline",
                task=_BASELINE_TASKS[entry.task_namespace],
                variant="alt" if original_with_snapshot else "official",
                column=entry.column,
                source_path=entry.path,
            )
        )

    runs.sort(key=lambda r: (r.kind, r.run_id, r.variant))
    counts = Counter((r.kind, r.run_id, r.variant) for r in runs)
    repeated = sorted(key for key, n in counts.items() if n > 1)
    if repeated:
        raise ValueError(f"duplicate runs: {repeated}")
    return runs
