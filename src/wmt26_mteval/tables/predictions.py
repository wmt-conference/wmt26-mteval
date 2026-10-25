"""The ``submissions`` and ``baselines`` tables: one row per run, item and system.

A record in the grand file holds up to three dicts keyed by system name, one per task:
``task1_pred`` (error spans, an omission and an instruction fault), ``task2_pred`` (a score) and
``task3_pred`` (an error-free flag).  A submission has one of them; a Task 1 baseline has the first
two.  The table has one row per system found in any of them, with null columns for the tasks the
record lacks.

``normalise_raw_record`` gives the one canonical shape of a record, ``build_run_table`` flattens
that shape into rows and ``roundtrip_record`` folds the rows of one item back into it, so that
validation can prove the table loses nothing.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from wmt26_mteval.context import Context
from wmt26_mteval.grand import Grand
from wmt26_mteval.runs import RunSpec
from wmt26_mteval.schemas import SCHEMAS

T1, T2, T3 = "task1_pred", "task2_pred", "task3_pred"
PREDICTION_KEYS = (T1, T2, T3)
CHUNK_ITEMS = 1000  # items per Arrow chunk while a table is being built


def _normalise_task1(prediction: dict) -> dict:
    """A copy of one system's Task 1 prediction with ``instruction_fault`` and ``category`` set."""
    out = dict(prediction)
    out.setdefault("instruction_fault", None)
    out["errors"] = [{"category": None, **error} for error in prediction["errors"]]
    return out


def normalise_raw_record(rec: dict) -> dict:
    """The canonical form of a raw record; ``rec`` itself is left untouched.

    Missing ``instruction_fault`` and ``category`` keys become ``None``, so that "missing" and
    null are one thing, and Task 2 scores become ``float`` (some are ints).  Only the prediction
    dicts that the record has are kept: other keys, such as the ``stage1_finish_reason``
    diagnostic that a few baselines carry, are not part of the tables.
    """
    out: dict[str, Any] = {}
    if T1 in rec:
        out[T1] = {system: _normalise_task1(p) for system, p in rec[T1].items()}
    if T2 in rec:
        out[T2] = {system: float(score) for system, score in rec[T2].items()}
    if T3 in rec:
        out[T3] = dict(rec[T3])
    return out


def _chunk_table(schema: pa.Schema, constant: dict[str, str], cols: dict[str, list]) -> pa.Table:
    """Arrow table of the rows collected so far, with the run's constant columns filled in."""
    n = len(cols["system"])
    arrays = [
        pa.repeat(constant[f.name], n) if f.name in constant else pa.array(cols[f.name], f.type)
        for f in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def build_run_table(grand: Grand, run: RunSpec, ctx: Context) -> pa.Table:
    """The table of one run, sorted by ``(item_id, system)``.

    Only the run's own column is read, so items the run does not cover have no rows.  The table
    conforms to ``SCHEMAS["submissions"]``, or to ``SCHEMAS["baselines"]`` (which adds
    ``variant``) for a baseline.  An ``alt`` run keeps its Task 1 fields only.

    The rows are turned into Arrow every ``CHUNK_ITEMS`` items: the Python dicts for the error
    spans are what takes the memory, and a chunk lets them go.
    """
    tasks = (T1,) if run.variant == "alt" else PREDICTION_KEYS
    schema = SCHEMAS["baselines" if run.kind == "baseline" else "submissions"]
    constant = {"run_id": run.run_id, "task": run.task, "variant": run.variant}
    names = [f.name for f in schema if f.name not in constant]
    chunks = []
    cols: dict[str, list] = {name: [] for name in names}
    for count, (item_id, raw) in enumerate(grand.iter_records(run.column), start=1):
        record = normalise_raw_record(raw)
        t1, t2, t3 = (record.get(key, {}) if key in tasks else {} for key in PREDICTION_KEYS)
        systems = sorted(t1.keys() | t2.keys() | t3.keys())
        cols["item_id"] += [item_id] * len(systems)
        cols["lp"] += [ctx.describe(item_id)["lp"]] * len(systems)
        cols["system"] += systems
        for system in systems:
            task1 = t1.get(system)
            cols["t1_errors"].append(None if task1 is None else task1["errors"])
            cols["t1_omission"].append(None if task1 is None else task1["omission"])
            cols["t1_instruction_fault"].append(
                None if task1 is None else task1["instruction_fault"]
            )
            cols["t2_score"].append(t2.get(system))
            cols["t3_error_free"].append(t3.get(system))
        if count % CHUNK_ITEMS == 0:
            chunks.append(_chunk_table(schema, constant, cols))
            cols = {name: [] for name in names}
    chunks.append(_chunk_table(schema, constant, cols))
    return pa.concat_tables(chunks).sort_by([("item_id", "ascending"), ("system", "ascending")])


def roundtrip_record(rows: list[dict]) -> dict:
    """Rebuild the normalised record of one item from its table rows (one run, one item).

    A task appears in the result when at least one row carries it, and a system appears under a
    task when its row has a value for it: a null ``t1_errors`` means no Task 1 prediction, while
    an empty list is a prediction without errors.
    """
    task1: dict[str, dict] = {}
    task2: dict[str, float] = {}
    task3: dict[str, bool] = {}
    for row in rows:
        system = row["system"]
        if row["t1_errors"] is not None:
            task1[system] = {
                "errors": row["t1_errors"],
                "omission": row["t1_omission"],
                "instruction_fault": row["t1_instruction_fault"],
            }
        if row["t2_score"] is not None:
            task2[system] = row["t2_score"]
        if row["t3_error_free"] is not None:
            task3[system] = row["t3_error_free"]
    found = zip(PREDICTION_KEYS, (task1, task2, task3))
    return {key: predictions for key, predictions in found if predictions}
