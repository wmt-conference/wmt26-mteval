"""The ``items`` and ``pairs`` tables: one row per test item, and one per item and system."""

from __future__ import annotations

from operator import itemgetter

import pyarrow as pa

from wmt26_mteval.context import Context
from wmt26_mteval.grand import Grand
from wmt26_mteval.ids import classify_mt_system, is_reference_candidate, map_ref_type
from wmt26_mteval.schemas import SCHEMAS


def _table(schema: pa.Schema, rows: list[dict]) -> pa.Table:
    """Arrow table with exactly ``schema``; a row missing one of its columns is a ``KeyError``."""
    columns = [pa.array([row[field.name] for row in rows], type=field.type) for field in schema]
    return pa.Table.from_arrays(columns, schema=schema)


def _ref_text(ref: dict | None) -> str | None:
    return None if ref is None else ref.get("text")


def _resource(item: dict, key: str) -> str | None:
    resources = item["resources"]
    return None if resources is None else resources.get(key)


def build_items(grand: Grand, ctx: Context) -> pa.Table:
    """One row per test item, sorted by ``item_id``."""
    rows = []
    for item in grand.items():
        ref = item["ref"]
        ref_type, ref_type_orig = map_ref_type(ref)
        rows.append(
            {
                "item_id": item["item_id"],
                **ctx.describe(item["item_id"]),
                "src": item["src"],
                "ref_text": _ref_text(ref),
                "ref_type": ref_type,
                "ref_type_orig": ref_type_orig,
                "asr": _resource(item, "asr"),
                "video_path": _resource(item, "video"),
                "screenshot_path": _resource(item, "screenshot"),
                "n_systems": len(item["hyps"]),
            }
        )
    rows.sort(key=itemgetter("item_id"))
    return _table(SCHEMAS["items"], rows)


def build_pairs(grand: Grand, ctx: Context) -> pa.Table:
    """One row per item and system, sorted by ``(item_id, system)``.

    Hypotheses are copied verbatim, so empty and whitespace-only outputs stay as they are and
    are only flagged.  An item with no hypotheses gets no rows.
    """
    rows = []
    for item in grand.items():
        ref = item["ref"]
        ref_text = _ref_text(ref)
        ref_type, _ = map_ref_type(ref)
        shared = ctx.describe(item["item_id"])
        for system, hyp in item["hyps"].items():
            rows.append(
                {
                    "item_id": item["item_id"],
                    **shared,
                    "system": system,
                    "system_type": classify_mt_system(system),
                    "src": item["src"],
                    "hyp": hyp,
                    "ref_text": ref_text,
                    "ref_type": ref_type,
                    "is_reference_candidate": is_reference_candidate(hyp, ref_text),
                    "hyp_is_empty": not hyp.strip(),
                }
            )
    rows.sort(key=itemgetter("item_id", "system"))
    return _table(SCHEMAS["pairs"], rows)
