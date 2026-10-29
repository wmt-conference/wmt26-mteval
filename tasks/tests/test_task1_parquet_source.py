"""Focused checks for reconstructing Task 1 sources from parquet provenance."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from wmt26_mteval_task1.io import read_jsonl
from wmt26_mteval_task1.parquet_source import Task1ParquetSource, current_source
from wmt26_mteval_task1.significance import _significance_input_sha256
from wmt26_mteval_task1.submissions import discover_baselines


def test_parquet_restores_source_order_and_canonical_alias(tmp_path: Path) -> None:
    """A reused baseline column must retain each source's own line order."""

    test_path = "test-set-files/mteval-test26.jsonl"
    gold_path = "human_annotations/ground-truth.jsonl"
    baseline_path = "baselines/task1_deduplicated/example__deduplicated/predictions.jsonl"
    canonical = "baseline__task1__example__predictions"
    alias = "baseline__task1_deduplicated__example__deduplicated__predictions"
    manifest = [
        {
            "path": path,
            "column": column,
            "canonical_column": canonical_column,
            "kind": kind,
            "task_namespace": namespace,
            "run_id": None,
            "sha256": hashlib.sha256(path.encode()).hexdigest(),
        }
        for path, column, canonical_column, kind, namespace in (
            (test_path, "test_set__record", "__item_fields__", "test_set", "test_set"),
            (gold_path, "ground_truth__record", "ground_truth__record", "ground_truth", "ground_truth"),
            (baseline_path, alias, canonical, "baseline", "task1_deduplicated"),
        )
    ]
    provenance = [
        {
            "test_set__record": {
                "line_numbers": [2], "record_fields": [["item_id", "src", "hyps"]],
                "extra_fields": [{}],
            },
            alias: {"line_numbers": [2]},
        },
        {
            "test_set__record": {
                "line_numbers": [1], "record_fields": [["item_id", "src", "hyps"]],
                "extra_fields": [{}],
            },
            alias: {"line_numbers": [1]},
        },
    ]
    table = pa.table({
        "item_id": ["second", "first"],
        "src": ["second source", "first source"],
        "ref": [None, None],
        "hyps": [json.dumps({"system": "second"}), json.dumps({"system": "first"})],
        "resources": [None, None],
        "ground_truth__record": [None, None],
        canonical: [json.dumps([{"task1_pred": {"system": 2}}]), json.dumps([{"task1_pred": {"system": 1}}])],
        "source_provenance": [json.dumps(value) for value in provenance],
    }).replace_schema_metadata({
        b"wmt26_schema_version": b"wmt26-grand-parquet-v1",
        b"wmt26_source_manifest": json.dumps(manifest).encode(),
    })
    parquet_path = tmp_path / "grand.parquet"
    pq.write_table(table, parquet_path)
    (tmp_path / "cards").mkdir()
    source = Task1ParquetSource(parquet_path, tmp_path / "cards")

    assert current_source() is None
    with source.activate():
        assert [record["item_id"] for _, record in read_jsonl(source.TEST_SET)] == ["first", "second"]
        assert list(read_jsonl(Path(baseline_path))) == [
            (1, {"item_id": "first", "task1_pred": {"system": 1}}),
            (2, {"item_id": "second", "task1_pred": {"system": 2}}),
        ]
        baselines, issues = discover_baselines(Path("baselines/task1_deduplicated"))
        assert not issues
        assert len(baselines) == 1
        assert baselines[0].model_card_path == (
            tmp_path / "cards/baselines/task1_deduplicated/example__deduplicated/model_card.json"
        )
        assert source.fingerprint(Path(baseline_path)) != source.fingerprint(source.TEST_SET)
        assert _significance_input_sha256(Path(baseline_path)) == manifest[-1]["sha256"]
    assert current_source() is None

    changed = table.set_column(1, "src", pa.array(["changed", "first source"]))
    changed_path = tmp_path / "changed.parquet"
    pq.write_table(changed, changed_path)
    changed_source = Task1ParquetSource(changed_path, tmp_path / "cards")
    assert source.fingerprint(source.TEST_SET) != changed_source.fingerprint(source.TEST_SET)
