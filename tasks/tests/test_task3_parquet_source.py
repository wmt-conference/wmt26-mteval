"""Audit parquet-only preparation with synthetic records and independent input hashes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wmt26_mteval_task1.parquet_source import current_source
from wmt26_mteval_task3 import parquet_source as integrated
from wmt26_mteval_task3 import preparation
from wmt26_mteval_task3.io import DataError


def digest(data: bytes) -> str:
    """Return a synthetic fixture's independently calculated SHA-256."""
    return hashlib.sha256(data).hexdigest()


def parquet_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    """Create shuffled rows, three submission tracks and an aliased historical baseline.

    Pin reconstructed hashes from hand-built source records, not from the reader
    under test. Include an unselected current baseline to guard catalog selection.
    """
    specs = [
        ("test-set-files/mteval-test26.jsonl", "test", "__item_fields__", "test_set", "test_set"),
        ("human_annotations/ground-truth.jsonl", "gold", "gold", "ground_truth", "ground_truth"),
        ("submissions/task1/11/predictions.jsonl", "t1", "t1", "submission", "task1"),
        ("submissions/task2/22/predictions.jsonl", "t2", "t2", "submission", "task2"),
        ("baselines/task2/23/predictions.jsonl", "relocated", "relocated", "baseline", "task2"),
        ("submissions/task3/33/prediction.jsonl", "t3", "t3", "submission", "task3"),
        ("baselines/task1_prededup_and_enhy_fix/example__pre_dedup_pre_enhy_fix/predictions.jsonl",
         "historical_alias", "baseline", "baseline", "task1_prededup_and_enhy_fix"),
        ("baselines/task1_deduplicated/example/predictions.jsonl", "current_alias", "baseline", "baseline", "task1_deduplicated"),
    ]
    manifest = [{"path": path, "column": column, "canonical_column": canonical,
                 "kind": kind, "task_namespace": namespace, "records": 2,
                 "sha256": digest(path.encode())}
                for path, column, canonical, kind, namespace in specs]
    payloads = {
        "gold": {"screens": []},
        "t1": {"task1_pred": {"system": {"errors": [], "omission": None}}},
        "t2": {"task2_pred": {"system": 80}},
        "relocated": {"task2_pred": {"system": 70}},
        "t3": {"task3_pred": {"system": True}},
        "baseline": {"task1_pred": {"system": {"errors": [], "omission": None}},
                     "task2_pred": {"system": 85}},
    }
    provenance = []
    for line in (2, 1):
        per_item = {entry["column"]: {"line_numbers": [line]} for entry in manifest}
        per_item["test"].update(record_fields=[["item_id", "src", "hyps"]], extra_fields=[{}])
        provenance.append(json.dumps(per_item))
    table = pa.table({
        "item_id": ["second", "first"], "src": ["second source", "first source"],
        "ref": [None, None], "hyps": [json.dumps({"system": "synthetic"})] * 2,
        "resources": [None, None], "source_provenance": provenance,
        **{key: [json.dumps([value])] * 2 for key, value in payloads.items()},
    }).replace_schema_metadata({
        b"wmt26_schema_version": b"wmt26-grand-parquet-v1",
        b"wmt26_source_manifest": json.dumps(manifest).encode(),
    })
    data_file = tmp_path / "grand.parquet"
    pq.write_table(table, data_file)
    cards = tmp_path / "cards"
    families = []
    for name, ids, indexes, required in (
        ("submissions", ["33"], [5], []),
        ("task1_converted", ["11"], [2], ["Permission"]),
        ("task2_converted", ["22", "23"], [3, 4], ["Permission", "Threshold"]),
        ("task1_baselines_converted", ["example"], [6], []),
        ("task2_baselines_converted", ["example"], [6], []),
    ):
        entries = []
        for entry_id, index in zip(ids, indexes):
            source = manifest[index]
            member = str(Path(source["path"]).parent / "model_card.json")
            if "baselines_converted" in name:
                task, member = 2, "baselines/task2/example/model_card.json"
            else:
                task = {"submissions": 3, "task1_converted": 1, "task2_converted": 2}[name]
            card = cards / f"task{task}" / member
            card.parent.mkdir(parents=True, exist_ok=True)
            card.write_text(json.dumps([{"question": "Permission", "answer": False},
                                        {"question": "Threshold", "answer": ""}]))
            entries.append({"entry_id": entry_id, "dump_candidates": [str(Path(source["path"]).parent)],
                            "prediction_filename": Path(source["path"]).name,
                            "historical_prediction_sha256": source["sha256"],
                            "model_card_member": member})
        families.append({"name": name, "entries": entries, "required_card_questions": required})
    targets = {"families": families, "shared_inputs": [
        {"member": entry["path"], "sha256": entry["sha256"]} for entry in manifest[:2]]}
    monkeypatch.setattr(integrated, "load_targets", lambda: targets)
    monkeypatch.setattr(preparation, "load_targets", lambda: targets)
    sources = {}
    for relative, entry in integrated.selected_sources(integrated.Task3ParquetSource(data_file, cards)).items():
        records = []
        for item in ("first", "second"):
            if entry["kind"] == "test_set":
                record = {"item_id": item, "src": f"{item} source", "hyps": {"system": "synthetic"}}
            else:
                record = {"item_id": item, **payloads[entry["canonical_column"]]}
            records.append((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        raw = b"".join(records)
        sources[relative] = {"sha256": digest(raw), "bytes": len(raw), "records": 2,
                             "source_path": entry["path"], "logical_sha256": entry["sha256"]}
    catalog = {"parquet_sha256": digest(data_file.read_bytes()), "sources": sources}
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    monkeypatch.setattr(integrated, "CATALOG_PATH", catalog_path)
    return data_file, cards, catalog


def test_parquet_only_preparation_preserves_source_order_and_public_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepare all tracks without old files or repairs, including the relocated Task 2 entry."""
    data_file, cards, _ = parquet_fixture(tmp_path, monkeypatch)
    # An adjacent legacy scientific file must not become a data source.
    (cards / "task3/submissions/task3/33/prediction.jsonl").write_text("DO NOT READ")
    output = tmp_path / "prepared"
    manifest = integrated.prepare_parquet_inputs(data_file, output, cards)
    assert current_source() is None
    assert manifest["metadata_repairs"] == []
    assert manifest["schema_version"] == 2
    assert preparation.validate_inputs(output) == manifest
    records = [json.loads(line) for line in (output / "submissions/task3/33/prediction.jsonl").read_text().splitlines()]
    assert [row["item_id"] for row in records] == ["first", "second"]
    assert (output / "submissions/task2/23/model_card.json").read_bytes() == (cards / "task2/baselines/task2/23/model_card.json").read_bytes()
    assert json.loads((output / "submissions/task1/11/model_card.json").read_text())[0]["answer"] is False
    assert not (output / "baselines/task1_deduplicated").exists()
    assert len(manifest["files"]) == 12
    with pytest.raises(DataError, match="existing input directory"):
        integrated.prepare_parquet_inputs(data_file, output, cards)


def test_parquet_source_pin_rejects_changed_input_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a repacked parquet until its physical snapshot is explicitly audited."""
    data_file, cards, _ = parquet_fixture(tmp_path, monkeypatch)
    pq.write_table(pq.read_table(data_file), data_file, compression="gzip")
    with pytest.raises(DataError, match="Parquet identity changed"):
        integrated.prepare_parquet_inputs(data_file, tmp_path / "prepared", cards)
    assert not (tmp_path / "prepared").exists()


def test_parquet_reconstruction_is_checked_independently_of_run_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing a staged record and its manifest cannot bypass the packaged input audit."""
    data_file, cards, _ = parquet_fixture(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    manifest = integrated.prepare_parquet_inputs(data_file, output, cards)
    relative = "submissions/task3/33/prediction.jsonl"
    path = output / relative
    path.write_text("changed\n")
    manifest["files"][relative].update(sha256=digest(path.read_bytes()), bytes=path.stat().st_size)
    (output / "input_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataError, match="audited source"):
        preparation.validate_inputs(output)


def test_incomplete_public_cards_fail_without_legacy_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never fill an absent public opt-in from a neighboring old card or default value."""
    data_file, cards, _ = parquet_fixture(tmp_path, monkeypatch)
    (cards / "task1/submissions/task1/11/model_card.json").write_text("[]")
    with pytest.raises(DataError, match="Public card lacks conversion metadata"):
        integrated.prepare_parquet_inputs(data_file, tmp_path / "prepared", cards)
    assert not list(tmp_path.glob(".prepared-*"))


def test_changed_logical_source_identity_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin historical source identities independently of reconstructed JSON whitespace."""
    data_file, cards, _ = parquet_fixture(tmp_path, monkeypatch)
    source = integrated.Task3ParquetSource(data_file, cards)
    source.entries = copy.deepcopy(source.entries)
    source.entries[Path("submissions/task3/33/prediction.jsonl")]["sha256"] = "0" * 64
    with pytest.raises(DataError, match="prediction identity changed"):
        integrated.selected_sources(source)
