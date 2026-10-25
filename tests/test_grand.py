import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wmt26_mteval.grand import GRAND_SHA256, ChecksumError, Grand, ManifestEntry

BASELINE_873002 = "baseline__task2__873002__predictions"


def test_checksum_enforced(mini_grand_path):
    with pytest.raises(ChecksumError):
        Grand(mini_grand_path)  # default expects the real sha


def test_checksum_constant():
    assert GRAND_SHA256 == "a61675552cebb68fb8303cf08a23fbec31e5bb4e0bd26a1fdee1a9bfafc2fb7a"


def test_checksum_accepts_matching_sha(mini_grand_path):
    sha = hashlib.sha256(mini_grand_path.read_bytes()).hexdigest()
    assert len(Grand(mini_grand_path, expected_sha256=sha).manifest) == 122


def test_checksum_error_names_both_digests(mini_grand_path):
    with pytest.raises(ChecksumError) as exc:
        Grand(mini_grand_path, expected_sha256="0" * 64)
    assert "0" * 64 in str(exc.value)
    assert hashlib.sha256(mini_grand_path.read_bytes()).hexdigest() in str(exc.value)


def test_manifest_and_canonical(mini_grand_path):
    g = Grand(mini_grand_path, expected_sha256=None)
    # 1 ground truth + 50 submissions + 50 baselines
    assert len(g.manifest) == 122 and len(g.canonical_entries()) == 101


def test_manifest_entries_are_typed(mini_grand_path):
    g = Grand(mini_grand_path, expected_sha256=None)
    assert all(isinstance(e, ManifestEntry) for e in g.manifest)
    test_set = [e for e in g.manifest if e.kind == "test_set"]
    assert len(test_set) == 1 and test_set[0].canonical_column == "__item_fields__"
    assert test_set[0].run_id is None and test_set[0].records == 26885
    baseline = next(e for e in g.manifest if e.column == BASELINE_873002)
    assert (baseline.kind, baseline.task_namespace) == ("baseline", "task2")
    assert baseline.run_id == "873002"
    assert baseline.source_id == baseline.canonical_source_id == 105
    with pytest.raises(AttributeError):
        baseline.sha256 = "x"  # frozen


def test_canonical_entries_exclude_test_set_and_duplicates(mini_grand_path):
    g = Grand(mini_grand_path, expected_sha256=None)
    canonical = g.canonical_entries()
    assert {e.kind for e in canonical} == {"ground_truth", "submission", "baseline"}
    assert all(e.canonical_source_id == e.source_id for e in canonical)
    assert [e.source_id for e in canonical] == sorted(e.source_id for e in canonical)


def test_items_and_records(mini_grand_path):
    g = Grand(mini_grand_path, expected_sha256=None)
    items = list(g.items())
    assert len(items) == 16
    gamb = next(i for i in items if "_###_GAMBITP_###_" in i["item_id"])
    assert gamb["ref"] is None
    iid, rec = next(g.iter_records(BASELINE_873002))
    assert "item_id" not in rec and "task2_pred" in rec


def test_items_shape(mini_grand_path):
    g = Grand(mini_grand_path, expected_sha256=None)
    items = {i["item_id"]: i for i in g.items()}
    assert all(set(i) == {"item_id", "src", "ref", "hyps", "resources"} for i in items.values())
    assert list(items) == sorted(items)
    ukr = next(i for k, i in items.items() if "ces_Latn_###_ukr_Cyrl" in k)
    assert ukr["ref"] == {"text": None, "type": None}
    assert all(isinstance(i["hyps"], dict) and i["hyps"] for i in items.values())
    assert all(isinstance(i["src"], str) for i in items.values())
    assert all(i["resources"] is None or isinstance(i["resources"], dict) for i in items.values())
    assert any(i["resources"] and "video" in i["resources"] for i in items.values())
    assert any("" in i["hyps"].values() for i in items.values())  # empty hypotheses stay verbatim


def test_iter_records_yields_only_covered_items(mini_grand_path):
    g = Grand(mini_grand_path, expected_sha256=None)
    column = "submission__task1__875921__predictions"
    covered = {k for k, rec in g.iter_records(column)}
    raw = pq.read_table(mini_grand_path, columns=["item_id", column]).to_pylist()
    expected = {r["item_id"] for r in raw if r[column] is not None}
    assert covered == expected and 0 < len(covered) < 16


def test_iter_records_unknown_column(mini_grand_path):
    g = Grand(mini_grand_path, expected_sha256=None)
    with pytest.raises(KeyError):
        next(g.iter_records("no_such_column"))


def test_iter_records_rejects_cells_that_are_not_single_records(tmp_path):
    table = pa.table(
        {
            "item_id": ["a", "b"],
            "src": ["s", "s"],
            "ref": [None, None],
            "hyps": ["{}", "{}"],
            "resources": [None, None],
            "run__x": [json.dumps([{"v": 1}]), json.dumps([{"v": 1}, {"v": 2}])],
            "run__empty": [json.dumps([{"v": 1}]), json.dumps([])],
        }
    ).replace_schema_metadata({"wmt26_source_manifest": "[]"})
    path = tmp_path / "tiny.parquet"
    pq.write_table(table, path)
    g = Grand(path, expected_sha256=None)
    assert g.manifest == [] and g.canonical_entries() == []
    for column in ("run__x", "run__empty"):
        it = g.iter_records(column)
        assert next(it) == ("a", {"v": 1})
        with pytest.raises(ValueError):
            next(it)
