import importlib.util
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SUBMISSION_875921 = "submission__task1__875921__predictions"
GOLD = "ground_truth__record"


def test_fixture_shape(mini_grand_path):
    pf = pq.ParquetFile(mini_grand_path)
    assert pf.metadata.num_rows == 16 and pf.metadata.num_columns == 107
    assert b"wmt26_source_manifest" in pf.metadata.metadata


def test_fixture_cases(mini_grand_path):
    ids = pq.read_table(mini_grand_path, columns=["item_id"]).column(0).to_pylist()
    for frag in [
        "ces_Latn_###_ukr_Cyrl",
        "_###_GAMBITP_###_",
        "_###_de_DE_###_",
        "RefQuality-PD_RefQuality-PE_",
        "_###_hi_###_",
    ]:
        assert any(frag in i for i in ids), frag


def test_fixture_items_sorted_and_distinct(mini_grand_path):
    ids = pq.read_table(mini_grand_path, columns=["item_id"]).column(0).to_pylist()
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids) == 16


def test_fixture_schema_and_metadata_preserved(mini_grand_path):
    schema = pq.read_schema(mini_grand_path)
    assert len(schema.names) == 107
    assert [str(schema.field(n).type) for n in ("item_id", "src")] == ["string", "string"]
    json_cols = [f.name for f in schema if str(f.type) == "extension<arrow.json>"]
    assert len(json_cols) == 105
    assert SUBMISSION_875921 in json_cols
    for key in (
        b"huggingface",
        b"wmt26_schema_version",
        b"wmt26_source_manifest",
        b"wmt26_scope",
        b"wmt26_intended_split",
    ):
        assert key in schema.metadata, key
    assert schema.metadata[b"wmt26_schema_version"] == b"wmt26-grand-parquet-v1"


def test_fixture_item_shapes(mini_grand_path):
    """The 16 rows must still exhibit every cell shape the later builders handle."""
    rows = pq.read_table(mini_grand_path).to_pylist()
    refs = [None if r["ref"] is None else json.loads(r["ref"]) for r in rows]
    resources = [None if r["resources"] is None else json.loads(r["resources"]) for r in rows]
    hyps = [json.loads(r["hyps"]) for r in rows]

    assert any(ref is None for ref in refs), "an item whose ref cell is null (GAMBITP)"
    assert {"text": None, "type": None} in refs, "official ces_Latn-ukr_Cyrl: ref text null"
    assert {ref["type"] for ref in refs if ref} >= {"postedit", "pseudo", "undefined"}
    assert any(res and "video" in res for res in resources)
    assert any(res and "screenshot" in res for res in resources)
    assert any("" in h.values() for h in hyps), "an item with an empty-string hypothesis"

    covered = [r[SUBMISSION_875921] is not None for r in rows]
    assert any(covered) and not all(covered), "875921 must be a partial-coverage run"


def load_generator(repo_root: Path):
    path = repo_root / "tests" / "fixtures" / "make_mini_grand.py"
    spec = importlib.util.spec_from_file_location("make_mini_grand", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_holds_no_raw_gold(mini_grand_path):  # Ruling R24
    """Every cell of ``ground_truth__record`` is null: the raw cESA records are not committed."""
    column = pq.read_table(mini_grand_path, columns=[GOLD]).column(GOLD)
    assert len(column) == 16
    assert column.null_count == 16


def test_the_generator_nulls_the_gold_and_changes_nothing_else(repo_root, mini_grand_path):
    generator = load_generator(repo_root)
    table = pq.read_table(mini_grand_path)
    field = table.schema.field(GOLD)
    index = table.schema.get_field_index(GOLD)
    records = pa.array(['[{"annotation": 1}]'] * table.num_rows, pa.string())
    with_gold = table.set_column(index, field, pa.ExtensionArray.from_storage(field.type, records))
    assert with_gold.column(GOLD).null_count == 0  # the stand-in really holds records

    cleaned = generator.drop_gold(with_gold)
    assert cleaned.column(GOLD).null_count == table.num_rows
    assert cleaned.schema.equals(with_gold.schema, check_metadata=True)
    assert cleaned.equals(table)  # the fixture is exactly this, and the other columns are intact


def test_regenerating_the_fixture_keeps_the_gold_null(repo_root, mini_grand_path, tmp_path):
    """The generator, run on the fixture itself, selects the same 16 rows and nulls the gold."""
    generator = load_generator(repo_root)
    again = tmp_path / "again.parquet"
    assert generator.main([str(mini_grand_path), str(again)]) == 0
    regenerated = pq.read_table(again)
    assert regenerated.column(GOLD).null_count == 16
    assert regenerated.equals(pq.read_table(mini_grand_path))
    assert pq.read_schema(again).equals(pq.read_schema(mini_grand_path), check_metadata=True)


@pytest.mark.fulldata
def test_fixture_rows_are_the_real_rows_but_for_the_gold(mini_grand_path):
    """Needs ``GRAND``: each fixture row equals its row of the real file in every other column."""
    grand = os.environ.get("GRAND")
    if not grand or not Path(grand).is_file():
        pytest.skip("GRAND is not set to the real wmt26-grand.parquet")
    fixture = {r["item_id"]: r for r in pq.read_table(mini_grand_path).to_pylist()}
    others = [n for n in pq.read_schema(mini_grand_path).names if n != GOLD]
    seen = set()
    with pq.ParquetFile(grand) as real:
        for batch in real.iter_batches(batch_size=256, columns=others):
            for row in batch.to_pylist():
                mine = fixture.get(row["item_id"])
                if mine is None:
                    continue
                seen.add(row["item_id"])
                assert {name: mine[name] for name in others} == row, row["item_id"]
    assert seen == set(fixture)
    assert all(row[GOLD] is None for row in fixture.values())


@pytest.mark.fulldata
def test_fixture_schema_equals_real_grand(mini_grand_path):
    """Needs the real grand file: ``GRAND=/path/to/wmt26-grand.parquet pytest -m fulldata``."""
    grand = os.environ.get("GRAND")
    if not grand or not Path(grand).is_file():
        pytest.skip("GRAND is not set to the real wmt26-grand.parquet")
    assert pq.read_schema(mini_grand_path).equals(pq.read_schema(grand), check_metadata=True)
