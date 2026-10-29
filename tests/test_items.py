import os
from pathlib import Path

import pyarrow as pa
import pytest

from wmt26_mteval.grand import Grand
from wmt26_mteval.schemas import SCHEMAS, SHARED_FIELDS
from wmt26_mteval.tables.items import build_items, build_pairs

SHARED_NAMES = [
    "collection",
    "challenge_set",
    "domain",
    "src_lang",
    "tgt_lang",
    "src_lang_orig",
    "tgt_lang_orig",
    "lp",
    "is_priority",
    "doc_id",
    "para_idx",
]

ITEMS_COLUMNS = [
    "item_id",
    *SHARED_NAMES,
    "src",
    "ref_text",
    "ref_type",
    "ref_type_orig",
    "asr",
    "video_path",
    "screenshot_path",
    "n_systems",
]

PAIRS_COLUMNS = [
    "item_id",
    *SHARED_NAMES,
    "system",
    "system_type",
    "src",
    "hyp",
    "ref_text",
    "ref_type",
    "is_reference_candidate",
    "hyp_is_empty",
]


class FakeGrand:
    """Stands in for ``Grand``: the builders only call ``items()``."""

    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


def official_item(para, hyps, ref=None, src="Source.", resources=None):
    return {
        "item_id": f"official_###_eng_Latn_###_deu_Latn_###_news_###_doc_###_{para}",
        "src": src,
        "ref": ref,
        "hyps": hyps,
        "resources": resources,
    }


def test_items(mini_grand_path, ctx):
    t = build_items(Grand(mini_grand_path, expected_sha256=None), ctx)
    assert t.schema.equals(SCHEMAS["items"], check_metadata=False) and t.num_rows == 16
    r = {x["item_id"]: x for x in t.to_pylist()}
    ukr = next(v for k, v in r.items() if "ces_Latn_###_ukr_Cyrl" in k)
    # Review Focus 5
    assert (ukr["ref_type"], ukr["ref_text"], ukr["ref_type_orig"]) == ("none", None, None)
    fece = next(v for k, v in r.items() if "_###_FECE_###_" in k)
    # a reference text of unrecorded type is "unknown"; "none" means no reference text
    assert (fece["ref_type"], fece["ref_type_orig"], fece["domain"]) == (
        "unknown",
        "undefined",
        None,
    )
    assert fece["ref_text"] is not None
    sp = next(
        v for k, v in r.items() if k.startswith("official_###_ces_Latn_###_deu_Latn_###_speech")
    )
    assert sp["video_path"].startswith("multimodal_inputs/") and sp["asr"]


def test_pairs(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    t = build_pairs(g, ctx)
    assert t.schema.equals(SCHEMAS["pairs"], check_metadata=False)
    assert t.num_rows == sum(len(i["hyps"]) for i in g.items())
    rows = t.to_pylist()
    assert any(r["hyp_is_empty"] and r["hyp"] == "" for r in rows)  # empty kept verbatim
    native = [
        r for r in rows if r["ref_type"] == "native" and r["system"] == "Human (from scratch)"
    ]
    assert native and all(r["is_reference_candidate"] for r in native)
    assert {r["system_type"] for r in rows} >= {"mt", "challenge_hyp"}
    assert [(r["item_id"], r["system"]) for r in rows] == sorted(
        (r["item_id"], r["system"]) for r in rows
    )


def test_column_order():
    assert SCHEMAS["items"].names == ITEMS_COLUMNS
    assert SCHEMAS["pairs"].names == PAIRS_COLUMNS
    assert [f.name for f in SHARED_FIELDS] == SHARED_NAMES


def test_shared_field_types():
    types = {f.name: f.type for f in SHARED_FIELDS}
    assert types.pop("is_priority") == pa.bool_()
    assert types.pop("para_idx") == pa.int32()
    assert set(types.values()) == {pa.string()}


@pytest.mark.parametrize("config", ["items", "pairs"])
def test_every_field_has_a_one_sentence_description(config):
    for field in SCHEMAS[config]:
        description = (field.metadata or {}).get(b"description", b"").decode()
        assert description, f"{config}.{field.name} has no description"
        assert description.endswith(".") and description.count(". ") == 0, (config, field.name)


def test_para_idx_is_nullable_and_documented():
    field = SCHEMAS["items"].field("para_idx")
    assert field.type == pa.int32() and field.nullable
    description = field.metadata[b"description"].decode()
    assert "null for FECE and QEbreak, whose item ids end in a segment hash" in description


def test_items_sorted_unique_and_shared_columns_match_context(mini_grand_path, ctx):
    t = build_items(Grand(mini_grand_path, expected_sha256=None), ctx)
    ids = t.column("item_id").to_pylist()
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    for row in t.to_pylist():
        assert {k: row[k] for k in SHARED_NAMES} == ctx.describe(row["item_id"])


def test_items_null_para_idx_only_for_fece_and_qebreak(mini_grand_path, ctx):
    rows = build_items(Grand(mini_grand_path, expected_sha256=None), ctx).to_pylist()
    for row in rows:
        assert (row["para_idx"] is None) == (row["challenge_set"] in ("FECE", "QEbreak"))
    assert any(row["para_idx"] is None for row in rows)
    assert any(isinstance(row["para_idx"], int) for row in rows)


def test_items_resources_columns(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    resources = {i["item_id"]: i["resources"] for i in g.items()}
    rows = build_items(g, ctx).to_pylist()
    assert any(r["screenshot_path"] for r in rows) and any(r["video_path"] for r in rows)
    for row in rows:
        res = resources[row["item_id"]] or {}
        assert row["asr"] == res.get("asr")
        assert row["video_path"] == res.get("video")
        assert row["screenshot_path"] == res.get("screenshot")


def test_items_n_systems_and_reference_columns(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    by_id = {i["item_id"]: i for i in g.items()}
    for row in build_items(g, ctx).to_pylist():
        item = by_id[row["item_id"]]
        assert row["src"] == item["src"]
        assert row["n_systems"] == len(item["hyps"])
        ref = item["ref"]
        assert row["ref_text"] == (ref["text"] if ref else None)


def test_pairs_carry_item_level_columns(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    by_id = {i["item_id"]: i for i in g.items()}
    for row in build_pairs(g, ctx).to_pylist():
        item = by_id[row["item_id"]]
        assert row["src"] == item["src"]
        assert row["hyp"] == item["hyps"][row["system"]]
        assert row["ref_text"] == (item["ref"]["text"] if item["ref"] else None)
        assert {k: row[k] for k in SHARED_NAMES} == ctx.describe(row["item_id"])


def test_builders_sort_unordered_input(ctx):
    items = [
        official_item(2, {"b": "x", "a": "y"}),
        official_item(0, {"z": "x", "m": "y"}),
    ]
    grand = FakeGrand(items)
    assert build_items(grand, ctx).column("item_id").to_pylist() == sorted(
        i["item_id"] for i in items
    )
    rows = build_pairs(grand, ctx).to_pylist()
    assert [(r["item_id"], r["system"]) for r in rows] == sorted(
        (r["item_id"], r["system"]) for r in rows
    )
    assert len(rows) == 4


def test_pairs_flags_and_verbatim_text(ctx):
    ref = {"text": "Hallo Welt", "type": "human"}
    hyps = {
        "Human (from scratch)": "Hallo Welt",
        "PseudoRef": "Hallo Welt\n ",  # Review Focus 2: only surrounding whitespace differs
        "Some MT": "Hallo Welt!",
        "Empty MT": "",
        "Blank MT": " \n ",
        "FECE_hyp1": "Hallo",
    }
    rows = {
        r["system"]: r
        for r in build_pairs(FakeGrand([official_item(0, hyps, ref)]), ctx).to_pylist()
    }
    assert {s: r["hyp"] for s, r in rows.items()} == hyps  # never trimmed
    assert {s: r["is_reference_candidate"] for s, r in rows.items()} == {
        "Human (from scratch)": True,
        "PseudoRef": True,
        "Some MT": False,
        "Empty MT": False,
        "Blank MT": False,
        "FECE_hyp1": False,
    }
    assert {s: r["hyp_is_empty"] for s, r in rows.items()} == {
        "Human (from scratch)": False,
        "PseudoRef": False,
        "Some MT": False,
        "Empty MT": True,
        "Blank MT": True,  # whitespace-only counts as empty, like the reference comparison
        "FECE_hyp1": False,
    }
    assert {s: r["system_type"] for s, r in rows.items()} == {
        "Human (from scratch)": "human_scratch",
        "PseudoRef": "pseudo_ref",
        "Some MT": "mt",
        "Empty MT": "mt",
        "Blank MT": "mt",
        "FECE_hyp1": "challenge_hyp",
    }
    assert {r["ref_type"] for r in rows.values()} == {"native"}


@pytest.mark.parametrize(
    "ref, expected",
    [
        (None, (None, "none", None)),
        ({"text": None, "type": None}, (None, "none", None)),
        ({"text": "Ref", "type": "undefined"}, ("Ref", "unknown", "undefined")),
        ({"text": None, "type": "undefined"}, (None, "none", "undefined")),
        ({"text": "Ref", "type": None}, ("Ref", "unknown", None)),
        ({"text": "Ref", "type": "postedit"}, ("Ref", "postedit", "postedit")),
    ],
)
def test_no_reference_spellings(ctx, ref, expected):
    grand = FakeGrand([official_item(0, {"Some MT": "Hyp"}, ref)])
    item = build_items(grand, ctx).to_pylist()[0]
    assert (item["ref_text"], item["ref_type"], item["ref_type_orig"]) == expected
    pair = build_pairs(grand, ctx).to_pylist()[0]
    assert (pair["ref_text"], pair["ref_type"]) == expected[:2]
    assert pair["is_reference_candidate"] is False


def test_item_without_systems_gets_no_pair_rows(ctx):
    grand = FakeGrand([official_item(0, {})])
    assert build_items(grand, ctx).to_pylist()[0]["n_systems"] == 0
    assert build_pairs(grand, ctx).num_rows == 0


@pytest.fixture(scope="module")
def real_grand():
    path = os.environ.get("GRAND")
    if not path or not Path(path).is_file():
        pytest.skip("GRAND is not set to the real wmt26-grand.parquet")
    return Grand(Path(path))


@pytest.mark.fulldata
def test_real_items_and_pairs_row_counts(real_grand, ctx):
    items = build_items(real_grand, ctx)
    pairs = build_pairs(real_grand, ctx)
    assert items.num_rows == 26885
    assert pairs.num_rows == 363242
    assert sum(items.column("n_systems").to_pylist()) == pairs.num_rows
