import copy
import re

import pyarrow as pa
import pytest

from wmt26_mteval.grand import Grand
from wmt26_mteval.runs import RunSpec, list_runs
from wmt26_mteval.schemas import SCHEMAS
from wmt26_mteval.tables import predictions
from wmt26_mteval.tables.predictions import (
    KNOWN_DIAGNOSTIC_KEYS,
    build_run_table,
    normalise_raw_record,
    roundtrip_record,
)

SUBMISSION_COLUMNS = [
    "run_id",
    "task",
    "item_id",
    "lp",
    "system",
    "t1_errors",
    "t1_omission",
    "t1_instruction_fault",
    "t2_score",
    "t3_error_free",
]

ERROR_TYPE = pa.struct(
    [
        pa.field("start", pa.int32()),
        pa.field("end", pa.int32()),
        pa.field("severity", pa.string()),
        pa.field("category", pa.string()),
    ]
)


def item_id(para):
    return f"official_###_eng_Latn_###_deu_Latn_###_news_###_doc_###_{para}"


class FakeGrand:
    """Stands in for ``Grand``: ``build_run_table`` only calls ``iter_records(column)``."""

    def __init__(self, records):
        self._records = records
        self.requested = []

    def iter_records(self, column):
        self.requested.append(column)
        return iter(self._records)


def spec(kind="submission", task="task1", variant="official", run_id="111111"):
    return RunSpec(
        run_id=run_id,
        kind=kind,
        task=task,
        variant=variant,
        column="the_column",
        source_path="x/predictions.jsonl",
    )


def t1(errors=(), omission=None, **extra):
    return {"errors": list(errors), "omission": omission, **extra}


def err(start, end, severity="minor", **extra):
    return {"start": start, "end": end, "severity": severity, **extra}


def rows_by_system(table):
    return {(r["item_id"], r["system"]): r for r in table.to_pylist()}


# --- schemas -------------------------------------------------------------------------------------


def test_prediction_schemas():
    assert SCHEMAS["submissions"].names == SUBMISSION_COLUMNS
    assert SCHEMAS["baselines"].names == [*SUBMISSION_COLUMNS, "variant"]
    sub = SCHEMAS["submissions"]
    assert [str(sub.field(n).type) for n in ("run_id", "task", "item_id", "lp", "system")] == [
        "string"
    ] * 5
    assert sub.field("t1_errors").type == pa.list_(ERROR_TYPE)
    assert sub.field("t1_omission").type == pa.string()
    assert sub.field("t1_instruction_fault").type == pa.string()
    assert sub.field("t2_score").type == pa.float64()
    assert sub.field("t3_error_free").type == pa.bool_()
    assert SCHEMAS["baselines"].field("variant").type == pa.string()


def test_prediction_fields_are_described_in_one_sentence():
    for name in ("submissions", "baselines"):
        for field in SCHEMAS[name]:
            text = field.metadata[b"description"].decode()
            assert text.endswith(".") and not re.search(r"\.\s+[A-Z]", text), (name, field.name)


def test_shared_fields_are_reused_not_redefined():
    for name in ("submissions", "baselines"):
        for column in ("item_id", "lp", "system"):
            assert (
                SCHEMAS[name]
                .field(column)
                .equals(SCHEMAS["pairs"].field(column), check_metadata=True)
            )


# --- the main behaviours, on the mini grand ------------------------------------------------------


def test_scores_kept_verbatim(mini_grand_path, ctx):  # Review Focus 4
    g = Grand(mini_grand_path, expected_sha256=None)
    run = next(r for r in list_runs(g) if r.run_id == "869546")
    t = build_run_table(g, run, ctx)
    s = t.column("t2_score").to_pylist()
    assert t.schema.field("t2_score").type == pa.float64() and min(s) < 0
    raw = {
        (iid, system): value
        for iid, rec in g.iter_records(run.column)
        for system, value in rec["task2_pred"].items()
    }
    assert {(r["item_id"], r["system"]): r["t2_score"] for r in t.to_pylist()} == raw


def test_missing_keys_become_null():  # Review Focus 3
    raw = {
        "task1_pred": {
            "A": {"errors": [{"start": 1, "end": 2, "severity": "minor"}], "omission": None}
        }
    }
    n = normalise_raw_record(raw)
    assert n["task1_pred"]["A"]["instruction_fault"] is None
    assert n["task1_pred"]["A"]["errors"][0]["category"] is None


def test_alt_has_no_task2(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    alt = next(r for r in list_runs(g) if r.variant == "alt")
    table = build_run_table(g, alt, ctx)
    assert table.column("t2_score").null_count == table.num_rows > 0
    assert table.column("t1_errors").null_count == 0


def test_partial_runs_emit_no_null_rows(mini_grand_path, ctx):  # Review Focus 5
    g = Grand(mini_grand_path, expected_sha256=None)
    run = next(r for r in list_runs(g) if r.run_id == "875921")  # PragmaSpan, 397 items
    t = build_run_table(g, run, ctx)
    covered = {i for i, _ in g.iter_records(run.column)}
    assert covered and len(covered) < 16
    assert set(t.column("item_id").to_pylist()) == covered
    assert t.column("system").null_count == 0 and t.column("lp").null_count == 0


def test_roundtrip_every_run(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    for run in list_runs(g):
        by_item = {}
        for row in build_run_table(g, run, ctx).to_pylist():
            by_item.setdefault(row["item_id"], []).append(row)
        records = list(g.iter_records(run.column))
        assert set(by_item) == {iid for iid, _ in records}, run.run_id
        for iid, rec in records:
            got = roundtrip_record(by_item[iid])
            exp = normalise_raw_record(rec)
            if run.variant == "alt":
                exp = {"task1_pred": exp["task1_pred"]}
            assert got == exp, (run.run_id, iid)


# --- shape of the built tables on the mini grand -------------------------------------------------


def test_built_tables_match_their_schemas_including_metadata(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    runs = list_runs(g)
    submission = next(r for r in runs if r.kind == "submission" and r.task == "task1")
    baseline = next(r for r in runs if r.kind == "baseline" and r.variant == "official")
    sub_table = build_run_table(g, submission, ctx)
    base_table = build_run_table(g, baseline, ctx)
    assert sub_table.schema.equals(SCHEMAS["submissions"], check_metadata=True)
    assert base_table.schema.equals(SCHEMAS["baselines"], check_metadata=True)
    assert set(base_table.column("variant").to_pylist()) == {"official"}
    assert set(sub_table.column("run_id").to_pylist()) == {submission.run_id}
    assert set(sub_table.column("task").to_pylist()) == {submission.task}


def test_every_run_table_is_sorted_with_unique_keys(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    for run in list_runs(g):
        table = build_run_table(g, run, ctx)
        keys = list(zip(table.column("item_id").to_pylist(), table.column("system").to_pylist()))
        assert keys == sorted(keys) and len(set(keys)) == len(keys), run.run_id
        assert table.num_rows > 0, run.run_id


def test_lp_comes_from_the_item_id(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    run = next(r for r in list_runs(g) if r.run_id == "873483")
    for row in build_run_table(g, run, ctx).to_pylist():
        assert row["lp"] == ctx.describe(row["item_id"])["lp"]


def test_rows_cover_the_union_of_systems_in_each_record(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    for run in list_runs(g):
        expected = set()
        for iid, rec in g.iter_records(run.column):
            tasks = ["task1_pred"] if run.variant == "alt" else list(rec)
            tasks = [t for t in tasks if t.endswith("_pred")]
            expected |= {(iid, s) for t in tasks for s in rec[t]}
        table = build_run_table(g, run, ctx)
        got = set(zip(table.column("item_id").to_pylist(), table.column("system").to_pylist()))
        assert got == expected, run.run_id


def test_official_task1_baselines_keep_their_task2_scores(mini_grand_path, ctx):
    g = Grand(mini_grand_path, expected_sha256=None)
    runs = [r for r in list_runs(g) if r.kind == "baseline" and r.task == "task1"]
    official = [r for r in runs if r.variant == "official"]
    assert official
    for run in official:
        table = build_run_table(g, run, ctx)
        assert table.column("t2_score").null_count == 0, run.run_id
        assert table.column("t3_error_free").null_count == table.num_rows, run.run_id


def test_only_the_runs_column_is_read(ctx):
    grand = FakeGrand([(item_id(0), {"task2_pred": {"A": 1}})])
    build_run_table(grand, spec(task="task2"), ctx)
    assert grand.requested == ["the_column"]


# --- hand-made records ---------------------------------------------------------------------------


def test_systems_are_the_union_and_other_tasks_are_null(ctx):
    record = {
        "task1_pred": {"A": t1([err(0, 3, category="fluency")], "major"), "B": t1()},
        "task2_pred": {"B": 70, "C": 55.5},
        "task3_pred": {"C": True},
    }
    table = build_run_table(FakeGrand([(item_id(0), record)]), spec(), ctx)
    rows = rows_by_system(table)
    assert sorted(s for _, s in rows) == ["A", "B", "C"]
    a, b, c = (rows[(item_id(0), s)] for s in "ABC")
    assert a["t1_errors"] == [{"start": 0, "end": 3, "severity": "minor", "category": "fluency"}]
    assert (a["t1_omission"], a["t2_score"], a["t3_error_free"]) == ("major", None, None)
    assert (b["t1_errors"], b["t1_omission"], b["t2_score"], b["t3_error_free"]) == (
        [],
        None,
        70.0,
        None,
    )
    assert (c["t1_errors"], c["t1_omission"], c["t2_score"], c["t3_error_free"]) == (
        None,
        None,
        55.5,
        True,
    )


def test_absent_task1_entry_is_null_but_empty_errors_is_an_empty_list(ctx):
    record = {"task1_pred": {"A": t1()}, "task2_pred": {"A": 1, "B": 2}}
    table = build_run_table(FakeGrand([(item_id(0), record)]), spec(), ctx)
    rows = rows_by_system(table)
    assert rows[(item_id(0), "A")]["t1_errors"] == []
    assert rows[(item_id(0), "B")]["t1_errors"] is None
    assert table.column("t1_errors").null_count == 1


def test_missing_category_and_instruction_fault_are_null(ctx):  # Review Focus 3
    record = {
        "task1_pred": {
            "A": t1([err(1, 2), err(3, 4, "major", category=None), err(5, 6, category="term")]),
            "B": t1([], None, instruction_fault="off-topic"),
            "C": t1([], "minor", instruction_fault=None),
        }
    }
    rows = rows_by_system(build_run_table(FakeGrand([(item_id(0), record)]), spec(), ctx))
    a, b, c = (rows[(item_id(0), s)] for s in "ABC")
    assert [e["category"] for e in a["t1_errors"]] == [None, None, "term"]
    assert [(e["start"], e["end"], e["severity"]) for e in a["t1_errors"]] == [
        (1, 2, "minor"),
        (3, 4, "major"),
        (5, 6, "minor"),
    ]
    assert a["t1_instruction_fault"] is None
    assert b["t1_instruction_fault"] == "off-topic"
    assert (c["t1_omission"], c["t1_instruction_fault"]) == ("minor", None)


def test_error_order_and_int32_offsets_are_kept(ctx):
    errors = [err(40, 50), err(0, 5), err(10, 20), err(0, 5)]
    record = {"task1_pred": {"A": t1(errors)}}
    table = build_run_table(FakeGrand([(item_id(0), record)]), spec(), ctx)
    assert table.schema.field("t1_errors").type == pa.list_(ERROR_TYPE)
    got = table.to_pylist()[0]["t1_errors"]
    assert [(e["start"], e["end"]) for e in got] == [(40, 50), (0, 5), (10, 20), (0, 5)]


def test_scores_are_float64_and_unchanged_outside_0_100(ctx):  # Review Focus 4
    scores = {"int": 100, "zero": 0, "neg": -137, "negf": -0.25, "high": 138.6, "big": 250}
    table = build_run_table(
        FakeGrand([(item_id(0), {"task2_pred": scores})]), spec(task="task2"), ctx
    )
    assert table.schema.field("t2_score").type == pa.float64()
    got = {r["system"]: r["t2_score"] for r in table.to_pylist()}
    assert got == {k: float(v) for k, v in scores.items()}
    assert all(type(v) is float for v in got.values())


def test_rows_are_sorted_whatever_the_record_order(ctx):
    records = [
        (item_id(2), {"task3_pred": {"b": True, "a": False}}),
        (item_id(0), {"task3_pred": {"z": True, "m": False}}),
    ]
    table = build_run_table(FakeGrand(records), spec(task="task3"), ctx)
    keys = [(r["item_id"], r["system"]) for r in table.to_pylist()]
    assert keys == sorted(keys) and len(keys) == 4


@pytest.mark.parametrize("chunk_items", [1, 2, 3, 5])
def test_chunking_does_not_change_the_table(monkeypatch, ctx, chunk_items):
    records = [
        (item_id(i), {"task1_pred": {"B": t1([err(i, i + 1)]), "A": t1()}, "task2_pred": {"B": i}})
        for i in (4, 1, 3, 0, 2)
    ]
    expected = build_run_table(FakeGrand(records), spec(), ctx)
    monkeypatch.setattr(predictions, "CHUNK_ITEMS", chunk_items)
    got = build_run_table(FakeGrand(records), spec(), ctx)
    assert got.equals(expected, check_metadata=True) and got.num_rows == 10


def test_baseline_tables_carry_the_variant(ctx):
    record = {"task1_pred": {"A": t1()}, "task2_pred": {"A": 3}}
    for variant in ("official", "alt"):
        run = spec(kind="baseline", variant=variant, run_id="gemma")
        table = build_run_table(FakeGrand([(item_id(0), record)]), run, ctx)
        assert table.schema.equals(SCHEMAS["baselines"], check_metadata=True)
        assert table.column("variant").to_pylist() == [variant]
        assert table.column("run_id").to_pylist() == ["gemma"]


def test_alt_rows_ignore_task2_and_its_extra_systems(ctx):
    record = {"task1_pred": {"A": t1()}, "task2_pred": {"A": 3, "B": 4}}
    run = spec(kind="baseline", variant="alt", run_id="gemma")
    table = build_run_table(FakeGrand([(item_id(0), record)]), run, ctx)
    assert [(r["system"], r["t2_score"]) for r in table.to_pylist()] == [("A", None)]


def test_empty_run_gives_an_empty_table_with_the_schema(ctx):
    table = build_run_table(FakeGrand([]), spec(), ctx)
    assert table.num_rows == 0
    assert table.schema.equals(SCHEMAS["submissions"], check_metadata=True)


# --- normalise_raw_record and roundtrip_record ---------------------------------------------------


def raw_record():
    return {
        "task1_pred": {
            "A": {"errors": [{"start": 1, "end": 2, "severity": "minor"}], "omission": "minor"},
            "B": {"errors": [], "omission": None, "instruction_fault": None},
        },
        "task2_pred": {"A": 5, "B": 7.5},
        "task3_pred": {"A": True},
        "stage1_finish_reason": {"A": "stop"},
    }


def test_normalise_does_not_mutate_its_input():
    raw = raw_record()
    before = copy.deepcopy(raw)
    out = normalise_raw_record(raw)
    assert raw == before
    out["task1_pred"]["A"]["errors"][0]["start"] = 99
    out["task1_pred"]["B"]["errors"].append(err(0, 1))
    out["task3_pred"]["A"] = False
    assert raw == before


def test_normalise_fills_keys_casts_scores_and_keeps_only_predictions():
    out = normalise_raw_record(raw_record())
    assert set(out) == {"task1_pred", "task2_pred", "task3_pred"}  # no diagnostics
    assert out["task1_pred"]["A"] == {
        "errors": [{"start": 1, "end": 2, "severity": "minor", "category": None}],
        "omission": "minor",
        "instruction_fault": None,
    }
    assert out["task1_pred"]["B"]["errors"] == []
    assert out["task2_pred"] == {"A": 5.0, "B": 7.5}
    assert type(out["task2_pred"]["A"]) is float
    assert out["task3_pred"] == {"A": True}


def test_known_diagnostic_keys_are_dropped_and_nothing_else():
    assert KNOWN_DIAGNOSTIC_KEYS == frozenset({"stage1_finish_reason"})
    raw = raw_record()
    assert "stage1_finish_reason" in raw
    without = {k: v for k, v in raw.items() if k != "stage1_finish_reason"}
    out = normalise_raw_record(raw)
    assert "stage1_finish_reason" not in out
    assert out == normalise_raw_record(without)
    assert normalise_raw_record({"stage1_finish_reason": {"A": "stop"}}) == {}


def test_unknown_top_level_key_raises():
    raw = raw_record()
    raw["mystery"] = {"A": 1}
    with pytest.raises(ValueError, match=r"record.*mystery"):
        normalise_raw_record(raw)


def test_unknown_system_key_raises():
    raw = raw_record()
    raw["task1_pred"]["B"]["mystery"] = 1
    with pytest.raises(ValueError, match=r"task1_pred\['B'\].*mystery"):
        normalise_raw_record(raw)


def test_unknown_error_key_raises():
    raw = raw_record()
    raw["task1_pred"]["B"]["errors"] = [err(0, 1), err(2, 3, mystery="x")]
    with pytest.raises(ValueError, match=r"task1_pred\['B'\]\.errors\[1\].*mystery"):
        normalise_raw_record(raw)


def test_building_a_table_fails_on_unknown_keys_too(ctx):
    for bad in (
        {"task2_pred": {"A": 1}, "mystery": 1},
        {"task1_pred": {"A": t1(mystery=1)}},
        {"task1_pred": {"A": t1([err(0, 1, mystery=1)])}},
    ):
        with pytest.raises(ValueError, match="mystery"):
            build_run_table(FakeGrand([(item_id(0), bad)]), spec(), ctx)


def test_normalise_keeps_only_the_tasks_present():
    assert normalise_raw_record({"task2_pred": {"A": 1}}) == {"task2_pred": {"A": 1.0}}
    assert normalise_raw_record({}) == {}


def test_roundtrip_rebuilds_only_the_tasks_present(ctx):
    record = {
        "task1_pred": {"A": t1([err(0, 2, category="x")], "major", instruction_fault="f")},
        "task2_pred": {"A": 1, "B": 2.5},
    }
    rows = build_run_table(FakeGrand([(item_id(0), record)]), spec(), ctx).to_pylist()
    got = roundtrip_record(rows)
    assert got == normalise_raw_record(record)
    assert set(got) == {"task1_pred", "task2_pred"}
    assert set(got["task1_pred"]) == {"A"}  # B has no Task 1 entry


def test_roundtrip_of_an_alt_run_rebuilds_task1_only(ctx):
    record = {"task1_pred": {"A": t1([err(0, 2)])}, "task2_pred": {"A": 3}}
    run = spec(kind="baseline", variant="alt", run_id="gemma")
    rows = build_run_table(FakeGrand([(item_id(0), record)]), run, ctx).to_pylist()
    assert roundtrip_record(rows) == {"task1_pred": normalise_raw_record(record)["task1_pred"]}


def test_roundtrip_treats_missing_and_null_alike(ctx):  # Review Focus 3
    with_nulls = {
        "task1_pred": {
            "A": {
                "errors": [{"start": 1, "end": 2, "severity": "minor", "category": None}],
                "omission": None,
                "instruction_fault": None,
            }
        }
    }
    with_gaps = {
        "task1_pred": {
            "A": {"errors": [{"start": 1, "end": 2, "severity": "minor"}], "omission": None}
        }
    }
    assert normalise_raw_record(with_gaps) == normalise_raw_record(with_nulls)
    rows = build_run_table(FakeGrand([(item_id(0), with_gaps)]), spec(), ctx).to_pylist()
    assert roundtrip_record(rows) == normalise_raw_record(with_nulls)


def test_roundtrip_of_no_rows_is_empty():
    assert roundtrip_record([]) == {}
