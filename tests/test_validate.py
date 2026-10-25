import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from wmt26_mteval.build import build_all
from wmt26_mteval.grand import Grand
from wmt26_mteval.schemas import SCHEMAS
from wmt26_mteval.validate import (
    ALLOWED_EMAILS,
    MAX_ITEM_FAILURES,
    find_pii,
    load_table2,
    validate,
)

SYSTEMS = "data/systems/test-00000.parquet"
CHALLENGE_SETS = "data/challenge_sets/test-00000.parquet"
LABELS = ["Participant name", "Email", "Data Collection Consent"]


# --- helpers -------------------------------------------------------------------------------------


def rewrite(path: Path, change, **options) -> None:
    """Read the Parquet file at ``path``, apply ``change(table) -> table`` and write it back."""
    pq.write_table(change(pq.read_table(path)), path, **options)


def set_values(table: pa.Table, column: str, values: list) -> pa.Table:
    """``table`` with ``column`` replaced by ``values``, keeping the field and its metadata."""
    field = table.schema.field(column)
    index = table.schema.get_field_index(column)
    return table.set_column(index, field, pa.array(values, field.type))


def set_cell(path: Path, column: str, row: int, value) -> None:
    def change(table):
        values = table.column(column).to_pylist()
        values[row] = value
        return set_values(table, column, values)

    rewrite(path, change)


def first_file(out: Path, pattern: str) -> Path:
    return sorted(out.glob(pattern))[0]


@pytest.fixture(scope="module")
def built_mini(tmp_path_factory, mini_grand_path, repo_root) -> Path:
    """One build of the mini grand file; the tests that edit it work on a copy (``out``)."""
    out = tmp_path_factory.mktemp("validate") / "hf"
    build_all(mini_grand_path, repo_root, out, expected_sha256=None)
    return out


@pytest.fixture
def out(built_mini, tmp_path) -> Path:
    copy_of_build = tmp_path / "hf"
    shutil.copytree(built_mini, copy_of_build)
    return copy_of_build


# --- the PII gate --------------------------------------------------------------------------------


def test_pii_gate_ignores_test_text(tmp_path, mini_grand_path, repo_root):  # Review Focus 1
    build_all(mini_grand_path, repo_root, tmp_path, expected_sha256=None)
    t = pq.read_table(tmp_path / "data/items/test-00000.parquet")
    t = t.set_column(
        t.schema.get_field_index("src"), "src", pa.array(["mail me: a.b@example.com"] * t.num_rows)
    )
    pq.write_table(t, tmp_path / "data/items/test-00000.parquet")
    assert find_pii(tmp_path) == []


def test_pii_gate_catches_card_leak(tmp_path, mini_grand_path, repo_root):
    build_all(mini_grand_path, repo_root, tmp_path, expected_sha256=None)
    (tmp_path / "README.md").write_text("contact ada@example.org")
    assert find_pii(tmp_path)


def test_pii_gate_does_not_scan_pairs_or_run_tables(out):
    for pattern, column in (
        ("data/pairs/*.parquet", "hyp"),
        ("data/pairs/*.parquet", "ref_text"),
        ("data/submissions/*.parquet", "system"),
    ):
        text = "Email ada@example.org, Participant name, Data Collection Consent"
        set_cell(first_file(out, pattern), column, 0, text)
    assert find_pii(out) == []


def test_pii_gate_is_silent_about_paths_that_do_not_exist(tmp_path):
    assert find_pii(tmp_path) == []


def test_pii_gate_names_the_file_and_column_but_not_the_address(out):
    set_cell(out / SYSTEMS, "description", 3, "ask ada.lovelace@example.org about it")
    failures = find_pii(out)
    assert len(failures) == 1
    assert SYSTEMS in failures[0] and "description" in failures[0]
    assert "ada.lovelace" not in failures[0] and "example.org" not in failures[0]


def test_pii_gate_reads_list_cells(out):
    set_cell(out / CHALLENGE_SETS, "lps", 0, ["eng-deu", "eve@example.org"])
    failures = find_pii(out)
    assert len(failures) == 1
    assert CHALLENGE_SETS in failures[0] and "lps" in failures[0]
    assert "eve@" not in failures[0]


def test_pii_gate_reports_a_column_once(out):
    path = out / SYSTEMS
    rewrite(path, lambda t: set_values(t, "link", ["a@b.org"] * t.num_rows))
    assert len(find_pii(out)) == 1


def test_pii_gate_allows_only_the_organisers_address(out):
    organisers = "wmt-qe-metrics-organizers@googlegroups.com"
    assert ALLOWED_EMAILS == {organisers}
    set_cell(out / SYSTEMS, "description", 0, f"Questions to {organisers}.")
    (out / "README.md").write_text(f"Contact: {organisers}\n", encoding="utf-8")
    assert find_pii(out) == []
    (out / "README.md").write_text(f"Contact: {organisers}, or ada@example.org\n")
    failures = find_pii(out)
    assert len(failures) == 1 and "README.md" in failures[0]
    # an address that merely contains the organisers' address is somebody else's
    (out / "README.md").write_text(f"Contact: x{organisers}\n", encoding="utf-8")
    assert len(find_pii(out)) == 1


@pytest.mark.parametrize("label", LABELS)
def test_pii_gate_flags_the_personal_data_labels_in_parquet_and_manifest(out, label):
    set_cell(out / CHALLENGE_SETS, "what_it_tests", 0, f"x {label}: y")
    failures = find_pii(out)
    assert len(failures) == 1
    assert CHALLENGE_SETS in failures[0] and "what_it_tests" in failures[0]

    set_cell(out / CHALLENGE_SETS, "what_it_tests", 0, "clean")
    manifest = json.loads((out / "build_manifest.json").read_text(encoding="utf-8"))
    manifest["note"] = label
    (out / "build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    failures = find_pii(out)
    assert len(failures) == 1 and "build_manifest.json" in failures[0]


@pytest.mark.parametrize("label", LABELS)
def test_pii_gate_allows_the_labels_in_readme_prose(out, label):
    (out / "README.md").write_text(f"The model card forms asked for a {label}.\n")
    assert find_pii(out) == []


def test_pii_gate_flags_an_address_in_the_manifest(out):
    manifest = json.loads((out / "build_manifest.json").read_text(encoding="utf-8"))
    manifest["note"] = "ada@example.org"
    (out / "build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    failures = find_pii(out)
    assert len(failures) == 1 and "build_manifest.json" in failures[0]
    assert "ada@" not in failures[0]


# --- the counts gate -----------------------------------------------------------------------------


def test_validate_mini(tmp_path, mini_grand_path, repo_root, expected_mini):
    build_all(mini_grand_path, repo_root, tmp_path, expected_sha256=None)
    assert validate(tmp_path, Grand(mini_grand_path, expected_sha256=None), expected_mini) == []


def test_the_mini_expectations_are_not_vacuous(expected_mini):
    rows = expected_mini["rows"]
    assert rows["items"] == 16 and rows["systems"] == 73 and rows["challenge_sets"] == 7
    assert all(rows[config] > 0 for config in rows)
    assert set(expected_mini["pairs_by_collection"]) == {"official", "challenge"}
    assert not expected_mini["check_table2"]


@pytest.mark.parametrize("config", ["items", "pairs", "submissions", "baselines", "systems"])
def test_wrong_row_count_is_reported(out, mini_grand, expected_mini, config):
    expected = copy.deepcopy(expected_mini)
    expected["rows"][config] += 1
    failures = validate(out, mini_grand, expected)
    assert len(failures) == 1
    assert config in failures[0] and str(expected["rows"][config]) in failures[0]


def test_pairs_by_collection_is_checked(out, mini_grand, expected_mini):
    expected = copy.deepcopy(expected_mini)
    expected["pairs_by_collection"]["challenge"] += 1
    failures = validate(out, mini_grand, expected)
    assert len(failures) == 1 and "challenge" in failures[0]


def test_pairs_by_collection_is_optional(out, mini_grand, expected_mini):
    expected = {k: v for k, v in expected_mini.items() if k != "pairs_by_collection"}
    assert validate(out, mini_grand, expected) == []


def test_counts_come_from_the_files_not_the_manifest(out, mini_grand, expected_mini):
    manifest = json.loads((out / "build_manifest.json").read_text(encoding="utf-8"))
    for info in manifest["configs"].values():
        info["rows"] += 5
    (out / "build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate(out, mini_grand, expected_mini) == []


def test_a_config_without_files_is_reported(out, mini_grand, expected_mini):
    shutil.rmtree(out / "data" / "mt_systems")
    failures = validate(out, mini_grand, expected_mini)
    assert any("mt_systems" in f for f in failures)


def test_an_unreadable_file_is_a_failure_not_a_crash(out, mini_grand, expected_mini):
    (out / "data/items/test-00000.parquet").write_bytes(b"not parquet")
    failures = validate(out, mini_grand, expected_mini)
    assert failures and any("items" in f for f in failures)


# --- the Table 2 gate ----------------------------------------------------------------------------


def official_counts(mini_pairs) -> dict[str, tuple[int, int]]:
    """``lp -> (distinct official items, official pair rows)`` of the mini fixture."""
    official = mini_pairs.filter(pc.equal(mini_pairs.column("collection"), "official"))
    rows = official.select(["lp", "item_id"]).to_pylist()
    items = Counter(lp for lp, _ in {(r["lp"], r["item_id"]) for r in rows})
    pairs = Counter(r["lp"] for r in rows)
    return {lp: (items[lp], pairs[lp]) for lp in sorted(pairs)}


def write_table2(path: Path, counts: dict[str, tuple[int, int]]) -> Path:
    lines = ["lp\tsrc_items\tpairs", *(f"{lp}\t{i}\t{p}" for lp, (i, p) in counts.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_table2_file_is_the_papers_table(repo_root):
    table = load_table2(repo_root / "data" / "metadata" / "table2.tsv")
    assert len(table) == 21
    assert table["ces-deu"] == (438, 13578) and table["zho-jpn"] == (335, 10385)
    assert sum(pairs for _, pairs in table.values()) == 252923


def test_table2_file_agrees_with_the_expected_full_counts(repo_root, expected_full):
    table = load_table2(repo_root / "data" / "metadata" / "table2.tsv")
    pairs = expected_full["pairs_by_collection"]
    assert sum(p for _, p in table.values()) == pairs["official"]
    assert pairs["official"] + pairs["challenge"] == expected_full["rows"]["pairs"]
    assert expected_full["check_table2"] is True


def test_load_table2_rejects_a_wrong_header_and_duplicates(tmp_path):
    bad = tmp_path / "t.tsv"
    bad.write_text("lp\titems\tpairs\neng-deu\t1\t2\n")
    with pytest.raises(ValueError, match="header"):
        load_table2(bad)
    bad.write_text("lp\tsrc_items\tpairs\neng-deu\t1\t2\neng-deu\t3\t4\n")
    with pytest.raises(ValueError, match="eng-deu"):
        load_table2(bad)


def test_table2_passes_when_official_counts_match(
    out, mini_grand, expected_mini, mini_pairs, tmp_path
):
    counts = official_counts(mini_pairs)
    assert mini_pairs.num_rows > sum(p for _, p in counts.values()), "needs challenge rows"
    path = write_table2(tmp_path / "table2.tsv", counts)
    expected = {**expected_mini, "check_table2": True}
    assert validate(out, mini_grand, expected, table2_path=path) == []


def test_table2_is_only_checked_when_asked(out, mini_grand, expected_mini, tmp_path):
    path = write_table2(tmp_path / "table2.tsv", {"ces-deu": (1, 1)})
    assert validate(out, mini_grand, expected_mini, table2_path=path) == []


def test_table2_reports_each_mismatch(out, mini_grand, expected_mini, mini_pairs, tmp_path):
    counts = official_counts(mini_pairs)
    lps = sorted(counts)
    wrong = dict(counts)
    wrong[lps[0]] = (counts[lps[0]][0] + 1, counts[lps[0]][1])  # items
    wrong[lps[1]] = (counts[lps[1]][0], counts[lps[1]][1] - 1)  # pairs
    del wrong[lps[2]]  # a language pair the data has and the table lacks
    wrong["ces-deu"] = (438, 13578)  # a language pair the table has and the data lacks
    path = write_table2(tmp_path / "table2.tsv", wrong)
    failures = validate(out, mini_grand, {**expected_mini, "check_table2": True}, table2_path=path)
    assert len(failures) == 4, failures
    assert any(lps[0] in f and "items" in f for f in failures)
    assert any(lps[1] in f and "pairs" in f for f in failures)
    assert any(lps[2] in f for f in failures)
    assert any("ces-deu" in f for f in failures)


def test_table2_counts_official_rows_only(out, mini_grand, expected_mini, mini_pairs, tmp_path):
    everything = Counter(mini_pairs.column("lp").to_pylist())
    items = {lp: n for lp, n in official_counts(mini_pairs).items()}
    counts = {lp: (items.get(lp, (0, 0))[0], everything[lp]) for lp in sorted(items)}
    assert counts != official_counts(mini_pairs), "needs a pair with challenge rows"
    path = write_table2(tmp_path / "table2.tsv", counts)
    failures = validate(out, mini_grand, {**expected_mini, "check_table2": True}, table2_path=path)
    assert failures


# --- the schema gate -----------------------------------------------------------------------------


def make_description_different(table: pa.Table) -> pa.Table:
    index = table.schema.get_field_index("t2_score")
    field = table.schema.field("t2_score").with_metadata({"description": "Something else."})
    return table.set_column(index, field, table.column("t2_score"))


def make_description_missing(table: pa.Table) -> pa.Table:
    index = table.schema.get_field_index("t2_score")
    return table.set_column(
        index, table.schema.field("t2_score").remove_metadata(), table.column("t2_score")
    )


def make_type_different(table: pa.Table) -> pa.Table:
    field = table.schema.field("t2_score")
    index = table.schema.get_field_index("t2_score")
    column = pc.cast(table.column("t2_score"), pa.float32())
    return table.set_column(index, pa.field("t2_score", pa.float32(), True, field.metadata), column)


def make_nested_type_different(table: pa.Table) -> pa.Table:
    wide = pa.list_(
        pa.struct(
            [
                pa.field("start", pa.int64()),
                pa.field("end", pa.int32()),
                pa.field("severity", pa.string()),
                pa.field("category", pa.string()),
            ]
        )
    )
    field = table.schema.field("t1_errors")
    index = table.schema.get_field_index("t1_errors")
    column = pc.cast(table.column("t1_errors"), wide)
    return table.set_column(index, pa.field("t1_errors", wide, True, field.metadata), column)


def make_required(table: pa.Table) -> pa.Table:
    index = table.schema.get_field_index("lp")
    return table.set_column(
        index, table.schema.field("lp").with_nullable(False), table.column("lp")
    )


def make_column_missing(table: pa.Table) -> pa.Table:
    return table.remove_column(table.schema.get_field_index("t3_error_free"))


def make_column_renamed(table: pa.Table) -> pa.Table:
    index = table.schema.get_field_index("t2_score")
    return table.set_column(
        index, table.schema.field("t2_score").with_name("t2_scores"), table.column("t2_score")
    )


def make_columns_reordered(table: pa.Table) -> pa.Table:
    return table.select(list(reversed(table.schema.names)))


@pytest.mark.parametrize(
    ("change", "column"),
    [
        (make_description_different, "t2_score"),
        (make_description_missing, "t2_score"),
        (make_type_different, "t2_score"),
        (make_nested_type_different, "t1_errors"),
        (make_required, "lp"),
        (make_column_missing, "t3_error_free"),
        (make_column_renamed, "t2_score"),
    ],
)
def test_schema_drift_is_reported(out, mini_grand, expected_mini, change, column):
    path = first_file(out, "data/submissions/test-task2-*.parquet")
    rewrite(path, change)
    failures = validate(out, mini_grand, expected_mini)
    assert failures
    assert any(path.name in f and column in f for f in failures), failures


def test_reordered_columns_are_reported(out, mini_grand, expected_mini):
    path = first_file(out, "data/submissions/test-task2-*.parquet")
    rewrite(path, make_columns_reordered)
    failures = validate(out, mini_grand, expected_mini)
    assert any(path.name in f and "order" in f for f in failures), failures


def test_list_child_names_do_not_matter(out, mini_grand, expected_mini):
    path = first_file(out, "data/submissions/test-task1-*.parquet")
    assert pq.read_schema(path).field("t1_errors").type.value_field.name == "element"
    rewrite(path, lambda t: t.cast(SCHEMAS["submissions"]), use_compliant_nested_type=False)
    assert pq.read_schema(path).field("t1_errors").type.value_field.name == "item"
    assert validate(out, mini_grand, expected_mini) == []


def test_schema_gate_checks_every_file_of_a_config(out, mini_grand, expected_mini):
    path = sorted(out.glob("data/pairs/*.parquet"))[-1]
    rewrite(path, lambda t: t.remove_column(0))
    failures = validate(out, mini_grand, expected_mini)
    assert any(path.name in f and "item_id" in f for f in failures), failures


# --- the round-trip gate -------------------------------------------------------------------------


def test_round_trip_catches_a_changed_score(out, mini_grand, expected_mini):
    path = first_file(out, "data/submissions/test-task2-*.parquet")
    table = pq.read_table(path)
    item_id = table.column("item_id")[0].as_py()
    set_cell(path, "t2_score", 0, table.column("t2_score")[0].as_py() + 1.0)
    failures = validate(out, mini_grand, expected_mini)
    assert len(failures) == 1, failures
    assert path.name in failures[0] and item_id in failures[0]


def test_round_trip_catches_a_changed_error_span(out, mini_grand, expected_mini):
    path = first_file(out, "data/submissions/test-task1-*.parquet")
    table = pq.read_table(path)
    row = next(i for i, errors in enumerate(table.column("t1_errors").to_pylist()) if errors)

    def change(table):
        errors = table.column("t1_errors").to_pylist()
        errors[row][0]["end"] += 1
        return set_values(table, "t1_errors", errors)

    rewrite(path, change)
    failures = validate(out, mini_grand, expected_mini)
    assert len(failures) == 1 and path.name in failures[0], failures


def test_round_trip_catches_an_error_span_turned_into_none(out, mini_grand, expected_mini):
    path = first_file(out, "data/submissions/test-task1-*.parquet")
    table = pq.read_table(path)
    row = next(i for i, errors in enumerate(table.column("t1_errors").to_pylist()) if errors)
    set_cell(path, "t1_errors", row, None)  # "no Task 1 prediction" instead of a prediction
    failures = validate(out, mini_grand, expected_mini)
    assert any(path.name in f for f in failures), failures


def test_the_order_of_rows_does_not_matter(out, mini_grand, expected_mini):
    for pattern in ("data/submissions/test-task1-*.parquet", "data/baselines/*-alt.parquet"):
        path = first_file(out, pattern)
        rewrite(path, lambda t: t.take(list(reversed(range(t.num_rows)))))
    assert validate(out, mini_grand, expected_mini) == []


def test_a_dropped_row_is_reported_as_a_count_and_as_a_lost_system(out, mini_grand, expected_mini):
    path = first_file(out, "data/submissions/test-task2-*.parquet")
    rewrite(path, lambda t: t.slice(1))
    failures = validate(out, mini_grand, expected_mini)
    assert any(f.startswith("submissions") for f in failures)
    assert any(path.name in f for f in failures)


def test_a_duplicated_row_is_reported(out, mini_grand, expected_mini):
    path = first_file(out, "data/submissions/test-task2-*.parquet")
    rewrite(path, lambda t: pa.concat_tables([t, t.slice(0, 1)]))
    failures = validate(out, mini_grand, expected_mini)
    assert any(path.name in f for f in failures), failures


def test_rows_for_an_item_the_run_does_not_cover_are_reported(out, mini_grand, expected_mini):
    path = first_file(out, "data/submissions/test-task2-*.parquet")

    def change(table):
        extra = set_values(table.slice(0, 1), "item_id", ["not_an_item"])
        return pa.concat_tables([table, extra])

    rewrite(path, change)
    failures = validate(out, mini_grand, expected_mini)
    assert any(path.name in f and "not_an_item" in f for f in failures), failures


def test_an_item_without_rows_is_reported(out, mini_grand, expected_mini):
    path = first_file(out, "data/submissions/test-task2-*.parquet")
    table = pq.read_table(path)
    item_id = table.column("item_id")[0].as_py()
    rewrite(path, lambda t: t.filter(pc.not_equal(t.column("item_id"), item_id)))
    failures = validate(out, mini_grand, expected_mini)
    assert any(path.name in f and item_id in f for f in failures), failures


def test_a_missing_run_file_is_reported(out, mini_grand, expected_mini):
    path = first_file(out, "data/baselines/*-official.parquet")
    path.unlink()
    failures = validate(out, mini_grand, expected_mini)
    assert any(path.name in f and "missing" in f for f in failures), failures


def test_a_file_that_belongs_to_no_run_is_reported(out, mini_grand, expected_mini):
    source = first_file(out, "data/submissions/test-task2-*.parquet")
    stray = source.with_name("test-task2-999999.parquet")
    shutil.copy(source, stray)
    failures = validate(out, mini_grand, expected_mini)
    assert any(stray.name in f for f in failures), failures


@pytest.mark.parametrize(
    ("pattern", "column"),
    [
        ("data/submissions/test-task2-*.parquet", "run_id"),
        ("data/submissions/test-task2-*.parquet", "task"),
        ("data/baselines/*-official.parquet", "variant"),
        ("data/baselines/*-alt.parquet", "variant"),
    ],
)
def test_run_labels_are_checked(out, mini_grand, expected_mini, pattern, column):
    path = first_file(out, pattern)
    rewrite(path, lambda t: set_values(t, column, ["wrong"] * t.num_rows))
    failures = validate(out, mini_grand, expected_mini)
    assert any(path.name in f and column in f for f in failures), failures


def test_an_alt_run_holds_task_1_only(out, mini_grand, expected_mini):
    path = first_file(out, "data/baselines/*-alt.parquet")
    assert set(pq.read_table(path).column("t2_score").to_pylist()) == {None}
    set_cell(path, "t2_score", 0, 1.0)
    failures = validate(out, mini_grand, expected_mini)
    assert any(path.name in f for f in failures), failures


def test_failures_per_run_are_capped(out, mini_grand, expected_mini):
    path = next(
        p
        for p in sorted(out.glob("data/submissions/test-task2-*.parquet"))
        if len(set(pq.read_table(p, columns=["item_id"]).column(0).to_pylist()))
        > MAX_ITEM_FAILURES + 1
    )
    rewrite(path, lambda t: set_values(t, "t2_score", [1e9] * t.num_rows))
    failures = [f for f in validate(out, mini_grand, expected_mini) if path.name in f]
    assert len(failures) == MAX_ITEM_FAILURES + 1
    assert "more" in failures[-1]


# --- the scripts ---------------------------------------------------------------------------------


def load_script(repo_root: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, repo_root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_validate_cli(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    script = repo_root / "scripts" / "validate_hf.py"
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def test_validate_cli_passes_a_good_build(out, mini_grand_path, repo_root, expected_mini, tmp_path):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(expected_mini), encoding="utf-8")
    done = run_validate_cli(
        repo_root,
        *("--out", str(out), "--grand", str(mini_grand_path), "--expected", str(expected)),
        "--no-checksum",
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip().endswith("0 failures")


def test_validate_cli_exits_1_and_prints_each_failure(
    out, mini_grand_path, repo_root, expected_mini, tmp_path
):
    set_cell(out / SYSTEMS, "description", 0, "ada@example.org")
    (out / "README.md").write_text("Email")
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(expected_mini), encoding="utf-8")
    done = run_validate_cli(
        repo_root,
        *("--out", str(out), "--grand", str(mini_grand_path), "--expected", str(expected)),
        "--no-checksum",
    )
    assert done.returncode == 1
    assert SYSTEMS in done.stdout and "description" in done.stdout
    assert done.stdout.strip().endswith("1 failure")
    assert "ada@example.org" not in done.stdout + done.stderr


def test_validate_cli_checks_the_grand_checksum_unless_told_not_to(
    out, mini_grand_path, repo_root, expected_mini, tmp_path
):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(expected_mini), encoding="utf-8")
    done = run_validate_cli(
        repo_root,
        *("--out", str(out), "--grand", str(mini_grand_path), "--expected", str(expected)),
    )
    assert done.returncode != 0 and "sha256" in done.stderr


def test_the_expected_full_file_has_every_config(repo_root, expected_full):
    assert set(expected_full["rows"]) == {
        "items",
        "pairs",
        "submissions",
        "baselines",
        "systems",
        "mt_systems",
        "challenge_sets",
    }


# --- check_loading.py (the datasets calls themselves run in Task 12's matrix) ---------------------

README = """---
configs:
- config_name: items
  data_files:
  - split: test
    path: data/items/*.parquet
- config_name: grand
  data_files:
  - split: test
    path: wmt26-grand.parquet
---

# WMT26 MT evaluation
"""


class FakeDatasets:
    """Stands in for the ``datasets`` module: loads succeed unless the config is in ``broken``."""

    def __init__(self, version="5.1.0", broken=None):
        self.__version__ = version
        self.broken = broken or {}
        self.calls = []

    def load_dataset(self, path, name=None, split=None, **kwargs):
        staging = Path(path)
        self.calls.append(
            {
                "path": staging,
                "name": name,
                "split": split,
                "readme": (staging / "README.md").is_file(),
                "grand": (staging / "wmt26-grand.parquet").is_file(),
                "cache_dir": kwargs.get("cache_dir"),
            }
        )
        if name in self.broken:
            raise self.broken[name]
        return range(7)


@pytest.fixture
def staged_out(out) -> Path:
    (out / "README.md").write_text(README, encoding="utf-8")
    return out


def test_config_names_come_from_the_yaml_front_matter(repo_root):
    check_loading = load_script(repo_root, "check_loading")
    assert check_loading.config_names(README) == ["items", "grand"]
    with pytest.raises(ValueError, match="front matter"):
        check_loading.config_names("# no front matter\n")
    with pytest.raises(ValueError, match="configs"):
        check_loading.config_names("---\nlicense: cc\n---\n")
    with pytest.raises(ValueError, match="config_name"):
        check_loading.config_names("---\nconfigs:\n- data_files: x\n---\n")


def staged_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_staging_mirrors_the_output_with_links_and_adds_the_grand_file(
    staged_out, mini_grand_path, repo_root, tmp_path
):
    check_loading = load_script(repo_root, "check_loading")
    staging = tmp_path / "staging"
    check_loading.stage(staged_out, mini_grand_path, staging)
    files = staged_files(staged_out)
    assert staged_files(staging) == files | {"wmt26-grand.parquet"}
    assert (staging / "data").is_dir() and not (staging / "data").is_symlink()
    for relative in files:  # hard links: datasets 2.18.0 does not see symlinked files
        assert not (staging / relative).is_symlink()
        assert (staging / relative).samefile(staged_out / relative)
    assert not (staging / "wmt26-grand.parquet").is_symlink()
    assert (staging / "wmt26-grand.parquet").samefile(mini_grand_path)


def test_staging_copies_where_hard_links_are_impossible(
    staged_out, mini_grand_path, repo_root, tmp_path, monkeypatch
):
    check_loading = load_script(repo_root, "check_loading")

    def no_links(source, target):
        raise OSError("Invalid cross-device link")

    monkeypatch.setattr(check_loading.os, "link", no_links)
    staging = tmp_path / "staging"
    check_loading.stage(staged_out, mini_grand_path, staging)
    assert staged_files(staging) == staged_files(staged_out) | {"wmt26-grand.parquet"}
    for relative in staged_files(staged_out):
        assert not (staging / relative).samefile(staged_out / relative)
        assert (staging / relative).read_bytes() == (staged_out / relative).read_bytes()
    assert (staging / "wmt26-grand.parquet").read_bytes() == mini_grand_path.read_bytes()


def test_check_loading_prints_one_line_per_config(
    staged_out, mini_grand_path, repo_root, monkeypatch, capsys
):
    check_loading = load_script(repo_root, "check_loading")
    fake = FakeDatasets()
    monkeypatch.setitem(sys.modules, "datasets", fake)
    argv = [str(staged_out), "5.1.0", "--grand", str(mini_grand_path)]
    assert check_loading.main(argv) == 0
    assert capsys.readouterr().out.splitlines() == ["OK 5.1.0 items 7", "OK 5.1.0 grand 7"]
    assert [(c["name"], c["split"], c["readme"], c["grand"]) for c in fake.calls] == [
        ("items", "test", True, True),
        ("grand", "test", True, True),
    ]
    for call in fake.calls:  # a scratch cache beside the staging directory, both gone afterwards
        assert Path(call["cache_dir"]).parent == call["path"].parent
        assert not call["path"].parent.exists()


def test_check_loading_fails_on_any_config_but_grand(
    staged_out, mini_grand_path, repo_root, monkeypatch, capsys
):
    check_loading = load_script(repo_root, "check_loading")
    argv = [str(staged_out), "2.18.0", "--grand", str(mini_grand_path)]

    grand_only = FakeDatasets("2.18.0", {"grand": ValueError("Feature type 'Json' not found")})
    monkeypatch.setitem(sys.modules, "datasets", grand_only)
    assert check_loading.main(argv) == 0
    assert capsys.readouterr().out.splitlines() == [
        "OK 2.18.0 items 7",
        "FAIL 2.18.0 grand ValueError: Feature type 'Json' not found",
    ]

    items_broken = FakeDatasets("2.18.0", {"items": ValueError("boom\nsecond line")})
    monkeypatch.setitem(sys.modules, "datasets", items_broken)
    assert check_loading.main(argv) == 1
    assert capsys.readouterr().out.splitlines() == [
        "FAIL 2.18.0 items ValueError: boom",  # the first line of the message only
        "OK 2.18.0 grand 7",
    ]


def test_check_loading_refuses_a_version_label_that_is_not_the_installed_one(
    staged_out, mini_grand_path, repo_root, monkeypatch, capsys
):
    check_loading = load_script(repo_root, "check_loading")
    monkeypatch.setitem(sys.modules, "datasets", FakeDatasets("4.0.0"))
    argv = [str(staged_out), "5.1.0", "--grand", str(mini_grand_path)]
    assert check_loading.main(argv) == 2
    assert "4.0.0" in capsys.readouterr().err


def test_check_loading_needs_a_readme(out, mini_grand_path, repo_root, monkeypatch, capsys):
    check_loading = load_script(repo_root, "check_loading")
    monkeypatch.setitem(sys.modules, "datasets", FakeDatasets())
    assert check_loading.main([str(out), "5.1.0", "--grand", str(mini_grand_path)]) == 2
    assert "README.md" in capsys.readouterr().err
