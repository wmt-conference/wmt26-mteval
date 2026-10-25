import csv
import importlib.util
import json
import os
from pathlib import Path

import pytest

from wmt26_mteval.cards import (
    ANSWER_OVERRIDES,
    CARD_FIELDS,
    EMAIL_RE,
    EMAIL_REMOVED,
    NAMES_REPLACED,
    card_source,
    normalise_bool,
    parse_threshold,
    scrub_card,
    scrub_card_report,
    scrub_runs,
)
from wmt26_mteval.grand import Grand
from wmt26_mteval.runs import RunSpec, list_runs

RUN_NAMES_HEADER = ["run_id", "canonical_name", "underlying_model", "paper_citation"]
KEYS = set(CARD_FIELDS.values())


def run_spec(run_id, kind="submission", task="task1", variant="official"):
    return RunSpec(
        run_id=run_id,
        kind=kind,
        task=task,
        variant=variant,
        column=f"{kind}__{task}__{run_id}__predictions",
        source_path=f"{kind}s/{task}/{run_id}/predictions.jsonl",
    )


def write_card(path: Path, name="Example-Metric") -> Path:
    """A minimal synthetic raw card at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    card = [
        {"question": "Participant name", "required": True, "answer": "Ada Example"},
        {"question": "Submission name", "required": True, "answer": name},
        {"question": "Email", "required": True, "answer": "ada@example.org"},
    ]
    path.write_text(json.dumps(card), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def raw_root(repo_root) -> Path:
    return repo_root / "tests" / "fixtures" / "cards" / "raw"


def load_fixture(raw_root, task, run_id):
    path = raw_root / "submissions" / task / run_id / "model_card.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def raw_card_with_pii(raw_root):
    return load_fixture(raw_root, "task1", "111111")


# --- scrub_card ---------------------------------------------------------------------------------


def test_scrub_drops_pii(raw_card_with_pii):
    out = scrub_card(raw_card_with_pii)
    blob = json.dumps(out)
    assert "ada@example.org" not in blob and "Ada Example" not in blob
    assert "Data Collection Consent" not in blob
    assert set(out) <= set(CARD_FIELDS.values())


def test_scrub_keeps_every_allow_listed_answer_as_a_raw_string(raw_card_with_pii):
    assert scrub_card(raw_card_with_pii) == {
        "org": "Example University",
        "name": "Example-QE-1",
        "description": "A synthetic reference-based metric that exists only in the test suite.",
        "uses_reference": "True",
        "reference_usage": "Uses the native references to rerank candidates.",
        "ensemble": "No",
        "fine_tuned": "true",
        "open_weight": "True",
        "model_size": "7B",
        "closed_model_name": "",
        "api_provider": "",
        "open_source": "No",
        "link": "https://example.org/example-qe-1",
        "allow_t1_to_t2": "true",
        "allow_t1_to_t3": "false",
        "allow_t2_to_t3": "true",
        "t3_threshold": "85",
        "threshold_inclusive": "inclusive",
    }


def test_card_fields_cover_the_brief_and_prefixes_do_not_overlap():
    assert len(CARD_FIELDS) == 18 and len(KEYS) == 18
    prefixes = list(CARD_FIELDS)
    for a in prefixes:
        assert [b for b in prefixes if b != a and b.startswith(a)] == []


def test_every_question_of_the_blank_card_is_decided(repo_root):
    """Each question of the committed template is kept under one key, or is one of the PII three."""
    blank = json.loads((repo_root / "model_card.json").read_text(encoding="utf-8"))
    kept = {}
    for item in blank:
        keys = [key for prefix, key in CARD_FIELDS.items() if item["question"].startswith(prefix)]
        assert len(keys) <= 1, item["question"]
        if keys:
            kept[keys[0]] = item["question"]
        else:
            assert item["question"].startswith(("Participant name", "Email", "Data Collection"))
    assert set(kept) == KEYS


def test_unlisted_questions_are_dropped_without_reading_their_answers():
    raw = [
        {"question": "Email"},  # no "answer" key at all: it must never be looked up
        {"question": "A question nobody listed", "required": False},
        {"question": "Submission name", "required": True, "answer": "Example"},
    ]
    assert scrub_card(raw) == {"name": "Example"}


def test_json_booleans_become_strings():
    raw = [
        {"question": "Is an ensemble (true|false)", "required": True, "answer": False},
        {"question": "Is the submitted system open-source? (true|false)", "answer": True},
    ]
    out = scrub_card(raw)
    assert out == {"ensemble": "false", "open_source": "true"}
    assert [normalise_bool(v) for v in out.values()] == [False, True]


def test_non_text_answers_other_than_booleans_fail_loudly():
    with pytest.raises(TypeError, match="model_size"):
        scrub_card([{"question": "Model size", "answer": 7}])


def test_repeated_question_fails_loudly():
    raw = [{"question": "Model size", "answer": "1B"}, {"question": "Model size", "answer": "2B"}]
    with pytest.raises(ValueError, match="model_size"):
        scrub_card(raw)


def test_email_addresses_inside_kept_answers_are_replaced(raw_root):
    raw = load_fixture(raw_root, "task3", "222222")
    out, redacted = scrub_card_report(raw)
    assert redacted == ["description", "link"]
    assert out["description"] == (
        "A synthetic LLM judge. Questions go to [email removed], the address on this fake card."
    )
    assert out["link"] == "see [email removed] or https://example.org/example-judge"
    assert "@" not in json.dumps(out)
    assert scrub_card(raw) == out


def test_clean_cards_report_no_redactions(raw_card_with_pii):
    assert scrub_card_report(raw_card_with_pii)[1] == []


@pytest.mark.parametrize(
    "text, found",
    [
        ("write to ada@example.org now", True),
        ("a@example.org", True),
        ("@handle on a forum", False),
    ],
)
def test_email_re(text, found):
    assert bool(EMAIL_RE.search(text)) is found


# --- typing helpers -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s,v",
    [("True", True), ("true", True), ("No", False), ("false", False), ("", None), ("maybe", None)],
)
def test_bool(s, v):
    assert normalise_bool(s) is v


@pytest.mark.parametrize("s,v", [("Yes", True), ("yes", True), ("False", False), ("no", False)])
def test_bool_spellings(s, v):
    assert normalise_bool(s) is v


@pytest.mark.parametrize(
    "t,i,out",
    [
        ("85", "inclusive", (85.0, True)),
        ("true", "inclusive", (None, True)),
        ("", "≥", (None, True)),
        ("", "inclusive (≥)", (None, True)),
        ("", "exclusive", (None, False)),
        ("", ">", (None, False)),
        ("", "", (None, None)),
    ],
)
def test_threshold(t, i, out):
    assert parse_threshold(t, i) == out


@pytest.mark.parametrize(
    "t,i,out",
    [
        (" 0.5 ", " Inclusive ", (0.5, True)),
        ("-12", "exclusive (>)", (-12.0, False)),
        ("nan", "", (None, None)),
        ("inf", "", (None, None)),
        ("", ">=", (None, True)),
        ("", "inclusive (≥) or exclusive (>)", (None, None)),
    ],
)
def test_threshold_edge_cases(t, i, out):
    assert parse_threshold(t, i) == out


@pytest.mark.parametrize(
    "task, run_id, uses_reference, ensemble, threshold",
    [
        ("task1", "111111", True, False, (85.0, True)),
        ("task3", "222222", False, True, (None, True)),  # threshold "true", a misfilled answer
        ("task2", "333333", False, False, (None, True)),
        ("task2", "444444", False, False, (None, False)),
    ],
)
def test_fixture_cards_type_cleanly(raw_root, task, run_id, uses_reference, ensemble, threshold):
    answers = scrub_card(load_fixture(raw_root, task, run_id))
    assert normalise_bool(answers["uses_reference"]) is uses_reference
    assert normalise_bool(answers["ensemble"]) is ensemble
    assert parse_threshold(answers["t3_threshold"], answers["threshold_inclusive"]) == threshold


# --- card_source and scrub_runs -----------------------------------------------------------------


def test_card_source_submission(tmp_path):
    card = write_card(tmp_path / "submissions" / "task2" / "333333" / "model_card.json")
    assert card_source(tmp_path, run_spec("333333", task="task2")) == card


def test_card_source_prefers_the_task2_card_for_llm_baselines(tmp_path):
    t1 = write_card(tmp_path / "baselines" / "task1" / "gemma_task1" / "model_card.json")
    t2 = write_card(tmp_path / "baselines" / "task2" / "gemma_task2" / "model_card.json")
    run = run_spec("gemma", kind="baseline")
    assert card_source(tmp_path, run) == t2
    t2.unlink()
    assert card_source(tmp_path, run) == t1


def test_card_source_numeric_baseline(tmp_path):
    card = write_card(tmp_path / "baselines" / "task2" / "872013" / "model_card.json")
    assert card_source(tmp_path, run_spec("872013", kind="baseline", task="task2")) == card


def test_card_source_ignores_macosx_junk(tmp_path):
    write_card(tmp_path / "submissions" / "task1" / "111111" / "__MACOSX" / "model_card.json")
    with pytest.raises(FileNotFoundError, match="111111"):
        card_source(tmp_path, run_spec("111111"))


@pytest.mark.parametrize(
    "run",
    [run_spec("999999"), run_spec("nothing", kind="baseline"), run_spec("872013", "baseline")],
)
def test_card_source_missing_card_fails_loudly(tmp_path, run):
    with pytest.raises(FileNotFoundError, match=run.run_id):
        card_source(tmp_path, run)


def test_scrub_runs_writes_one_deterministic_file_per_official_run(tmp_path, raw_root):
    runs = [
        run_spec("111111", task="task1"),
        run_spec("222222", task="task3"),
        run_spec("333333", task="task2", variant="alt"),  # alt runs get no card of their own
    ]
    out = tmp_path / "out"
    changes = scrub_runs(raw_root, runs, out)
    assert sorted(p.name for p in out.iterdir()) == ["111111.json", "222222.json"]
    assert changes == [
        ("222222", "description", EMAIL_REMOVED),
        ("222222", "link", EMAIL_REMOVED),
    ]

    text = (out / "111111.json").read_text(encoding="utf-8")
    assert text.endswith("}\n") and not text.endswith("}\n\n")
    doc = json.loads(text)
    assert list(doc) == ["answers", "run_id"] and doc["run_id"] == "111111"
    assert list(doc["answers"]) == sorted(doc["answers"])
    assert text == json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    assert "Ada" not in text and "@" not in text

    again = tmp_path / "again"
    scrub_runs(raw_root, runs, again)
    assert (again / "222222.json").read_bytes() == (out / "222222.json").read_bytes()


def test_scrub_runs_applies_and_reports_answer_overrides(tmp_path, raw_root):
    overrides = {("111111", "org"): "Replaced Org", ("222222", "no_such_key"): "ignored"}
    out = tmp_path / "out"
    changes = scrub_runs(raw_root, [run_spec("111111")], out, overrides=overrides)
    assert changes == [("111111", "org", NAMES_REPLACED)]
    assert json.loads((out / "111111.json").read_text(encoding="utf-8"))["answers"]["org"] == (
        "Replaced Org"
    )
    # Only runs being written are touched, and a key the card lacks is not invented.
    assert scrub_runs(raw_root, [run_spec("222222", task="task3")], out, overrides=overrides) == [
        ("222222", "description", EMAIL_REMOVED),
        ("222222", "link", EMAIL_REMOVED),
    ]
    assert "no_such_key" not in (out / "222222.json").read_text(encoding="utf-8")


def test_answer_overrides_only_name_organisations_for_the_org_field():
    assert {key for _, key in ANSWER_OVERRIDES} == {"org"}
    for text in ANSWER_OVERRIDES.values():
        assert "(" not in text and not EMAIL_RE.search(text)


def test_scrub_runs_keeps_non_ascii_text_unescaped(tmp_path):
    card = [{"question": "Submission name", "answer": "Métrique ≥ 1"}]
    path = tmp_path / "raw" / "submissions" / "task1" / "111111" / "model_card.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(card), encoding="utf-8")
    scrub_runs(tmp_path / "raw", [run_spec("111111")], tmp_path / "out")
    assert "Métrique ≥ 1" in (tmp_path / "out" / "111111.json").read_text(encoding="utf-8")


def test_scrub_runs_fails_loudly_when_a_card_is_missing(tmp_path, raw_root):
    with pytest.raises(FileNotFoundError, match="555555"):
        scrub_runs(raw_root, [run_spec("111111"), run_spec("555555")], tmp_path / "out")


def test_scrub_runs_rejects_two_official_runs_with_one_id(tmp_path, raw_root):
    runs = [run_spec("111111"), run_spec("111111", task="task2")]
    with pytest.raises(ValueError, match="111111"):
        scrub_runs(raw_root, runs, tmp_path / "out")


# --- committed data -----------------------------------------------------------------------------


def read_run_names(repo_root):
    path = repo_root / "data" / "metadata" / "run_names.tsv"
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == RUN_NAMES_HEADER
        return list(reader)


def test_run_names_complete(repo_root):
    rows = list(csv.DictReader(open(repo_root / "data/metadata/run_names.tsv"), delimiter="\t"))
    assert len(rows) == 73 and len({r["run_id"] for r in rows}) == 73
    assert all(r["canonical_name"] for r in rows)


def test_run_names_rows_are_complete_and_sorted(repo_root):
    rows = read_run_names(repo_root)
    assert [r["run_id"] for r in rows] == sorted(r["run_id"] for r in rows)
    for row in rows:
        assert all(row[c] == row[c].strip() and row[c] for c in RUN_NAMES_HEADER), row["run_id"]


def test_run_names_cover_exactly_the_official_runs(repo_root, mini_grand_path):
    official = {
        r.run_id
        for r in list_runs(Grand(mini_grand_path, expected_sha256=None))
        if r.variant == "official"
    }
    assert {r["run_id"] for r in read_run_names(repo_root)} == official


def test_run_names_baselines_follow_table_8_and_citations_follow_table_3(repo_root):
    by_id = {r["run_id"]: r for r in read_run_names(repo_root)}
    assert by_id["gemma"]["canonical_name"] == "Gemma 4 - Reasoning"
    assert by_id["gemma"]["paper_citation"] == "Lavie et al., 2026 (findings paper)"
    assert by_id["xcomet_xl_ref"]["canonical_name"] == "xCOMET XL (reference-based)"
    assert by_id["xcomet_xl_no_ref"]["canonical_name"] == "xCOMET XL (reference-free)"
    assert (
        by_id["872028"]["canonical_name"] == "BLEU" and by_id["872028"]["underlying_model"] == "—"
    )
    assert by_id["872030"]["underlying_model"] == "XLM-R-large"
    assert by_id["869556"]["canonical_name"] == by_id["869557"]["canonical_name"] == "FACET"
    assert by_id["869556"]["paper_citation"] == "Kim et al., 2026"


def test_committed_card_of_the_aegis_run_names_only_organisations(repo_root):
    doc = json.loads((repo_root / "data/model_cards/871298.json").read_text(encoding="utf-8"))
    assert doc["answers"]["org"] == "IIT Patna; Shiv Nadar University"


def test_committed_cards_clean(repo_root):
    for p in (repo_root / "data/model_cards").glob("*.json"):
        txt = p.read_text()
        assert not EMAIL_RE.search(txt) and "Participant name" not in txt, p.name


def test_committed_cards_are_scrubbed_allow_list_documents(repo_root, mini_grand_path):
    official = {
        r.run_id
        for r in list_runs(Grand(mini_grand_path, expected_sha256=None))
        if r.variant == "official"
    }
    files = sorted((repo_root / "data" / "model_cards").glob("*.json"))
    assert {p.stem for p in files} == official and len(files) == 73
    for path in files:
        text = path.read_text(encoding="utf-8")
        doc = json.loads(text)
        assert list(doc) == ["answers", "run_id"] and doc["run_id"] == path.stem
        assert set(doc["answers"]) <= KEYS and all(
            isinstance(v, str) for v in doc["answers"].values()
        )
        assert text == json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        for banned in ("Participant name", "Data Collection Consent", '"Email"', "@"):
            assert banned not in text, (path.name, banned)


@pytest.fixture(scope="module")
def real_inputs():
    raw, grand = os.environ.get("RAW"), os.environ.get("GRAND")
    if not raw or not grand or not Path(raw).is_dir() or not Path(grand).is_file():
        pytest.skip("RAW and GRAND are not set to the organiser drop and the real grand file")
    return Path(raw), Path(grand)


@pytest.mark.fulldata
def test_cli_reproduces_the_committed_cards(repo_root, real_inputs, tmp_path):
    raw, grand = real_inputs
    spec = importlib.util.spec_from_file_location(
        "scrub_model_cards", repo_root / "scripts" / "scrub_model_cards.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "model_cards"
    assert module.main(["--raw", str(raw), "--grand", str(grand), "--out", str(out)]) == 0
    committed = sorted((repo_root / "data" / "model_cards").glob("*.json"))
    assert [p.name for p in sorted(out.glob("*.json"))] == [p.name for p in committed]
    for path in committed:
        assert (out / path.name).read_bytes() == path.read_bytes(), path.name
