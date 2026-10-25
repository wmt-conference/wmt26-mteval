import json

import pyarrow as pa
import pytest
import yaml

from wmt26_mteval.cards import CARD_FIELDS, EMAIL_RE
from wmt26_mteval.runs import RunSpec
from wmt26_mteval.schemas import SCHEMAS
from wmt26_mteval.tables.metadata import (
    Coverage,
    build_challenge_sets,
    build_mt_systems,
    build_systems,
    coverage_of,
    priority_lps,
)

CHALLENGE_SETS = [
    "AGGREE",
    "FECE",
    "GAMBITP",
    "LIGATUR",
    "QEbreak",
    "RefQuality-PD",
    "RefQuality-PE",
]
PRIORITY = {"ces-deu", "ces-vie", "eng-arz", "eng-ekk", "eng-ind", "eng-isl", "eng-sme", "zho-jpn"}
RUN_NAMES_HEADER = "run_id\tcanonical_name\tunderlying_model\tpaper_citation"

# Titles exactly as the reference list of the findings paper (2026.wmt-1.49) prints them.
PAPER_TITLES = {
    "AGGREE": (
        "AGGREE: An aggregation-grounded challenge set for segment-level MT metric evaluation"
    ),
    "FECE": "FECE: A challenge set for evaluating fine-grained translation error correction",
    "GAMBITP": (
        "Benchmarking gender bias in machine translation evaluation metrics across occupations"
    ),
    "LIGATUR": (
        "Translation metrics cannot judge what their tokeniser deletes: LIGATUR and AEGIS "
        "submissions to the WMT26 shared task on automated translation quality evaluation"
    ),
    "QEbreak": (
        "QEbreak at WMT26: A preregistered audit of omission and addition asymmetry in machine "
        "translation quality metrics"
    ),
}


def official(lp_src, lp_tgt, n):
    return f"official_###_{lp_src}_###_{lp_tgt}_###_news_###_doc_###_{n}"


def challenge(set_name, n):
    return f"challenge_###_eng_Latn_###_deu_Latn_###_{set_name}_###_doc_###_{n}"


def spec(run_id, kind="submission", task="task2", variant="official"):
    return RunSpec(
        run_id=run_id,
        kind=kind,
        task=task,
        variant=variant,
        column=f"{kind}__{task}__{run_id}__{variant}",
        source_path=f"{kind}s/{task}/{run_id}/predictions.jsonl",
    )


# --- priority_lps and coverage_of ----------------------------------------------------------------


def test_priority_lps_are_the_short_codes_of_the_eight_directions(ctx):
    assert priority_lps(ctx) == PRIORITY


def test_coverage_of_counts_items_rows_and_languages():
    table = pa.table(
        {
            "item_id": [official("a", "b", 0)] * 2 + [official("a", "b", 1), challenge("FECE", 0)],
            "lp": ["ces-deu", "ces-deu", "ces-deu", "eng-deu"],
        }
    )
    assert coverage_of(table, {"ces-deu"}) == Coverage(
        n_items=3, n_pairs=4, lps=["ces-deu", "eng-deu"], covers_all_priority=True
    )


def test_coverage_requires_every_priority_pair_among_the_official_rows():
    items = [official("a", "b", 0), challenge("RefQuality-PD", 0)]
    only_challenge = pa.table({"item_id": items, "lp": ["ces-deu", "eng-isl"]})
    assert coverage_of(only_challenge, {"ces-deu", "eng-isl"}).covers_all_priority is False
    items.append(official("a", "b", 1))
    both = pa.table({"item_id": items, "lp": ["ces-deu", "eng-isl", "eng-isl"]})
    assert coverage_of(both, {"ces-deu", "eng-isl"}).covers_all_priority is True


def test_coverage_of_a_run_without_rows():
    empty = pa.table({"item_id": pa.array([], pa.string()), "lp": pa.array([], pa.string())})
    assert coverage_of(empty, PRIORITY) == Coverage(0, 0, [], False)


# --- systems: the mini fixture and the committed cards -------------------------------------------


def test_systems_names_unique(mini_systems):  # mini-grand runs + the committed scrubbed cards
    names = mini_systems.column("name").to_pylist()
    assert len(names) == len(set(names))
    assert mini_systems.schema.equals(SCHEMAS["systems"], check_metadata=False)


def test_systems_schema_carries_the_field_descriptions(mini_systems):
    assert mini_systems.schema.equals(SCHEMAS["systems"], check_metadata=True)
    assert all(f.metadata and b"description" in f.metadata for f in mini_systems.schema)


def test_threshold_typed(mini_systems):
    assert mini_systems.schema.field("t3_threshold").type == pa.float64()
    assert mini_systems.schema.field("threshold_inclusive").type == pa.bool_()


def test_systems_has_one_row_per_official_run_sorted_by_run_id(mini_systems):
    run_ids = mini_systems.column("run_id").to_pylist()
    assert len(run_ids) == 73 == len(set(run_ids))
    assert run_ids == sorted(run_ids)
    flags = mini_systems.column("is_baseline").to_pylist()
    assert (sum(flags), len(flags) - sum(flags)) == (23, 50)


def test_systems_tasks(mini_systems):
    tasks = {r["run_id"]: r["tasks"] for r in mini_systems.to_pylist()}
    assert tasks["875921"] == ["task1"]  # a Task 1 submission
    assert tasks["872032"] == ["task2"]  # a Task 2 submission
    assert tasks["872036"] == ["task3"]  # a Task 3 submission
    assert tasks["gemma_ref"] == ["task1", "task2"]  # an LLM baseline
    assert tasks["xcomet_xl_ref"] == ["task1", "task2"]  # an xCOMET baseline
    assert tasks["872028"] == ["task2"]  # a numeric baseline (BLEU)


def test_systems_name_is_suffixed_only_where_the_card_name_is_shared(mini_systems):
    by_id = {r["run_id"]: r for r in mini_systems.to_pylist()}
    assert by_id["869543"]["name"] == "FACET (869543)"
    assert by_id["869556"]["name"] == "FACET (869556)"
    assert by_id["867652"]["name"] == "Pterodactyl"
    assert by_id["869543"]["canonical_name"] == "FACET"


def test_systems_takes_the_curated_columns_from_run_names(mini_systems):
    row = next(r for r in mini_systems.to_pylist() if r["run_id"] == "867652")
    assert (row["canonical_name"], row["underlying_model"], row["paper_citation"]) == (
        "Pterodactyl",
        "GPT-5.4",
        "Garland and Berger, 2026",
    )
    assert row["is_baseline"] is False
    assert row["org"] == "1020 Group" and row["closed_model_name"] == "GPT-5.4"
    assert row["ensemble"] is False and row["threshold_inclusive"] is True


def test_systems_coverage_columns_are_consistent_with_the_run_tables(mini_systems):
    for row in mini_systems.to_pylist():
        assert (row["n_items"] == 0) == (row["n_pairs"] == 0) == (row["lps"] == [])
        assert row["n_pairs"] >= row["n_items"]
        assert row["lps"] == sorted(row["lps"])
        assert row["covers_all_priority"] is False  # 16 items cannot cover eight directions
    assert sum(r["n_pairs"] for r in mini_systems.to_pylist()) > 0


def test_systems_never_holds_personal_data(mini_systems):
    assert not {"participant_name", "email", "consent"} & set(mini_systems.column_names)
    for field in mini_systems.schema:
        if field.type == pa.string():
            for text in mini_systems.column(field.name).to_pylist():
                assert text is None or not EMAIL_RE.search(text), field.name


# --- systems: synthetic repository ---------------------------------------------------------------


def write_repo(tmp_path, cards, names):
    """A repository skeleton with the given scrubbed cards and ``run_names.tsv`` rows."""
    (tmp_path / "data" / "model_cards").mkdir(parents=True)
    (tmp_path / "data" / "metadata").mkdir(parents=True)
    for run_id, answers in cards.items():
        text = json.dumps({"run_id": run_id, "answers": answers})
        (tmp_path / "data" / "model_cards" / f"{run_id}.json").write_text(text, encoding="utf-8")
    rows = [RUN_NAMES_HEADER] + ["\t".join(row) for row in names]
    (tmp_path / "data" / "metadata" / "run_names.tsv").write_text("\n".join(rows) + "\n")
    return tmp_path


@pytest.fixture
def synthetic():
    runs = [
        spec("gemma", "baseline", "task1"),
        spec("gemma", "baseline", "task1", variant="alt"),
        spec("872028", "baseline", "task2"),
        spec("333", task="task3"),
        spec("111", task="task2"),
        spec("222", task="task1"),
    ]
    cards = {
        "111": {
            "name": "Solo",
            "org": "Org A",
            "description": "Does X.",
            "uses_reference": "Yes",
            "ensemble": "false",
            "fine_tuned": "maybe",
            "open_weight": "TRUE",
            "link": "https://example.org/solo",
            "allow_t1_to_t2": "no",
            "t3_threshold": "85.5",
            "threshold_inclusive": "inclusive",
        },
        "222": {"name": "Twin", "org": "Org B"},
        "333": {"name": "Twin", "org": "Org C", "t3_threshold": "n/a", "threshold_inclusive": ""},
        "gemma": {
            "name": "gemma",
            "org": "Organisers",
            "t3_threshold": "0",
            "threshold_inclusive": "exclusive",
        },
        "872028": {"name": "BLEU", "link": ""},
    }
    names = [
        ("111", "Solo", "GPT-5.4", "Somebody, 2026"),
        ("222", "Twin", "Gemma 4 31B", "Somebody, 2026"),
        ("333", "Twin", "Gemma 4 31B", "Somebody, 2026"),
        ("gemma", "Gemma 4", "Gemma 4 31B IT", "Lavie et al., 2026 (findings paper)"),
        ("872028", "BLEU", "—", "Lavie et al., 2026 (findings paper)"),
    ]
    coverage = {
        "111": Coverage(5, 7, ["ces-deu", "eng-deu"], True),
        "222": Coverage(0, 0, [], False),
        "333": Coverage(1, 1, ["eng-deu"], False),
        "gemma": Coverage(9, 20, ["ces-deu"], False),
        "872028": Coverage(3, 3, ["eng-deu"], False),
    }
    return runs, cards, names, coverage


def rows_of(table):
    return {r["run_id"]: r for r in table.to_pylist()}


def test_build_systems_types_and_orders_the_rows(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    table = build_systems(write_repo(tmp_path, cards, names), runs, coverage)
    assert table.schema.equals(SCHEMAS["systems"], check_metadata=True)
    assert table.column("run_id").to_pylist() == ["111", "222", "333", "872028", "gemma"]  # no alt
    by_id = rows_of(table)
    assert [by_id[i]["name"] for i in ("111", "222", "333", "872028", "gemma")] == [
        "Solo",
        "Twin (222)",
        "Twin (333)",
        "BLEU",
        "gemma",
    ]
    solo = by_id["111"]
    assert solo["org"] == "Org A" and solo["description"] == "Does X."
    assert (solo["uses_reference"], solo["ensemble"], solo["open_weight"]) == (True, False, True)
    assert solo["fine_tuned"] is None  # unparseable
    assert solo["open_source"] is None and solo["allow_t1_to_t3"] is None  # key missing
    assert solo["allow_t1_to_t2"] is False
    assert (solo["t3_threshold"], solo["threshold_inclusive"]) == (85.5, True)
    assert solo["reference_usage"] is None and solo["model_size"] is None  # free text missing
    assert solo["link"] == "https://example.org/solo"
    assert (solo["canonical_name"], solo["underlying_model"]) == ("Solo", "GPT-5.4")
    assert solo["paper_citation"] == "Somebody, 2026"


def test_build_systems_threshold_variants(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    by_id = rows_of(build_systems(write_repo(tmp_path, cards, names), runs, coverage))
    assert (by_id["gemma"]["t3_threshold"], by_id["gemma"]["threshold_inclusive"]) == (0.0, False)
    assert (by_id["333"]["t3_threshold"], by_id["333"]["threshold_inclusive"]) == (None, None)
    assert (by_id["872028"]["t3_threshold"], by_id["872028"]["threshold_inclusive"]) == (None, None)


def test_build_systems_free_text_is_copied_verbatim(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    by_id = rows_of(build_systems(write_repo(tmp_path, cards, names), runs, coverage))
    assert by_id["872028"]["link"] == ""  # an empty answer stays empty; only a missing one is null
    assert by_id["872028"]["org"] is None


def test_build_systems_tasks_and_baseline_flag(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    by_id = rows_of(build_systems(write_repo(tmp_path, cards, names), runs, coverage))
    assert by_id["111"]["tasks"] == ["task2"] and by_id["333"]["tasks"] == ["task3"]
    assert by_id["gemma"]["tasks"] == ["task1", "task2"]
    assert by_id["872028"]["tasks"] == ["task2"]
    assert [by_id[i]["is_baseline"] for i in ("111", "gemma", "872028")] == [False, True, True]


def test_build_systems_copies_the_coverage(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    by_id = rows_of(build_systems(write_repo(tmp_path, cards, names), runs, coverage))
    assert by_id["111"]["lps"] == ["ces-deu", "eng-deu"]
    assert (by_id["111"]["n_items"], by_id["111"]["n_pairs"]) == (5, 7)
    assert (
        by_id["111"]["covers_all_priority"] is True and by_id["222"]["covers_all_priority"] is False
    )
    assert (by_id["222"]["n_items"], by_id["222"]["lps"]) == (0, [])


def test_build_systems_fails_loudly_without_a_card(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    del cards["222"]
    with pytest.raises(FileNotFoundError, match="222"):
        build_systems(write_repo(tmp_path, cards, names), runs, coverage)


def test_build_systems_fails_loudly_without_a_run_names_row(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    names = [row for row in names if row[0] != "333"]
    with pytest.raises(ValueError, match="333.*run_names"):
        build_systems(write_repo(tmp_path, cards, names), runs, coverage)


def test_build_systems_fails_loudly_without_coverage(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    del coverage["gemma"]
    with pytest.raises(ValueError, match="gemma.*coverage"):
        build_systems(write_repo(tmp_path, cards, names), runs, coverage)


def test_build_systems_rejects_a_card_of_another_run(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    repo = write_repo(tmp_path, cards, names)
    (repo / "data" / "model_cards" / "111.json").write_text(
        json.dumps({"run_id": "999", "answers": {"name": "Solo"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="111"):
        build_systems(repo, runs, coverage)


def test_build_systems_rejects_two_official_runs_with_one_id(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    with pytest.raises(ValueError, match="111"):
        build_systems(write_repo(tmp_path, cards, names), [*runs, spec("111")], coverage)


def test_build_systems_rejects_names_that_stay_ambiguous(tmp_path, synthetic):
    runs, cards, names, coverage = synthetic
    cards["333"]["name"] = "Twin (222)"
    cards["111"]["name"] = "Twin"
    with pytest.raises(ValueError, match="Twin"):
        build_systems(write_repo(tmp_path, cards, names), runs, coverage)


def test_card_keys_used_by_the_builder_are_the_scrubber_keys():
    assert {"name", "t3_threshold", "threshold_inclusive"} <= set(CARD_FIELDS.values())


# --- mt_systems ----------------------------------------------------------------------------------


def test_mt_systems(mini_pairs):
    t = build_mt_systems(mini_pairs)
    d = dict(zip(t.column("system").to_pylist(), t.column("system_type").to_pylist(), strict=True))
    assert d["PseudoRef"] == "pseudo_ref" and d["GAMBITP_hyp1"] == "challenge_hyp"
    assert t.schema.equals(SCHEMAS["mt_systems"], check_metadata=True)
    assert t.column("system").to_pylist() == sorted(d)


def test_mt_systems_counts_distinct_items_and_languages():
    pairs = pa.table(
        {
            "item_id": ["i1", "i1", "i2", "i1", "i3"],
            "lp": ["eng-deu", "eng-deu", "ces-deu", "eng-deu", "eng-jpn"],
            "system": ["sysB", "sysA", "sysB", "sysB", "sysA"],
            "system_type": ["mt", "mt", "mt", "mt", "mt"],
        }
    )
    rows = build_mt_systems(pairs).to_pylist()  # ``sysB`` has a repeated row for ``i1``
    assert rows == [
        {"system": "sysA", "system_type": "mt", "n_items": 2, "lps": ["eng-deu", "eng-jpn"]},
        {"system": "sysB", "system_type": "mt", "n_items": 2, "lps": ["ces-deu", "eng-deu"]},
    ]


def test_mt_systems_rejects_a_system_with_two_types():
    pairs = pa.table(
        {
            "item_id": ["i1", "i2"],
            "lp": ["eng-deu", "eng-deu"],
            "system": ["sysA", "sysA"],
            "system_type": ["mt", "human_scratch"],
        }
    )
    with pytest.raises(ValueError, match="sysA"):
        build_mt_systems(pairs)


# --- challenge_sets ------------------------------------------------------------------------------


def test_challenge_sets(repo_root, mini_pairs):
    t = build_challenge_sets(repo_root / "data/metadata/challenge_sets.yaml", mini_pairs)
    assert set(t.column("name").to_pylist()) <= set(CHALLENGE_SETS)
    assert all(t.column("full_name").to_pylist())
    assert t.schema.equals(SCHEMAS["challenge_sets"], check_metadata=True)


def test_challenge_sets_has_the_seven_sets_sorted(repo_root, mini_pairs):
    t = build_challenge_sets(repo_root / "data/metadata/challenge_sets.yaml", mini_pairs)
    assert t.column("name").to_pylist() == CHALLENGE_SETS
    by_name = {r["name"]: r for r in t.to_pylist()}
    for row in by_name.values():
        assert (row["n_items"] == 0) == (row["n_pairs"] == 0) == (row["lps"] == [])
    assert by_name["GAMBITP"]["n_items"] > 0  # the mini file holds a GAMBITP item


def test_challenge_sets_yaml_text(repo_root):
    entries = yaml.safe_load((repo_root / "data/metadata/challenge_sets.yaml").read_text("utf-8"))
    assert [e["name"] for e in entries] == CHALLENGE_SETS
    fields = {f.name for f in SCHEMAS["challenge_sets"]} - {"lps", "n_items", "n_pairs"}
    for entry in entries:
        assert set(entry) == fields
        assert all(isinstance(v, str) and v.strip() == v and v for v in entry.values())
    by_name = {e["name"]: e for e in entries}
    assert by_name["AGGREE"]["citation"] == "Griesbeck and Geierhos, 2026"
    assert by_name["FECE"]["citation"] == "Ding and Duan, 2026"
    assert by_name["GAMBITP"]["citation"] == "Menis Mastromichalakis et al., 2026"
    assert by_name["LIGATUR"]["citation"] == "Mishra et al., 2026"
    assert by_name["QEbreak"]["citation"] == "Silchenko, 2026"
    for name in ("RefQuality-PD", "RefQuality-PE"):
        assert by_name[name]["citation"] == "Lavie et al., 2026 (findings paper, §5.2.1)"


def test_challenge_set_full_names_are_paper_titles_or_plainly_descriptive(repo_root):
    entries = yaml.safe_load((repo_root / "data/metadata/challenge_sets.yaml").read_text("utf-8"))
    by_name = {e["name"]: e["full_name"] for e in entries}
    for name, title in PAPER_TITLES.items():
        assert by_name[name] == title
    # The two reference-swap sets have no published title, and must not look as if they had one.
    for name in ("RefQuality-PD", "RefQuality-PE"):
        assert by_name[name].startswith("Organisers' reference-swap set")
        assert by_name[name].endswith("(descriptive name, not a published title)")


def test_full_name_description_is_true_for_every_row():
    description = SCHEMAS["challenge_sets"].field("full_name").metadata[b"description"].decode()
    assert "reference list of the findings paper" in description
    assert "descriptive name" in description and "no published title" in description


def pairs_of(rows):
    """A minimal pairs table of ``(item_id, challenge_set, lp)`` rows."""
    items, sets, lps = zip(*rows, strict=True)
    return pa.table({"item_id": list(items), "challenge_set": list(sets), "lp": list(lps)})


def test_challenge_sets_counts_come_from_the_pairs(repo_root):
    pairs = pairs_of(
        [
            (challenge("FECE", 0), "FECE", "eng-deu"),
            (challenge("FECE", 0), "FECE", "eng-deu"),  # second hypothesis of the same item
            (challenge("FECE", 1), "FECE", "eng-deu"),
            (challenge("FECE", 2), "FECE", "eng-jpn"),
            (challenge("QEbreak", 0), "QEbreak", "ces-deu"),
            (official("a", "b", 0), None, "ces-deu"),  # official rows belong to no set
        ]
    )
    rows = build_challenge_sets(repo_root / "data/metadata/challenge_sets.yaml", pairs).to_pylist()
    by_name = {r["name"]: r for r in rows}
    assert len(rows) == 7
    assert (by_name["FECE"]["n_items"], by_name["FECE"]["n_pairs"]) == (3, 4)
    assert by_name["FECE"]["lps"] == ["eng-deu", "eng-jpn"]
    assert (by_name["QEbreak"]["n_items"], by_name["QEbreak"]["n_pairs"]) == (1, 1)
    assert by_name["AGGREE"]["n_items"] == 0 and by_name["AGGREE"]["lps"] == []


def test_challenge_sets_fails_loudly_on_a_set_missing_from_the_yaml(repo_root):
    pairs = pairs_of([(challenge("MYSTERY", 0), "MYSTERY", "eng-deu")])
    with pytest.raises(ValueError, match="MYSTERY"):
        build_challenge_sets(repo_root / "data/metadata/challenge_sets.yaml", pairs)


def test_challenge_sets_rejects_a_malformed_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- name: FECE\n  full_name: Only two fields\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FECE"):
        build_challenge_sets(bad, pairs_of([(challenge("FECE", 0), "FECE", "eng-deu")]))
