"""Tests for the dataset card, the generated documents and the hand-written GitHub documents."""

import csv
import importlib.util
import inspect
import json
import re
import shlex
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from wmt26_mteval.build import SPLIT, build_all
from wmt26_mteval.cards import EMAIL_RE
from wmt26_mteval.dataset_card import (
    CONFIG_GUIDE,
    card_stats,
    render_card,
    render_format_md,
    render_manifest_tsv,
    size_category,
)
from wmt26_mteval.grand import Grand
from wmt26_mteval.ids import SYSTEM_TYPES
from wmt26_mteval.schemas import SCHEMAS
from wmt26_mteval.validate import ALLOWED_EMAILS, find_pii, load_table2, validate

ORGANISERS = "wmt-qe-metrics-organizers@googlegroups.com"
HUMEVAL = "https://github.com/wmt-conference/wmt26-general-mt/releases/tag/humeval"
CONFIGS = [*SCHEMAS, "grand"]

# The 18 authors of the findings paper in the order of its title page (read row by row).
FINDINGS_AUTHORS = [
    ("Alon", "Lavie"),
    ("Greg", "Hanneman"),
    ("Stefano", "Perrella"),
    ("Shuoyang", "Ding"),
    ("Eleftherios", "Avramidis"),
    ("Lorenzo", "Proietti"),
    ("Chi-kiu", "Lo"),
    ("Ammon", "Shurtz"),
    ("Chrysoula", "Zerva"),
    ("Archchana", "Sindhujan"),
    ("Vilém", "Zouhar"),
    ("Diptesh", "Kanojia"),
    ("Frédéric", "Blain"),
    ("Brian", "Thompson"),
    ("Giorgos", "Filandrianos"),
    ("Orfeas", "Menis Mastromichalakis"),
    ("Tom", "Kocmi"),
    ("Pranav", "Gupta"),
]


def load_script(repo_root: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, repo_root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def squash(text: str) -> str:
    """``text`` with every run of whitespace turned into one space."""
    return " ".join(text.split())


def front_matter(readme: str) -> dict:
    return yaml.safe_load(readme.split("---")[1])


@pytest.fixture(scope="module")
def built(tmp_path_factory, mini_grand_path, repo_root):
    """One build of the mini grand file, shared by the read-only tests: ``(out_dir, manifest)``."""
    out = tmp_path_factory.mktemp("docs") / "hf"
    return out, build_all(mini_grand_path, repo_root, out, expected_sha256=None)


@pytest.fixture(scope="module")
def card(built) -> str:
    out, _ = built
    return (out / "README.md").read_text(encoding="utf-8")


def fake_manifest() -> dict:
    """A manifest with distinctive numbers, to show where the card takes its numbers from."""
    rows = {config: 1_000_000 + 11 * i for i, config in enumerate(SCHEMAS)}
    return {
        "grand_sha256": "ab" * 32,
        "configs": {config: {"rows": n, "files": []} for config, n in rows.items()},
        "stats": {
            "rows": rows,
            "hyp_empty_exact": 7_391,
            "hyp_whitespace_only": 482,
            "reference_candidates": 6_173,
            "reference_candidates_by_system_type": {
                "mt": 3_401,
                "human_scratch": 812,
                "human_postedit": 700,
                "pseudo_ref": 1_100,
                "challenge_hyp": 160,
            },
            "baseline_alt_rows": 1_234_567,
            "baseline_alt_runs": 9,
            "languages": ["eng", "zho"],
            "coverage": [
                {
                    "lp": "eng-zho_CN",
                    "src_items": 4_817,
                    "pairs": 94_231,
                    "ref_type": "pseudo",
                    "is_priority": False,
                },
                {
                    "lp": "zho-jpn",
                    "src_items": 3_351,
                    "pairs": 83_851,
                    "ref_type": "postedit",
                    "is_priority": True,
                },
            ],
        },
    }


# --- the dataset card ----------------------------------------------------------------------------


def test_card_yaml(tmp_path, mini_grand_path, repo_root):
    build_all(mini_grand_path, repo_root, tmp_path, expected_sha256=None)
    head = (tmp_path / "README.md").read_text().split("---")[1]
    y = yaml.safe_load(head)
    cfg = {c["config_name"]: c for c in y["configs"]}
    assert set(cfg) == {
        "items",
        "pairs",
        "submissions",
        "baselines",
        "systems",
        "mt_systems",
        "challenge_sets",
        "grand",
    }
    assert all(c["data_files"][0]["split"] == "evaluation" for c in cfg.values())
    assert cfg["grand"]["data_files"][0]["path"] == "wmt26-grand.parquet"
    assert cfg["pairs"].get("default") is True
    for name, c in cfg.items():
        if name != "grand":
            assert c["data_files"][0]["path"] == f"data/{name}/{SPLIT}-*.parquet", name


def test_the_split_is_the_organisers_evaluation_split(tmp_path, mini_grand_path, repo_root):
    # The organisers' grand file names its split "evaluation" (wmt26_intended_split), and the
    # first card on the Hub did too; every config and every shard uses that name.
    assert SPLIT == "evaluation"
    manifest = build_all(mini_grand_path, repo_root, tmp_path, expected_sha256=None)
    for config, info in manifest["configs"].items():
        for f in info["files"]:
            assert Path(f["path"]).name.startswith(f"{SPLIT}-"), (config, f["path"])
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "not partitioned into train/test sets" in " ".join(card.split())
    assert 'split="test"' not in card and "split: test" not in card


def test_card_mentions_datasets_floor_for_grand(tmp_path, mini_grand_path, repo_root):
    build_all(mini_grand_path, repo_root, tmp_path, expected_sha256=None)
    assert "datasets>=4.7.0" in (tmp_path / "README.md").read_text()


def test_card_metadata(card, built):
    out, manifest = built
    y = front_matter(card)
    assert y["pretty_name"] == "WMT26 MT-Eval"
    assert y["task_categories"] == ["translation"]
    assert "license" not in y  # no licence is attached to the data
    assert [c["config_name"] for c in y["configs"]].count("pairs") == 1
    assert sum(1 for c in y["configs"] if c.get("default")) == 1
    assert y["size_categories"] == [size_category(max(manifest["stats"]["rows"].values()))]
    assert y["size_categories"] == [size_category(manifest["configs"]["submissions"]["rows"])]


def test_card_is_valid_front_matter_without_further_delimiters(card):
    assert card.startswith("---\n")
    head, body = card[4:].split("\n---\n", 1)
    assert "---" not in head
    assert yaml.safe_load(head)["configs"]
    assert body.lstrip().startswith("# ")


def test_card_languages_cover_the_data_with_iso_639_3_codes(card, built):
    out, manifest = built
    languages = front_matter(card)["language"]
    assert languages == sorted(set(languages)) == manifest["stats"]["languages"]
    assert all(re.fullmatch(r"[a-z]{3}", code) for code in languages)
    items = pq.read_table(
        out / "data/items/evaluation-00000.parquet", columns=["src_lang", "tgt_lang"]
    )
    in_data = {
        code[:3] for column in items.column_names for code in items.column(column).to_pylist()
    }
    assert set(languages) == in_data


def test_every_data_files_path_matches_exactly_the_built_files(card, built):
    out, manifest = built
    for entry in front_matter(card)["configs"]:
        name = entry["config_name"]
        (data_files,) = entry["data_files"]
        if name == "grand":
            assert data_files["path"] == "wmt26-grand.parquet"
            continue
        matched = sorted(p.relative_to(out).as_posix() for p in out.glob(data_files["path"]))
        assert matched == [f["path"] for f in manifest["configs"][name]["files"]], name


def test_the_readme_sits_next_to_the_manifest_and_is_not_a_config_file(built):
    out, manifest = built
    assert (out / "README.md").is_file() and (out / "build_manifest.json").is_file()
    assert "README.md" not in {f["path"] for c in manifest["configs"].values() for f in c["files"]}


def test_card_states_the_facts_a_user_must_know(card):
    text = squash(card)
    for fragment in (
        "`para_idx` is null for FECE and QEbreak",
        "segment hash",
        "`stage1_finish_reason`",
        "only in the `grand` config",
        "`t1_instruction_fault` is null in every row",
        "empty or whitespace-only",
        "allow-list",
        "volunteered",
        "no human-annotation gold",
        "planned for v1.0",
        HUMEVAL,
        "`multimodal_inputs.zip`",
        "relative to",
        "`Cohere CAT+`",
        "outside 0-100",
        "datasets>=4.7.0",
        'load_dataset("wmt/wmt26-mteval-data", "pairs", split="evaluation")',
        "duckdb",
    ):
        assert fragment in text, fragment


def test_card_documents_the_five_reference_types(card):
    text = squash(card)
    for fragment in (
        "| `unknown` | `undefined` | A reference text is present, but the source data does not "
        "record its type. |",
        "| `none` | absent, or null text and type | The item has no reference text, and "
        "`ref_text` is null. |",
    ):
        assert fragment in text, fragment
    description = SCHEMAS["items"].field("ref_type").metadata[b"description"].decode()
    for value in ("native", "postedit", "pseudo", "unknown", "none"):
        assert f"`{value}`" in description, value


def test_card_scopes_the_reference_warnings_to_the_sets_that_have_them(card):
    text = squash(card)
    for fragment in (
        "**References are among the candidate translations of official items.**",
        "Challenge hypotheses can equal the reference too",
        "`challenge_hyp` rows",
        "GAMBITP has no reference",
        "AGGREE, FECE, LIGATUR and QEbreak have a reference translation",
        "`ref_type` is `unknown`",
        "RefQuality sets have pseudo-references or post-edits",
    ):
        assert fragment in text, fragment
    assert "They have no references here" not in text
    assert "Every available reference is also a hypothesis" not in text


def test_challenge_set_reference_semantics_follow_the_data(repo_root):
    sets = yaml.safe_load((repo_root / "data/metadata/challenge_sets.yaml").read_text("utf-8"))
    refs = {entry["name"]: squash(entry["ref_semantics"]) for entry in sets}
    for name in ("AGGREE", "FECE", "LIGATUR", "QEbreak"):
        assert "A reference translation is included" in refs[name], name
        assert "does not record its type (`ref_type` is `unknown`)" in refs[name], name
        assert "No reference translation is included" not in refs[name], name
    assert refs["GAMBITP"].startswith("No reference translation is included")
    assert "`pseudo`" in refs["RefQuality-PD"] and "`postedit`" in refs["RefQuality-PE"]


def test_none_is_exactly_the_items_without_a_reference_text(built):
    out, _ = built
    items = pq.read_table(out / "data/items/evaluation-00000.parquet").to_pylist()
    assert {r["ref_type"] for r in items} >= {"none", "unknown"}  # the mini grand has both
    for r in items:
        assert (r["ref_type"] == "none") == (r["ref_text"] is None), r["item_id"]
        if r["ref_type"] == "unknown":
            assert r["ref_type_orig"] == "undefined"


def test_every_official_reference_is_among_the_hypotheses(built):
    out, _ = built
    pairs = pa.concat_tables(
        pq.read_table(p, columns=["item_id", "collection", "ref_text", "is_reference_candidate"])
        for p in sorted((out / "data" / "pairs").glob("*.parquet"))
    ).to_pylist()
    with_reference = {
        r["item_id"] for r in pairs if r["collection"] == "official" and r["ref_text"]
    }
    candidates = {r["item_id"] for r in pairs if r["is_reference_candidate"]}
    assert with_reference and with_reference <= candidates


def test_card_names_the_thin_language_pairs_with_the_papers_percentages(card):
    text = squash(card)
    for fragment in ("ces-vie", "10.6 %", "eng-zho_CN", "17.1 %", "eng-sme", "19.0 %"):
        assert fragment in text, fragment


# No licence is attached to the data or the repository, and nothing says anything about one.
LICENCE_WORD = re.compile(r"licen[cs]", flags=re.I)


def test_the_card_says_nothing_about_a_licence(card):
    assert not LICENCE_WORD.search(card)


def test_the_repository_says_nothing_about_a_licence(repo_root):
    assert not list(repo_root.glob("LICEN[CS]E*")) and not list(repo_root.glob("DATA_LICEN[CS]E*"))
    for name in (
        "README.md",
        "CITATION.cff",
        "hf/README.template.md",
        "docs/CHANGELOG.md",
        "docs/RELEASE.md",
    ):
        text = (repo_root / name).read_text(encoding="utf-8")
        assert not LICENCE_WORD.search(text), name


def test_card_carries_every_section_a_newcomer_looks_for(card):
    headings = re.findall(r"^#{1,3} (.+)$", card, flags=re.M)
    wanted = [
        "Quick start",
        "Configs",
        "Item identifiers",
        "Warnings",
        "Coverage",
        "Field glossary",
        "Known issues",
        "Personal data",
        "Human annotations",
        "Citation",
    ]
    for word in wanted:
        assert any(word in h for h in headings), (word, headings)


def test_card_has_a_field_table_row_for_every_column(card):
    for config, schema in SCHEMAS.items():
        for field in schema:
            assert re.search(rf"^\| `{field.name}` \|", card, flags=re.M), (config, field.name)


def test_card_cites_the_findings_paper_and_each_related_work(card, repo_root):
    text = squash(card)
    assert "@inproceedings{" in card
    assert "pages = {931--1030}" in card
    assert "https://www2.statmt.org/wmt26/pdf/2026.wmt-1.49.pdf" in card
    assert "Proceedings of the Eleventh Conference on Machine Translation (WMT)" in text
    for title in (
        "Findings of the WMT26 general machine translation shared task: Contrastive dynamic "
        "human evaluation at scale",
        "Contrastive esa: Human evaluation of multiple translations at once",
        "In the blind: Building pseudo-references for MT evaluation",
    ):
        assert title in text, title
    sets = yaml.safe_load((repo_root / "data/metadata/challenge_sets.yaml").read_text("utf-8"))
    for entry in sets:
        if "descriptive name" not in entry["full_name"]:
            assert squash(entry["full_name"]) in text, entry["name"]
    for author in ("Lavie", "Kocmi", "Zouhar", "Kanojia"):
        assert author in card
    # the findings paper's authors, in order, in the BibTeX entry of the findings paper
    entry = card.split("@inproceedings{lavie2026findings,")[1].split("\n}\n")[0]
    names = re.search(r"author = \{(.+?)\},\n", entry, flags=re.S).group(1)
    assert [a.strip() for a in squash(names).split(" and ")] == [
        f"{family}, {given}" for given, family in FINDINGS_AUTHORS
    ]


def test_card_is_deterministic(built, repo_root):
    _, manifest = built
    template = repo_root / "hf" / "README.template.md"
    assert render_card(template, manifest) == render_card(template, manifest)


def test_a_rebuild_gives_the_same_readme(built, tmp_path, mini_grand_path, repo_root):
    out, _ = built
    build_all(mini_grand_path, repo_root, tmp_path / "again", expected_sha256=None)
    assert (tmp_path / "again" / "README.md").read_bytes() == (out / "README.md").read_bytes()


def test_the_card_passes_the_pii_gate_and_names_only_the_organisers_address(built, card):
    out, _ = built
    assert find_pii(out) == []
    found = [m.group(0).lower() for m in EMAIL_RE.finditer(card)]
    assert found and set(found) == set(ALLOWED_EMAILS)
    assert ORGANISERS in card


def test_the_card_never_prints_the_model_card_field_labels(card):
    for label in ("Participant name", "Email", "Data Collection Consent"):
        assert label not in card, label


def test_card_numbers_come_from_the_manifest(repo_root):
    manifest = fake_manifest()
    text = render_card(repo_root / "hf" / "README.template.md", manifest)
    stats = manifest["stats"]
    assert f"{stats['hyp_empty_exact']:,}" in text and f"{stats['hyp_whitespace_only']:,}" in text
    assert f"{stats['hyp_empty_exact'] + stats['hyp_whitespace_only']:,}" in text
    assert f"{stats['reference_candidates']:,}" in text
    for n in stats["reference_candidates_by_system_type"].values():
        assert f"{n:,}" in text
    assert f"{stats['baseline_alt_rows']:,}" in text and f"{stats['baseline_alt_runs']}" in text
    for n in stats["rows"].values():
        assert f"{n:,}" in text
    assert manifest["grand_sha256"] in text
    for row in stats["coverage"]:
        assert row["lp"] in text and f"{row['src_items']:,}" in text and f"{row['pairs']:,}" in text
    # the official pairs are the sum of the coverage table, the challenge pairs the rest
    official = sum(row["pairs"] for row in stats["coverage"])
    assert f"{official:,}" in text
    assert f"{stats['rows']['pairs'] - official:,}" in text


def test_the_coverage_table_marks_priority_pairs_and_reference_types(repo_root):
    text = render_card(repo_root / "hf" / "README.template.md", fake_manifest())
    rows = {
        cells[0]: cells
        for cells in (
            [c.strip() for c in line.strip("|").split("|")]
            for line in text.splitlines()
            if line.startswith("| `eng-zho_CN`") or line.startswith("| `zho-jpn`")
        )
    }
    assert rows["`eng-zho_CN`"][1:] == ["4,817", "94,231", "pseudo", "no"]
    assert rows["`zho-jpn`"][1:] == ["3,351", "83,851", "postedit", "yes"]


def test_the_template_hard_codes_no_counts(repo_root):
    """Every count in the card comes from the manifest; only the paper's percentages are typed."""
    template = (repo_root / "hf" / "README.template.md").read_text(encoding="utf-8")
    prose = re.sub(r"```bibtex.*?```", "", template, flags=re.S)  # BibTeX: pages and years
    prose = re.sub(r"https?://\S+", "", prose)
    prose = re.sub(r"\b20(25|26)[a-z]?\b", "", prose)  # years and citation labels
    assert re.findall(r"\d{1,3}(?:,\d{3})+|\d{4,}", prose) == []  # 26,885 or 26885
    assert re.findall(r"\b\d+\.\d+ %", prose) == ["10.6 %", "17.1 %", "19.0 %"]


def test_render_card_rejects_an_unknown_placeholder(tmp_path):
    template = tmp_path / "t.md"
    template.write_text("---\nconfigs: []\n---\n{{no_such_thing}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no_such_thing"):
        render_card(template, fake_manifest())


def test_the_config_guide_covers_every_config():
    assert set(CONFIG_GUIDE) == set(SCHEMAS) | {"grand"}


@pytest.mark.parametrize(
    ("rows", "label"),
    [
        (0, "n<1K"),
        (999, "n<1K"),
        (1_000, "1K<n<10K"),
        (9_999, "1K<n<10K"),
        (10_000, "10K<n<100K"),
        (10_703, "10K<n<100K"),
        (100_000, "100K<n<1M"),
        (1_000_000, "1M<n<10M"),
        (10_753_092, "10M<n<100M"),
        (11_297_384, "10M<n<100M"),
        (100_000_000, "100M<n<1B"),
    ],
)
def test_size_category(rows, label):
    assert size_category(rows) == label


def test_a_missing_template_stops_the_build_before_anything_is_written(
    tmp_path, mini_grand_path, repo_root
):
    repo = tmp_path / "repo"
    shutil.copytree(repo_root / "data", repo / "data")  # a checkout without hf/README.template.md
    out = tmp_path / "hf"
    with pytest.raises(FileNotFoundError, match="README.template.md"):
        build_all(mini_grand_path, repo, out, expected_sha256=None)
    assert not out.exists()


def test_the_validation_gates_pass_on_a_build_with_its_card(built, mini_grand, expected_mini):
    out, _ = built
    assert validate(out, mini_grand, expected_mini) == []


# The reference type of each official language pair in Table 2 of the findings paper
# ("Human" is `native`).
PAPER_REFERENCE_TYPES = {
    "ces-deu": "native",
    "ces-ukr": "none",
    "ces-vie": "native",
    "eng-arz": "postedit",
    "eng-bel": "pseudo",
    "eng-ces": "postedit",
    "eng-deu": "postedit",
    "eng-ekk": "native",
    "eng-hye": "pseudo",
    "eng-ind": "pseudo",
    "eng-isl": "native",
    "eng-kaz": "pseudo",
    "eng-lij": "native",
    "eng-lld": "native",
    "eng-rus": "pseudo",
    "eng-sme": "native",
    "eng-tha": "pseudo",
    "eng-ukr": "pseudo",
    "eng-zho_CN": "pseudo",
    "eng-zho_TW": "pseudo",
    "zho-jpn": "postedit",
}
PAPER_PRIORITY = {
    "ces-deu",
    "ces-vie",
    "eng-arz",
    "eng-ekk",
    "eng-ind",
    "eng-isl",
    "eng-sme",
    "zho-jpn",
}


@pytest.mark.fulldata
def test_the_real_card_statistics_are_table_2_of_the_paper(repo_root, full_out, expected_full):
    stats = json.loads((full_out / "build_manifest.json").read_text(encoding="utf-8"))["stats"]
    table2 = load_table2(repo_root / "data/metadata/table2.tsv")
    assert {r["lp"]: (r["src_items"], r["pairs"]) for r in stats["coverage"]} == table2
    assert {r["lp"]: r["ref_type"] for r in stats["coverage"]} == PAPER_REFERENCE_TYPES
    assert {r["lp"] for r in stats["coverage"] if r["is_priority"]} == PAPER_PRIORITY
    assert stats["rows"] == expected_full["rows"]
    # Ruling R10: 3,689 hypotheses are exactly empty and 96 more are whitespace-only
    assert (stats["hyp_empty_exact"], stats["hyp_whitespace_only"]) == (3689, 96)
    # the final review: 18,021 flagged pairs, of which 9,792 are not reference systems
    assert stats["reference_candidates"] == 18021
    assert stats["reference_candidates_by_system_type"] == {
        "mt": 8673,
        "human_scratch": 2888,
        "human_postedit": 1714,
        "pseudo_ref": 3627,
        "challenge_hyp": 1119,
    }
    assert (stats["baseline_alt_rows"], stats["baseline_alt_runs"]) == (3469756, 12)
    card = (full_out / "README.md").read_text(encoding="utf-8")
    assert "3,785 empty or whitespace-only hypotheses (3,689 exactly empty" in squash(card)
    assert find_pii(full_out) == []


# --- the card statistics -------------------------------------------------------------------------


def stat_tables():
    items = pa.table(
        {
            "item_id": ["a1", "a2", "b1", "c1", "x1"],
            "collection": ["official", "official", "official", "official", "challenge"],
            "lp": ["ces-deu", "ces-deu", "eng-ekk", "ces-deu", "eng-deu"],
            "ref_type": ["native", "native", "pseudo", "native", "pseudo"],
            "is_priority": [True, True, False, True, False],
            "src_lang": ["ces_Latn", "ces_Latn", "eng_Latn", "ces_Latn", "eng_Latn"],
            "tgt_lang": ["deu_Latn", "deu_Latn", "ekk_Latn", "deu_Latn", "zho_Hant_TW"],
        }
    )
    hyps = ["fine", "", " \n ", "copy", "\t", "also fine", "x", ""]
    pairs = pa.table(
        {
            "collection": ["official"] * 7 + ["challenge"],
            "lp": ["ces-deu"] * 4 + ["eng-ekk"] * 3 + ["eng-deu"],
            "system_type": ["mt", "mt", "mt", "human_scratch", "mt", "pseudo_ref", "mt"]
            + ["challenge_hyp"],
            "hyp": hyps,
            "hyp_is_empty": [not h.strip() for h in hyps],
            "is_reference_candidate": [False, False, False, True, False, True, True, True],
        }
    )
    return items, pairs


def test_card_stats_counts_empty_whitespace_only_and_candidates():
    stats = card_stats(*stat_tables())
    assert stats["hyp_empty_exact"] == 2  # "" twice
    assert stats["hyp_whitespace_only"] == 2  # " \n " and "\t"
    assert stats["reference_candidates"] == 4


def test_card_stats_counts_the_reference_candidates_by_system_type():
    stats = card_stats(*stat_tables())
    by_type = stats["reference_candidates_by_system_type"]
    assert by_type == {
        "mt": 1,
        "human_scratch": 1,
        "human_postedit": 0,
        "pseudo_ref": 1,
        "challenge_hyp": 1,
    }
    assert list(by_type) == list(SYSTEM_TYPES)  # the five system types, always in this order
    assert sum(by_type.values()) == stats["reference_candidates"]


def test_card_stats_refuses_a_reference_candidate_of_an_unknown_system_type():
    items, pairs = stat_tables()
    index = pairs.schema.get_field_index("system_type")
    types = pairs.column("system_type").to_pylist()
    types[3] = "oracle"
    pairs = pairs.set_column(index, "system_type", pa.array(types))
    with pytest.raises(ValueError, match="oracle"):
        card_stats(items, pairs)


def test_card_stats_coverage_counts_official_items_and_pairs_per_language_pair():
    stats = card_stats(*stat_tables())
    assert stats["coverage"] == [
        {
            "lp": "ces-deu",
            "src_items": 3,
            "pairs": 4,
            "ref_type": "native",
            "is_priority": True,
        },
        {
            "lp": "eng-ekk",
            "src_items": 1,
            "pairs": 3,
            "ref_type": "pseudo",
            "is_priority": False,
        },
    ]


def test_card_stats_languages_are_the_three_letter_codes_of_the_data():
    stats = card_stats(*stat_tables())
    assert stats["languages"] == ["ces", "deu", "ekk", "eng", "zho"]


def test_card_stats_joins_mixed_reference_types():
    items, pairs = stat_tables()
    items = items.set_column(
        items.schema.get_field_index("ref_type"),
        "ref_type",
        pa.array(["native", "pseudo", "pseudo", "native", "pseudo"]),
    )
    (first, _) = card_stats(items, pairs)["coverage"]
    assert first["ref_type"] == "native+pseudo"


def test_the_manifest_stats_match_a_recount_from_the_parquet_files(built):
    out, manifest = built
    stats = manifest["stats"]
    pairs = pa.concat_tables(
        pq.read_table(p, columns=["collection", "lp", "hyp", "hyp_is_empty"])
        for p in sorted((out / "data" / "pairs").glob("*.parquet"))
    )
    hyps = pairs.column("hyp").to_pylist()
    assert stats["hyp_empty_exact"] == sum(h == "" for h in hyps)
    assert stats["hyp_empty_exact"] + stats["hyp_whitespace_only"] == sum(
        not h.strip() for h in hyps
    )
    assert stats["hyp_empty_exact"] + stats["hyp_whitespace_only"] == sum(
        pairs.column("hyp_is_empty").to_pylist()
    )
    candidates = 0
    for path in (out / "data" / "pairs").glob("*.parquet"):
        candidates += sum(
            pq.read_table(path, columns=["is_reference_candidate"]).to_pydict()[
                "is_reference_candidate"
            ]
        )
    assert stats["reference_candidates"] == candidates
    by_type = Counter()
    for path in (out / "data" / "pairs").glob("*.parquet"):
        flagged = pq.read_table(path, columns=["system_type", "is_reference_candidate"])
        by_type.update(
            system_type
            for system_type, flag in zip(
                flagged.column("system_type").to_pylist(),
                flagged.column("is_reference_candidate").to_pylist(),
            )
            if flag
        )
    assert stats["reference_candidates_by_system_type"] == {
        system_type: by_type.get(system_type, 0) for system_type in SYSTEM_TYPES
    }
    alt_files = sorted((out / "data" / "baselines").glob("*-alt.parquet"))
    assert stats["baseline_alt_runs"] == len(alt_files) > 0
    assert stats["baseline_alt_rows"] == sum(pq.ParquetFile(p).metadata.num_rows for p in alt_files)
    official = [
        lp
        for c, lp in zip(pairs["collection"].to_pylist(), pairs["lp"].to_pylist())
        if c == "official"
    ]
    assert {row["lp"]: row["pairs"] for row in stats["coverage"]} == {
        lp: official.count(lp) for lp in set(official)
    }
    assert [row["lp"] for row in stats["coverage"]] == sorted(set(official))
    assert stats["rows"] == {c: info["rows"] for c, info in manifest["configs"].items()}


# --- advice that the final review corrected -------------------------------------------------------


def fake_card(repo_root) -> str:
    return render_card(repo_root / "hf" / "README.template.md", fake_manifest())


def test_the_card_does_not_tell_users_to_drop_every_reference_candidate(repo_root):
    text = squash(fake_card(repo_root))
    assert "exclude them" not in text
    for fragment in (
        "To drop the reference itself, filter on `system_type`",
        "`is_reference_candidate` also flags MT outputs that are identical to the reference",
        "Do not drop every `is_reference_candidate` row",
    ):
        assert fragment in text, fragment


def test_the_card_breaks_the_reference_candidates_down_by_system_type(repo_root):
    card = fake_card(repo_root)
    shown = re.findall(r"^ +- `(\w+)`: ([\d,]+) ", card, flags=re.M)
    by_type = fake_manifest()["stats"]["reference_candidates_by_system_type"]
    assert shown == [(system_type, f"{n:,}") for system_type, n in by_type.items()]


def test_the_card_warns_that_baselines_double_count_without_a_variant_filter(repo_root):
    text = squash(fake_card(repo_root))
    stats = fake_manifest()["stats"]
    for fragment in (
        "**`baselines` holds two snapshots of some runs.**",
        f"{stats['baseline_alt_rows']:,} `alt` rows",
        "Filter `variant == 'official'` unless you want the pre-deduplication snapshot.",
        "join the predictions to `systems` on `run_id`; for `baselines`, filter "
        "`variant == 'official'` first",
        'base = base.filter(lambda row: row["variant"] == "official")',
    ):
        assert fragment in text, fragment


def test_the_official_label_is_called_provisional_where_it_is_defined(repo_root):
    sentence = "follows decision A1"
    confirm = "confirmed against the paper in v1.0"
    card = squash(fake_card(repo_root))
    assert sentence in card and confirm in card
    errata = squash((repo_root / "docs" / "ERRATA.md").read_text(encoding="utf-8"))
    assert sentence in errata and confirm in errata
    variant = SCHEMAS["baselines"].field("variant").metadata[b"description"].decode()
    assert sentence in variant and confirm in variant


def test_the_card_wording_the_final_review_asked_for(repo_root):
    text = squash(fake_card(repo_root))
    for fragment in (
        "the sha256 and the rows of every Parquet file",
        "asked for contributions of four kinds",
        "Task 4 asked for challenge sets",
        "five were submitted by research teams",
        "the two RefQuality sets are the organisers' own",
        "A run may cover only some items",
        "the hypothesis equals the reference, up to surrounding whitespace",
    ):
        assert fragment in text, fragment
    for gone in ("predictions of four kinds", "of every file", "covers only some items"):
        assert gone not in text, gone


def test_the_personal_data_section_says_what_was_replaced_without_naming_anyone(repo_root):
    text = squash(fake_card(repo_root))
    section = text.split("## Personal data")[1].split("## Human annotations")[0]
    assert "were never read into this release" not in section
    for fragment in (
        "are not on the list, and their answers were never read",
        "an e-mail address inside an allow-listed answer is replaced by `[email removed]`",
        "one allow-listed answer that named individuals was replaced by the organisations",
    ):
        assert fragment in section, fragment
    assert "IIT" not in section and "Patna" not in section and "871298" not in section


def test_the_errata_make_no_guess_about_table_15(repo_root):
    text = squash((repo_root / "docs" / "ERRATA.md").read_text(encoding="utf-8"))
    assert "it may show only the items that were scored" not in text
    assert "The paper does not say why Table 15 lists fewer." in text


def test_the_schema_descriptions_the_final_review_corrected():
    def describe(config, name):
        return SCHEMAS[config].field(name).metadata[b"description"].decode()

    threshold = describe("systems", "t3_threshold")
    assert "above which" not in threshold and "`threshold_inclusive`" in threshold
    tasks = describe("systems", "tasks")
    assert "classic metrics" not in tasks and "seven sentence-level metric baselines" in tasks
    assert "classic metrics" not in describe("systems", "underlying_model")
    what = describe("challenge_sets", "what_it_tests")
    assert "One-sentence" not in what and what.startswith("Short description")
    reference = describe("pairs", "is_reference_candidate")
    assert "`system_type`" in reference


# --- generated documents -------------------------------------------------------------------------


def test_format_md_is_current(repo_root):
    assert (repo_root / "docs/FORMAT.md").read_text() == render_format_md()


def test_format_md_has_a_table_row_for_every_column_of_every_config(repo_root):
    text = render_format_md()
    for config, schema in SCHEMAS.items():
        section = text.split(f"\n## `{config}`\n")[1].split("\n## ")[0]
        rows = [line for line in section.splitlines() if line.startswith("| `")]
        assert [r.split("`")[1] for r in rows] == schema.names, config
    assert "\n## `grand`\n" in text


def test_format_md_descriptions_are_the_schema_descriptions():
    text = render_format_md()
    for schema in SCHEMAS.values():
        for field in schema:
            description = field.metadata[b"description"].decode()
            assert description.replace("|", "\\|") in text, field.name


def test_format_md_says_it_is_generated_and_documents_the_conventions(repo_root):
    text = render_format_md()
    assert "scripts/render_docs.py" in text.split("\n## ")[0]
    for fragment in (
        "`_###_`",
        "FECE and QEbreak",
        "`zho_CN`",
        "`zho_TW`",
        "Missing values",
    ):
        assert fragment in text, fragment
    with open(repo_root / "data/metadata/lang_codes.tsv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["orig"] != row["normalised"]:
                assert f"`{row['orig']}` | `{row['normalised']}`" in text, row


def test_the_two_published_description_fixes():
    t1_errors = SCHEMAS["submissions"].field("t1_errors").metadata[b"description"].decode()
    assert "`category` (which may be null)" in t1_errors
    priority = SCHEMAS["items"].field("is_priority").metadata[b"description"].decode()
    assert "false for every other item" in priority


def test_manifest_tsv_is_current(repo_root, mini_grand_path):
    grand = Grand(mini_grand_path, expected_sha256=None)  # the real grand has the same manifest
    assert (repo_root / "data/MANIFEST.tsv").read_text() == render_manifest_tsv(grand)


@pytest.mark.fulldata
def test_manifest_tsv_equals_the_render_of_the_real_grand(repo_root, full_grand):
    assert (repo_root / "data/MANIFEST.tsv").read_text() == render_manifest_tsv(full_grand)


def test_manifest_tsv_columns_and_rows(mini_grand_path):
    grand = Grand(mini_grand_path, expected_sha256=None)
    text = render_manifest_tsv(grand)
    assert text.endswith("\n") and "\r" not in text
    rows = list(csv.reader(text.splitlines(), delimiter="\t"))
    assert rows[0] == ["path", "bytes", "sha256", "records", "grand_column", "canonical_column"]
    assert len(rows) == len(grand.manifest) + 1
    for row, entry in zip(rows[1:], grand.manifest):
        assert row == [
            entry.path,
            str(entry.bytes),
            entry.sha256,
            str(entry.records),
            entry.column,
            entry.canonical_column,
        ]


def test_render_docs_writes_both_files(tmp_path, mini_grand_path, repo_root):
    render_docs = load_script(repo_root, "render_docs")
    (tmp_path / "data" / "metadata").mkdir(parents=True)
    shutil.copy(repo_root / "data/metadata/lang_codes.tsv", tmp_path / "data/metadata")
    argv = ["--grand", str(mini_grand_path), "--no-checksum", "--repo", str(tmp_path)]
    assert render_docs.main(argv) == 0
    assert (tmp_path / "docs/FORMAT.md").read_text() == render_format_md()
    grand = Grand(mini_grand_path, expected_sha256=None)
    assert (tmp_path / "data/MANIFEST.tsv").read_text() == render_manifest_tsv(grand)


def test_render_docs_checks_the_grand_checksum_by_default(
    tmp_path, mini_grand_path, repo_root, capsys
):
    render_docs = load_script(repo_root, "render_docs")
    code = render_docs.main(["--grand", str(mini_grand_path), "--repo", str(tmp_path)])
    assert code == 1
    assert "sha256" in capsys.readouterr().err
    assert not (tmp_path / "docs").exists() and not (tmp_path / "data").exists()


# --- scripts/download.py (no test reaches the network) -------------------------------------------


def test_download_patterns(repo_root):
    download = load_script(repo_root, "download")
    assert download.allow_patterns("grand") == ["wmt26-grand.parquet"]
    for config in SCHEMAS:
        assert download.allow_patterns(config) == [f"data/{config}/*"]
    assert set(download.CONFIGS) == set(CONFIGS)
    with pytest.raises(ValueError, match="nope"):
        download.allow_patterns("nope")
    with pytest.raises(ValueError):
        download.allow_patterns("../grand")


def test_download_calls_snapshot_download_with_the_documented_arguments(repo_root, tmp_path):
    download = load_script(repo_root, "download")
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return str(tmp_path)

    download.snapshot_download = fake
    assert download.main(["--config", "pairs", "--out", str(tmp_path)]) == 0
    download.main(["--config", "grand", "--out", str(tmp_path), "--revision", "refs/pr/1"])
    assert calls == [
        (
            ("wmt/wmt26-mteval-data",),
            {
                "repo_type": "dataset",
                "revision": "main",
                "allow_patterns": ["data/pairs/*"],
                "local_dir": str(tmp_path),
            },
        ),
        (
            ("wmt/wmt26-mteval-data",),
            {
                "repo_type": "dataset",
                "revision": "refs/pr/1",
                "allow_patterns": ["wmt26-grand.parquet"],
                "local_dir": str(tmp_path),
            },
        ),
    ]


def test_download_rejects_an_unknown_config_before_any_download(repo_root, tmp_path, capsys):
    download = load_script(repo_root, "download")
    download.snapshot_download = lambda *a, **k: pytest.fail("the network was called")
    with pytest.raises(SystemExit) as stop:
        download.main(["--config", "nope", "--out", str(tmp_path)])
    assert stop.value.code == 2
    assert "nope" in capsys.readouterr().err


def test_download_only_passes_arguments_that_huggingface_hub_accepts(repo_root):
    import huggingface_hub

    accepted = set(inspect.signature(huggingface_hub.snapshot_download).parameters)
    assert {"repo_type", "revision", "allow_patterns", "local_dir"} <= accepted


# --- the hand-written documents ------------------------------------------------------------------

TEXT_DOCS = [
    "README.md",
    "CITATION.cff",
    "docs/ERRATA.md",
    "docs/CHANGELOG.md",
    "docs/RELEASE.md",
    "docs/FORMAT.md",
    "instructions/baseline-prompts.md",
    "hf/README.template.md",
    "data/MANIFEST.tsv",
]


@pytest.mark.parametrize("name", TEXT_DOCS)
def test_documents_hold_no_e_mail_address_but_the_organisers(repo_root, name):
    text = (repo_root / name).read_text(encoding="utf-8")
    found = {m.group(0).lower() for m in EMAIL_RE.finditer(text)}
    assert found <= set(ALLOWED_EMAILS), (name, found)


# The baseline prompts are a verbatim transcription (American spelling included), so they are not
# checked here.
BRITISH_DOCS = [
    "README.md",
    "docs/ERRATA.md",
    "docs/CHANGELOG.md",
    "docs/RELEASE.md",
    "docs/FORMAT.md",
    "hf/README.template.md",
]


@pytest.mark.parametrize("name", BRITISH_DOCS)
def test_documents_use_british_spelling(repo_root, name):
    text = (repo_root / name).read_text(encoding="utf-8")
    prose = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # the YAML keys of the card
    prose = re.sub(r"```.*?```", "", prose, flags=re.S)  # code is not prose
    prose = re.sub(r"`[^`]*`", "", prose)
    prose = re.sub(r"https?://\S+", "", prose)
    for american in (
        r"behavior",
        r"color",
        r"organiz",
        r"normaliz",
        r"analyz",
        r"licens(e|ing)\b",
        r"center",
        r"favor",
        r"summariz",
        r"recogniz",
        r"serializ",
        r"tokeniz",
    ):
        assert not re.search(american, prose, flags=re.I), (name, american)


def test_the_readme_gives_the_contact_the_commands_and_the_layout(repo_root):
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert text.count(ORGANISERS) == 1
    for fragment in (
        "wmt/wmt26-mteval-data",
        'load_dataset("wmt/wmt26-mteval-data", "pairs", split="evaluation")',
        "scripts/build_hf.py",
        "scripts/validate_hf.py",
        "scripts/check_loading.py",
        "scripts/download.py",
        "docs/FORMAT.md",
        "docs/ERRATA.md",
        "docs/RELEASE.md",
        "CITATION.cff",
        HUMEVAL,
    ):
        assert fragment in text, fragment


@pytest.mark.parametrize("name", ["README.md", "docs/RELEASE.md", "docs/CHANGELOG.md"])
def test_every_file_a_document_points_to_exists(repo_root, name):
    text = (repo_root / name).read_text(encoding="utf-8")
    for path in set(re.findall(r"`((?:scripts|docs|data|instructions|hf|tests)/[\w./-]+)`", text)):
        if "*" in path or path.endswith("/") or "<" in path:
            continue
        assert (repo_root / path).exists(), (name, path)


def release_steps(repo_root) -> list[str]:
    text = (repo_root / "docs/RELEASE.md").read_text(encoding="utf-8")
    return re.split(r"^## (?=\d+\. )", text, flags=re.M)[1:]


def test_the_release_guide_orders_the_steps_and_marks_the_remote_ones_not_run(repo_root):
    steps = release_steps(repo_root)
    titles = [step.splitlines()[0] for step in steps]
    keywords = ["Build", "Validate", "Loading", "Stop", "Set", "Look", "Upload", "GitHub"]
    assert [t.split(". ", 1)[1].split()[0] for t in titles] == keywords
    commands = ["scripts/build_hf.py", "scripts/validate_hf.py", "scripts/check_loading.py"]
    for step, command in zip(steps[:3], commands):
        assert command in step
    before_upload = "".join(steps[:6])
    assert "hf upload" not in before_upload and "git push" not in before_upload
    assert "hf upload" in steps[6] and "--create-pr" in steps[6]
    assert "git push" in steps[7] and "gh pr create" in steps[7]
    assert all("Not run" in titles[i] for i in (5, 6, 7))
    matrix = steps[2]
    assert '--with "pyarrow<21" --with "numpy<2"' in matrix
    assert "--grand" in matrix and "TMPDIR" in matrix
    assert "Diptesh" in steps[3]
    assert "YYYY-MM-DD" in steps[4] and "docs/CHANGELOG.md" in steps[4] and "README.md" in steps[4]


def test_the_release_guide_looks_at_the_hub_repository_before_it_uploads(repo_root):
    look = squash(release_steps(repo_root)[5])
    for fragment in (
        "never seen",
        "hf datasets ls wmt/wmt26-mteval-data",
        "hf download wmt/wmt26-mteval-data README.md",
        'diff "$TMPDIR/hub-card/README.md" "$OUT/README.md"',
        "`wmt26-grand.parquet` is at the root of the repository",
        "Carry over",
        "rebuild",
    ):
        assert fragment in look, fragment
    assert "hf upload" not in look and "--create-pr" not in look


def test_the_release_guide_puts_the_new_open_decisions_in_the_stop_step(repo_root):
    stop = squash(release_steps(repo_root)[3])
    for fragment in (
        "`ref_type` `unknown`",
        "6,407",
        "decision A3",
        "the Hub card",
        "never seen",
    ):
        assert fragment in stop, fragment


def test_the_release_guide_checks_for_placeholders_without_matching_itself(repo_root):
    step = release_steps(repo_root)[4]
    (command,) = [line for line in step.splitlines() if line.startswith("grep ")]
    assert "--exclude=RELEASE.md" in command and "README.md" in command and "docs" in command
    # run as written (without the comment), the check may find the date placeholder in the
    # changelog (until the date is set), but never in the guide that describes the check
    result = subprocess.run(
        shlex.split(command.split("#")[0]), cwd=repo_root, capture_output=True, text=True
    )
    assert result.returncode in (0, 1), result.stderr
    assert {line.split(":")[0] for line in result.stdout.splitlines()} <= {"docs/CHANGELOG.md"}


def test_the_build_and_ci_install_from_the_lock_and_the_guide_keeps_the_build_commit(repo_root):
    guide = squash((repo_root / "docs/RELEASE.md").read_text(encoding="utf-8"))
    ci = (repo_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv sync --dev --locked" in guide and "uv sync --dev --locked" in ci
    assert "uv sync --dev\n" not in ci and "uv sync --dev`" not in guide
    for fragment in ("merge commit, not a squash", "`git_commit`", "stays resolvable"):
        assert fragment in guide, fragment


def test_the_quick_start_download_folder_is_git_ignored(repo_root):
    ignored = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/hf/" in ignored


def test_the_gates_are_listed_alike_in_the_guide_the_readme_and_the_changelog(repo_root):
    for name in ("README.md", "docs/CHANGELOG.md", "docs/RELEASE.md"):
        text = squash((repo_root / name).read_text(encoding="utf-8"))
        for fragment in ("Table 2", "the keys", "the texts of `items` and `pairs`", "alt"):
            assert fragment in text, (name, fragment)


def test_the_status_text_stays_true_after_publication(repo_root):
    """README and CHANGELOG are published with the release, so they must not say it has not been."""
    stale = ("not yet published", "unreleased", "nothing has been uploaded", "not been published")
    for name in ("README.md", "docs/CHANGELOG.md", "docs/RELEASE.md"):
        text = (repo_root / name).read_text(encoding="utf-8").lower()
        assert [phrase for phrase in stale if phrase in text] == [], name
    changelog = (repo_root / "docs/CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^## v0\.9: pre-release \(date: YYYY-MM-DD", changelog, flags=re.M)
    assert "pre-release" in (repo_root / "README.md").read_text(encoding="utf-8")


def test_the_errata_has_exactly_the_six_entries(repo_root):
    text = (repo_root / "docs/ERRATA.md").read_text(encoding="utf-8")
    headings = re.findall(r"^### (\d+)\. (.+)$", text, flags=re.M)
    assert [n for n, _ in headings] == ["1", "2", "3", "4", "5", "6"]
    flat = squash(text)
    for fragment in (
        "252,919",
        "252,923",
        "47",
        "23",
        "9 direct",
        "9 baselines",
        "2,886",
        "1,749",
        "171",
        "159",
        "run_names.tsv",
    ):
        assert fragment in flat, fragment


def test_the_changelog_has_a_v0_9_entry(repo_root):
    text = (repo_root / "docs/CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^## v0\.9", text, flags=re.M)
    assert "v1.0" in text


def test_baseline_prompts_transcribe_appendix_e(repo_root):
    text = (repo_root / "instructions/baseline-prompts.md").read_text(encoding="utf-8")
    flat = squash(text)
    for fragment in (
        "Your task is to identify machine translation errors and assess the quality of the "
        "translation.",
        "{src_name} source:",
        "Domain requirements: {domain_req}",
        "Return only the annotations in this format:",
        "Use one error per line and write no-error when a section is empty.",
        "assign a score on a continuous scale from 0 to 100",
        "Annotated error spans:",
        "Respond with ONLY a valid JSON object and nothing else: {“score”: N}",
        "where N is an integer from 0 to 100.",
        "Human reference translation",
        "Post-edited reference translation",
        "High-quality machine translation reference",
    ):
        assert fragment in flat, fragment
    for domain in ("news / factchecking", "speech", "social", "software", "edu", "general"):
        assert f"| {domain} |" in text, domain
    for figure in ("Figure 7", "Figure 8", "Table 63", "Table 64"):
        assert figure in text


def test_citation_cff_names_the_findings_paper(repo_root):
    cff = yaml.safe_load((repo_root / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0"
    assert all("name" in a and "given-names" not in a for a in cff["authors"])  # no personal names
    paper = cff["preferred-citation"]
    assert paper["type"] == "conference-paper"
    assert paper["title"] == (
        "Findings of the WMT26 Shared Task on Automated Translation Quality Evaluation "
        "Systems: Compact Open Models are Competitive Quality Evaluators"
    )
    assert [(a["given-names"], a["family-names"]) for a in paper["authors"]] == FINDINGS_AUTHORS
    assert paper["collection-title"] == (
        "Proceedings of the Eleventh Conference on Machine Translation (WMT)"
    )
    assert (paper["start"], paper["end"], paper["year"]) == (931, 1030, 2026)
    assert paper["url"] == "https://www2.statmt.org/wmt26/pdf/2026.wmt-1.49.pdf"
    assert cff["repository-code"] == "https://github.com/wmt-conference/wmt26-mteval"


@pytest.mark.fulldata
def test_the_real_reference_types_follow_ruling_r23(full_out):
    items = pq.read_table(
        full_out / "data/items/evaluation-00000.parquet",
        columns=["challenge_set", "collection", "lp", "ref_type", "ref_type_orig", "ref_text"],
    ).to_pylist()
    for r in items:
        assert (r["ref_type"] == "none") == (r["ref_text"] is None)
        assert (r["ref_type"] == "unknown") == (r["ref_type_orig"] == "undefined")
    unknown = {r["challenge_set"] for r in items if r["ref_type"] == "unknown"}
    assert unknown == {"AGGREE", "FECE", "LIGATUR", "QEbreak"}
    none = {(r["collection"], r["challenge_set"]) for r in items if r["ref_type"] == "none"}
    assert none == {("official", None), ("challenge", "GAMBITP")}
    official_none = {
        r["lp"] for r in items if r["collection"] == "official" and r["ref_text"] is None
    }
    assert official_none == {"ces-ukr"}
    sets = {r["challenge_set"]: r["ref_type"] for r in items if r["collection"] == "challenge"}
    assert sets["RefQuality-PD"] == "pseudo" and sets["RefQuality-PE"] == "postedit"
    pairs = pa.concat_tables(
        pq.read_table(p, columns=["item_id", "collection", "ref_text", "is_reference_candidate"])
        for p in sorted((full_out / "data" / "pairs").glob("*.parquet"))
    ).to_pylist()
    with_reference = {
        r["item_id"] for r in pairs if r["collection"] == "official" and r["ref_text"]
    }
    candidates = {r["item_id"] for r in pairs if r["is_reference_candidate"]}
    assert with_reference <= candidates  # every official reference is among the hypotheses
