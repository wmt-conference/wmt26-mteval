"""The dataset card of the Hugging Face repository and the documents generated from the code.

Three renderers live here, so that what a reader sees is derived from what the build wrote:

- ``render_card`` fills ``hf/README.template.md`` from the build manifest.  Every number the card
  states comes from the manifest: the rows of each config, the checksum of the grand file and the
  ``stats`` that ``build_all`` computes from the tables it has just built (see ``card_stats``).
  Only the paper's own percentages of thinly annotated language pairs are typed into the template.
- ``render_format_md`` writes ``docs/FORMAT.md``: the conventions and one table per config, taken
  from the field descriptions in ``schemas.SCHEMAS``.
- ``render_manifest_tsv`` writes ``data/MANIFEST.tsv``: one line per source file recorded in the
  grand file.

``card_stats`` returns the ``stats`` entry of the build manifest:

- ``rows``: total rows per config;
- ``hyp_empty_exact`` and ``hyp_whitespace_only``: the hypotheses that are exactly ``""`` and those
  that are non-empty but whitespace-only (``hyp_is_empty`` is true for both);
- ``reference_candidates``: the pairs whose hypothesis equals the reference;
- ``coverage``: for each official language pair (sorted by ``lp``) its source items, its pairs,
  its reference type (several types are joined with ``+``) and whether it is a priority pair;
- ``languages``: the sorted three-letter language codes of the data.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from wmt26_mteval.grand import Grand
from wmt26_mteval.schemas import SCHEMAS

CARD_TEMPLATE_PATH = Path("hf") / "README.template.md"
DEFAULT_LANG_CODES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "metadata" / "lang_codes.tsv"
)
MANIFEST_COLUMNS = ("path", "bytes", "sha256", "records", "grand_column", "canonical_column")

# What each config is, for the card and FORMAT.md: ``(one row per, what it holds)``.  The ``grand``
# file is not built here; it keeps one row per item, as ``items`` does.
CONFIG_GUIDE: dict[str, tuple[str, str]] = {
    "items": (
        "test item",
        "Source text, reference and its type, the columns parsed from `item_id`, the paths of "
        "the media files and the number of systems that translated the item.",
    ),
    "pairs": (
        "item and MT system",
        "The hypotheses: source, hypothesis, reference, whether the hypothesis is a copy of the "
        "reference (`is_reference_candidate`) and whether it is empty (`hyp_is_empty`). No gold "
        "columns in v0.9.",
    ),
    "submissions": (
        "participant run, item and system",
        "What each participant's evaluator predicted: error spans, omission, score or "
        "error-free judgement, for the task it was entered in.",
    ),
    "baselines": (
        "organiser baseline run, item and system",
        "The organisers' baselines in the same columns as `submissions`, plus `variant` "
        "(`official` or `alt`).",
    ),
    "systems": (
        "official evaluator run",
        "The run's name, organisation, curated paper name and citation, allow-listed answers "
        "from its model card, and what it covers.",
    ),
    "mt_systems": (
        "MT system",
        "Every system that has hypotheses in `pairs`, with its type, items and language pairs.",
    ),
    "challenge_sets": (
        "challenge set",
        "Authors, citation, what the set tests, what its hypotheses and references are, and its "
        "counts.",
    ),
    "grand": (
        "test item",
        "The organisers' lossless archive, unchanged: item fields, the raw annotation records and "
        "one JSON column per submission and baseline snapshot (needs `datasets>=4.7.0`).",
    ),
}

# The sections shared by the card and FORMAT.md: ``(title, Markdown body)``.
CONVENTIONS: tuple[tuple[str, str], ...] = (
    (
        "Item identifiers",
        "An `item_id` has six fields joined by the separator `_###_`:\n"
        "\n"
        "```text\n"
        "<collection>_###_<source>_###_<target>_###_<domain or set>_###_<document>_###_<para>\n"
        "official_###_eng_Latn_###_arz_Arab_###_news_###_brisbanetimes.com.au.377415_###_0\n"
        "```\n"
        "\n"
        "| field | column | meaning |\n"
        "| --- | --- | --- |\n"
        "| 1 | `collection` | `official` for the main test set, `challenge` for a challenge "
        "set. |\n"
        "| 2, 3 | `src_lang_orig`, `tgt_lang_orig` | The language codes as written in the id; "
        "`src_lang` and `tgt_lang` hold the normalised codes. |\n"
        "| 4 | `domain` or `challenge_set` | The domain of an official item (`news`, `social`, "
        "`speech`, `software`, `edu` or `factchecking`), or the name of the challenge set "
        "(`GAMBITP`, `FECE`, `QEbreak`, `AGGREE` or `LIGATUR`). The two RefQuality sets keep the "
        "domain here and carry their name in the document field. |\n"
        "| 5 | `doc_id` | The source document. The document of a RefQuality item starts with "
        "the name of its set (`RefQuality-PD_` or `RefQuality-PE_`); one that starts with "
        "`RefQuality-PD_RefQuality-PE_` marks the pseudo-reference variant of an item whose "
        "alternative reference is a post-edit. |\n"
        "| 6 | `para_idx` | The zero-based paragraph index (see below). |\n"
        "\n"
        "`para_idx` is null for FECE and QEbreak, whose ids end in a segment hash instead of a "
        "paragraph number.\n"
        "\n"
        'Split on the whole separator, not on `#`: `item_id.split("_###_")`. Document ids can '
        "themselves contain `_`, `|` and `.`.",
    ),
    (
        "Language codes",
        "Codes are normalised to an ISO 639-3 language code and an ISO 15924 script, plus a "
        "region where one is needed (`eng_Latn`, `zho_Hans`, `zho_Hant_TW`); the original codes "
        "stay in `src_lang_orig` and `tgt_lang_orig`. The `lp` column is the paper's short form: "
        "the first three letters of each side, except that a Simplified Chinese target is "
        "`zho_CN` and Traditional Chinese (Taiwan) is `zho_TW` (`eng-zho_CN`, `eng-zho_TW`, "
        "`zho-jpn`).",
    ),
    (
        "Reference types",
        "`ref_type` says how the reference was produced:\n"
        "\n"
        "| `ref_type` | source data | meaning |\n"
        "| --- | --- | --- |\n"
        "| `native` | `human` | A human translation from scratch. |\n"
        "| `postedit` | `postedit` | A post-edited machine translation. |\n"
        "| `pseudo` | `pseudo` | A machine-generated pseudo-reference. |\n"
        "| `none` | absent, null or `undefined` | The item has no reference, and `ref_text` is "
        "null. |\n"
        "\n"
        "`items.ref_type_orig` keeps the spelling of the source data, which tells an absent "
        "type (null) from an explicit `undefined`.",
    ),
    (
        "System types",
        "`system_type` classifies the system behind a hypothesis:\n"
        "\n"
        "| `system_type` | the `system` is |\n"
        "| --- | --- |\n"
        "| `mt` | A machine translation system. |\n"
        "| `human_scratch` | `Human (from scratch)`. |\n"
        "| `human_postedit` | `Human (postediting)`. |\n"
        "| `pseudo_ref` | `PseudoRef`, the pseudo-reference. |\n"
        "| `challenge_hyp` | A challenge-set hypothesis such as `FECE_hyp1`. |",
    ),
    (
        "Missing values",
        "Every column may be null; the description of each column says when. A run that does "
        "not cover an item has no rows for it (there are no null rows), and a run that makes no "
        "prediction for a task has null in that task's columns. Text values are copied exactly "
        "from the source data: nothing is trimmed, re-cased or clipped.",
    ),
)

_SIZE_CATEGORIES = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
    (100_000_000, "10M<n<100M"),
    (1_000_000_000, "100M<n<1B"),
    (10_000_000_000, "1B<n<10B"),
    (100_000_000_000, "10B<n<100B"),
    (1_000_000_000_000, "100B<n<1T"),
)

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_TYPE_NAMES = {"double": "float64", "float": "float32"}


# --- types and tables ----------------------------------------------------------------------------


def size_category(rows: int) -> str:
    """The Hub's ``size_categories`` label for ``rows`` rows (a count on a bound is above it)."""
    for bound, label in _SIZE_CATEGORIES:
        if rows < bound:
            return label
    return "n>1T"


def _type_name(type_: pa.DataType) -> str:
    """A short name for an Arrow type: ``string``, ``list<struct<start: int32, ...>>``."""
    if pa.types.is_list(type_):
        return f"list<{_type_name(type_.value_type)}>"
    if pa.types.is_struct(type_):
        fields = ", ".join(f"{f.name}: {_type_name(f.type)}" for f in type_)
        return f"struct<{fields}>"
    if pa.types.is_boolean(type_):
        return "bool"
    return _TYPE_NAMES.get(str(type_), str(type_))


def _cell(text: str) -> str:
    """``text`` made safe for a Markdown table cell."""
    return text.replace("|", "\\|")


def _field_table(schema: pa.Schema) -> str:
    """The ``column | type | description`` table of one config."""
    lines = ["| column | type | description |", "| --- | --- | --- |"]
    for field in schema:
        description = field.metadata[b"description"].decode("utf-8")
        lines.append(f"| `{field.name}` | `{_type_name(field.type)}` | {_cell(description)} |")
    return "\n".join(lines)


def _field_tables(level: int) -> str:
    """One heading and table per config, with the heading at Markdown ``level``."""
    blocks = []
    for config, schema in SCHEMAS.items():
        grain = CONFIG_GUIDE[config][0]
        blocks.append(f"{'#' * level} `{config}`\n\nOne row per {grain}.\n\n{_field_table(schema)}")
    return "\n\n".join(blocks)


def _conventions(level: int) -> str:
    """The shared conventions as sections whose headings are at Markdown ``level``."""
    return "\n\n".join(f"{'#' * level} {title}\n\n{body}" for title, body in CONVENTIONS)


# --- statistics ----------------------------------------------------------------------------------


def _count(mask: pa.ChunkedArray | pa.Array) -> int:
    """The number of true values in a Boolean array (nulls count as false)."""
    return pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0


def card_stats(items: pa.Table, pairs: pa.Table) -> dict:
    """The statistics of the card, counted from the built ``items`` and ``pairs`` tables.

    ``rows`` is not part of the result: ``build_all`` adds it once every config is written.  See
    the module docstring for the entries.
    """
    n_empty = _count(pairs.column("hyp_is_empty"))
    n_exact = _count(pc.equal(pairs.column("hyp"), ""))
    if n_exact > n_empty:
        raise ValueError("a hypothesis is exactly empty but not flagged hyp_is_empty")

    official_items = items.filter(pc.equal(items.column("collection"), "official"))
    official_pairs = pairs.filter(pc.equal(pairs.column("collection"), "official"))
    pairs_per_lp = {
        row["values"]: row["counts"]
        for row in pc.value_counts(official_pairs.column("lp")).to_pylist()
    }
    ids: dict[str, set[str]] = {}
    kinds: dict[str, set[str]] = {}
    priority: dict[str, bool] = {}
    for row in official_items.select(["lp", "item_id", "ref_type", "is_priority"]).to_pylist():
        ids.setdefault(row["lp"], set()).add(row["item_id"])
        kinds.setdefault(row["lp"], set()).add(row["ref_type"])
        priority[row["lp"]] = priority.get(row["lp"], True) and bool(row["is_priority"])
    coverage = [
        {
            "lp": lp,
            "src_items": len(ids[lp]),
            "pairs": pairs_per_lp.get(lp, 0),
            "ref_type": "+".join(sorted(kinds[lp])),
            "is_priority": priority[lp],
        }
        for lp in sorted(ids)
    ]

    codes = set()
    for column in ("src_lang", "tgt_lang"):
        codes.update(code[:3] for code in pc.unique(items.column(column)).to_pylist())
    return {
        "hyp_empty_exact": n_exact,
        "hyp_whitespace_only": n_empty - n_exact,
        "reference_candidates": _count(pairs.column("is_reference_candidate")),
        "coverage": coverage,
        "languages": sorted(codes),
    }


# --- the card ------------------------------------------------------------------------------------


def _config_table(rows: dict[str, int]) -> str:
    lines = ["| config | rows | one row per | contents |", "| --- | ---: | --- | --- |"]
    for config, (grain, contents) in CONFIG_GUIDE.items():
        n = rows["items"] if config == "grand" else rows[config]  # grand: one row per item
        lines.append(f"| `{config}` | {n:,} | {grain} | {_cell(contents)} |")
    return "\n".join(lines)


def _coverage_table(coverage: list[dict]) -> str:
    lines = [
        "| language pair | source items | pairs | reference | priority |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in coverage:
        priority = "yes" if row["is_priority"] else "no"
        lines.append(
            f"| `{row['lp']}` | {row['src_items']:,} | {row['pairs']:,} | {row['ref_type']} "
            f"| {priority} |"
        )
    items = sum(row["src_items"] for row in coverage)
    pairs = sum(row["pairs"] for row in coverage)
    lines.append(f"| **all official** | **{items:,}** | **{pairs:,}** | | |")
    return "\n".join(lines)


def _card_values(manifest: dict) -> dict[str, str]:
    """Every placeholder of the card template and the text that replaces it."""
    stats = manifest["stats"]
    rows = stats["rows"]
    coverage = stats["coverage"]
    n_official_items = sum(row["src_items"] for row in coverage)
    n_official_pairs = sum(row["pairs"] for row in coverage)
    n_exact, n_space = stats["hyp_empty_exact"], stats["hyp_whitespace_only"]
    n_empty = n_exact + n_space
    numbers = {
        "n_items": rows["items"],
        "n_official_items": n_official_items,
        "n_challenge_items": rows["items"] - n_official_items,
        "n_pairs": rows["pairs"],
        "n_official_pairs": n_official_pairs,
        "n_challenge_pairs": rows["pairs"] - n_official_pairs,
        "n_official_lps": len(coverage),
        "n_priority_lps": sum(row["is_priority"] for row in coverage),
        "n_hyp_empty": n_empty,
        "n_hyp_empty_exact": n_exact,
        "n_hyp_whitespace_only": n_space,
        "n_reference_candidates": stats["reference_candidates"],
        "n_submission_rows": rows["submissions"],
        "n_baseline_rows": rows["baselines"],
        "n_systems": rows["systems"],
        "n_mt_systems": rows["mt_systems"],
        "n_challenge_sets": rows["challenge_sets"],
    }
    counts = "\n".join(
        [
            f"- **{numbers['n_items']:,} items** in `items`: {n_official_items:,} official items "
            f"in {len(coverage)} language pairs and {numbers['n_challenge_items']:,} challenge "
            "items.",
            f"- **{numbers['n_pairs']:,} pairs** in `pairs`: {n_official_pairs:,} official and "
            f"{numbers['n_challenge_pairs']:,} challenge.",
            f"- **Empty or whitespace-only hypotheses: {n_empty:,}** ({n_exact:,} exactly "
            f"empty, {n_space:,} whitespace-only).",
            f"- **{stats['reference_candidates']:,} pairs are reference candidates**: the "
            "hypothesis equals the reference.",
            f"- **{numbers['n_submission_rows']:,} rows** in `submissions` and "
            f"**{numbers['n_baseline_rows']:,} rows** in `baselines`.",
            f"- **{numbers['n_systems']:,} evaluator runs** in `systems`, "
            f"**{numbers['n_mt_systems']:,} MT systems** in `mt_systems` and "
            f"**{numbers['n_challenge_sets']:,} challenge sets** in `challenge_sets`.",
        ]
    )
    values = {name: f"{n:,}" for name, n in numbers.items()}
    values.update(
        {
            "grand_sha256": manifest["grand_sha256"],
            "size_category": size_category(max(rows.values())),
            "language_list": "\n".join(f"- {code}" for code in stats["languages"]),
            "counts": counts,
            "config_table": _config_table(rows),
            "coverage_table": _coverage_table(coverage),
            "conventions": _conventions(3),
            "field_tables": _field_tables(3),
        }
    )
    return values


def render_card(template: Path, manifest: dict) -> str:
    """The dataset card: ``template`` with each ``{{name}}`` replaced from ``manifest``.

    ``manifest`` is a build manifest with its ``stats`` (see ``build_all``).  A placeholder that
    the card does not define raises ``ValueError``, so a typo cannot reach the Hub.
    """
    text = Path(template).read_text(encoding="utf-8")
    values = _card_values(manifest)

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"{Path(template).name}: unknown placeholder {{{{{name}}}}}")
        return values[name]

    return _PLACEHOLDER.sub(replace, text)


# --- generated documents -------------------------------------------------------------------------

_GRAND_SECTION = """\
The organisers' lossless archive of the shared task, kept unchanged next to the typed configs
(`wmt26-grand.parquet`, one row per test item). Its columns are `item_id` and `src` (text);
`ref`, `hyps` and `resources` (JSON); `ground_truth__record` (JSON), the raw human-annotation
records; one JSON column for each submission and each baseline snapshot, named by the
`grand_column` of [`data/MANIFEST.tsv`](../data/MANIFEST.tsv); and `source_provenance` (JSON), the
source file and line behind every column of the row. Source files with identical content are
stored once: `canonical_column` in the manifest names the column that holds them, and
`__item_fields__` stands for the five item columns. Loading the file needs `datasets>=4.7.0`,
which reads the JSON columns as dicts. The baseline diagnostic `stage1_finish_reason`, which some
baseline runs record, is dropped from the typed configs and appears only in this config."""


def _language_table(lang_codes_path: Path) -> str:
    """The table of language codes that are mapped to a different normalised code."""
    with open(lang_codes_path, encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["orig"] != r["normalised"]]
    lines = ["| code in the id | normalised |", "| --- | --- |"]
    lines += [f"| `{r['orig']}` | `{r['normalised']}` |" for r in rows]
    return "\n".join(lines)


def render_format_md(lang_codes_path: Path = DEFAULT_LANG_CODES_PATH) -> str:
    """``docs/FORMAT.md``: the conventions, then one table per config from ``SCHEMAS``."""
    return (
        "# Data format\n"
        "\n"
        "<!-- Generated by scripts/render_docs.py from src/wmt26_mteval/schemas.py and "
        "data/metadata/lang_codes.tsv. Do not edit by hand. -->\n"
        "\n"
        "Columns, types and conventions of the configs in `wmt/wmt26-mteval-data`. This file is "
        "generated by `scripts/render_docs.py`; the same descriptions are stored in the Parquet "
        "files and printed on the dataset card.\n"
        "\n"
        "## Conventions\n"
        "\n"
        f"{_conventions(3)}\n"
        "\n"
        "### Language codes found in challenge-set ids\n"
        "\n"
        "Two-letter and locale codes occur only in the ids of the challenge sets. The mapping "
        "below was chosen by the release builders, not by the organisers: the script or variant "
        "behind a bare code (`zh` as Simplified Chinese, `ar` as Standard Arabic) is an "
        "assumption.\n"
        "\n"
        f"{_language_table(lang_codes_path)}\n"
        "\n"
        f"{_field_tables(2)}\n"
        "\n"
        f"## `grand`\n\n{_GRAND_SECTION}\n"
    )


def render_manifest_tsv(grand: Grand) -> str:
    """``data/MANIFEST.tsv``: the source files recorded in the grand file, in ``source_id`` order.

    The columns are ``MANIFEST_COLUMNS``: the path of the source file, its size, sha256 and number
    of records, the name of its column in the grand file and the column that holds the same data
    when several files are identical or superseded.
    """
    out = io.StringIO()
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for entry in grand.manifest:
        writer.writerow(
            [
                entry.path,
                entry.bytes,
                entry.sha256,
                entry.records,
                entry.column,
                entry.canonical_column,
            ]
        )
    return out.getvalue()
