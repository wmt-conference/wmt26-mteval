"""The ``systems``, ``mt_systems`` and ``challenge_sets`` tables: what the other configs refer to.

``systems`` has one row per official run, typed from the scrubbed model cards and the curated
``run_names.tsv`` that are committed to the repository (never from the organisers' raw drop), plus
the coverage of the run's own table.  ``mt_systems`` lists the systems that have hypotheses in
``pairs`` and ``challenge_sets`` describes the seven challenge sets from ``challenge_sets.yaml``.
Each builder fails loudly on a gap, such as a run without a card, instead of writing a null row.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import yaml

from wmt26_mteval.cards import normalise_bool, parse_threshold
from wmt26_mteval.context import Context
from wmt26_mteval.ids import SEPARATOR
from wmt26_mteval.runs import RunSpec
from wmt26_mteval.schemas import SCHEMAS

RUN_NAMES_PATH = Path("data") / "metadata" / "run_names.tsv"
CARDS_DIR = Path("data") / "model_cards"
CHALLENGE_SETS_PATH = Path("data") / "metadata" / "challenge_sets.yaml"

RUN_NAMES_HEADER = ["run_id", "canonical_name", "underlying_model", "paper_citation"]
OFFICIAL_PREFIX = f"official{SEPARATOR}"

# Card answers copied as they are (a missing answer is null), and card answers read as yes or no.
_TEXT_KEYS = (
    "org",
    "description",
    "reference_usage",
    "model_size",
    "closed_model_name",
    "api_provider",
    "link",
)
_BOOL_KEYS = (
    "uses_reference",
    "ensemble",
    "fine_tuned",
    "open_weight",
    "open_source",
    "allow_t1_to_t2",
    "allow_t1_to_t3",
    "allow_t2_to_t3",
)
_CHALLENGE_TEXT_FIELDS = frozenset(
    {
        "name",
        "full_name",
        "authors",
        "citation",
        "what_it_tests",
        "hyp_semantics",
        "ref_semantics",
    }
)


@dataclass(frozen=True)
class Coverage:
    """What one run covers: its distinct items, its rows, its language pairs and the priority."""

    n_items: int
    n_pairs: int
    lps: list[str]
    covers_all_priority: bool


def _table(schema: pa.Schema, rows: list[dict]) -> pa.Table:
    """Arrow table with exactly ``schema``; a row missing one of its columns is a ``KeyError``."""
    columns = [pa.array([row[field.name] for row in rows], type=field.type) for field in schema]
    return pa.Table.from_arrays(columns, schema=schema)


def priority_lps(ctx: Context) -> set[str]:
    """The priority directions as the short ``src-tgt`` codes that the ``lp`` column uses."""
    lps = set()
    for direction in ctx.priority:
        src, tgt = direction.split("-")
        lps.add(f"{ctx.lang.short(src, 'src')}-{ctx.lang.short(tgt, 'tgt')}")
    return lps


def coverage_of(run_table: pa.Table, priority_lps: set[str]) -> Coverage:
    """Coverage of the table of one run.

    ``n_items`` counts distinct ``item_id`` values, ``n_pairs`` counts rows and ``lps`` holds every
    ``lp`` in the table.  ``covers_all_priority`` asks for every priority language pair among the
    run's official rows alone: rows of challenge items (RefQuality ones have priority pairs too)
    do not count.
    """
    item_ids = run_table.column("item_id")
    lp = run_table.column("lp")
    official = pc.starts_with(item_ids, OFFICIAL_PREFIX)
    official_lps = set(pc.unique(pc.filter(lp, official)).to_pylist())
    return Coverage(
        n_items=pc.count_distinct(item_ids).as_py(),
        n_pairs=run_table.num_rows,
        lps=sorted(set(pc.unique(lp).to_pylist())),
        covers_all_priority=set(priority_lps) <= official_lps,
    )


def _read_run_names(repo: Path) -> dict[str, dict[str, str]]:
    """``run_names.tsv`` as ``{run_id: row}``; a wrong header or a repeated run id raises."""
    path = Path(repo) / RUN_NAMES_PATH
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != RUN_NAMES_HEADER:
            raise ValueError(f"{path}: header must be {RUN_NAMES_HEADER}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            if row["run_id"] in rows:
                raise ValueError(f"{path}: run id {row['run_id']!r} appears twice")
            rows[row["run_id"]] = row
    return rows


def _read_card(repo: Path, run_id: str) -> dict[str, str]:
    """The scrubbed answers of ``run_id``; a missing or foreign card raises."""
    path = Path(repo) / CARDS_DIR / f"{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no scrubbed model card for run {run_id!r}: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("run_id") != run_id:
        raise ValueError(f"{path}: the card is for run {document.get('run_id')!r}, not {run_id!r}")
    answers = document["answers"]
    if not all(isinstance(text, str) for text in answers.values()):
        raise ValueError(f"{path}: every card answer must be a string")
    return answers


def _tasks(run: RunSpec) -> list[str]:
    """Tasks a run has predictions for.  Task 1 baselines carry Task 2 scores as well."""
    if run.kind == "submission":
        return [run.task]
    if run.task == "task1":
        return ["task1", "task2"]
    if run.task == "task2":
        return ["task2"]
    raise ValueError(f"baseline {run.run_id!r} has the unexpected task {run.task!r}")


def build_systems(repo: Path, runs: list[RunSpec], coverage: Mapping[str, Coverage]) -> pa.Table:
    """One row per ``official`` run, sorted by ``run_id``.

    The answers come from the committed ``data/model_cards/<run_id>.json`` and the curated names
    from ``data/metadata/run_names.tsv`` under ``repo``.  Yes/no answers become Booleans (null when
    missing or unclear), the Task 2 to 3 threshold comes from ``parse_threshold`` and the free-text
    answers are copied unchanged (null when missing).  A card ``name`` that more than one official
    run gives is written as ``"<name> (<run_id>)"``, so that every ``name`` is unique.

    A run without a card, a ``run_names.tsv`` row or an entry in ``coverage`` (keyed by run id),
    two official runs with one run id, and names that stay ambiguous all raise.
    """
    official = sorted((run for run in runs if run.variant == "official"), key=lambda r: r.run_id)
    repeated = sorted(run_id for run_id, n in Counter(r.run_id for r in official).items() if n > 1)
    if repeated:
        raise ValueError(f"two official runs share the run id(s) {repeated}")

    curated = _read_run_names(repo)
    cards = {run.run_id: _read_card(repo, run.run_id) for run in official}
    card_names = Counter(card["name"] for card in cards.values() if "name" in card)
    shared = {name for name, n in card_names.items() if n > 1}

    rows = []
    for run in official:
        run_id = run.run_id
        if run_id not in curated:
            raise ValueError(f"run {run_id!r} has no row in {RUN_NAMES_PATH.name}")
        if run_id not in coverage:
            raise ValueError(f"run {run_id!r} has no entry in the coverage mapping")
        answers, names, covered = cards[run_id], curated[run_id], coverage[run_id]
        name = answers.get("name")
        threshold, inclusive = parse_threshold(
            answers.get("t3_threshold", ""), answers.get("threshold_inclusive", "")
        )
        rows.append(
            {
                "run_id": run_id,
                "name": f"{name} ({run_id})" if name in shared else name,
                "canonical_name": names["canonical_name"],
                **{key: answers.get(key) for key in _TEXT_KEYS},
                "underlying_model": names["underlying_model"],
                "paper_citation": names["paper_citation"],
                "is_baseline": run.kind == "baseline",
                **{key: normalise_bool(answers.get(key, "")) for key in _BOOL_KEYS},
                "threshold_inclusive": inclusive,
                "covers_all_priority": covered.covers_all_priority,
                "t3_threshold": threshold,
                "tasks": _tasks(run),
                "lps": list(covered.lps),
                "n_items": covered.n_items,
                "n_pairs": covered.n_pairs,
            }
        )

    ambiguous = sorted(
        name
        for name, n in Counter(row["name"] for row in rows).items()
        if name is not None and n > 1
    )
    if ambiguous:
        raise ValueError(f"system names stay ambiguous after adding run ids: {ambiguous}")
    return _table(SCHEMAS["systems"], rows)


def build_mt_systems(pairs: pa.Table) -> pa.Table:
    """One row per distinct ``system`` in ``pairs``, sorted by ``system``.

    ``n_items`` counts distinct items and ``lps`` lists the distinct language pairs.  A system
    that appears with two ``system_type`` values raises ``ValueError``.
    """
    grouped = pairs.group_by("system").aggregate(
        [("system_type", "distinct"), ("item_id", "count_distinct"), ("lp", "distinct")]
    )
    rows = []
    for row in grouped.to_pylist():
        types = row["system_type_distinct"]
        if len(types) != 1:
            raise ValueError(f"system {row['system']!r} has several system types: {sorted(types)}")
        rows.append(
            {
                "system": row["system"],
                "system_type": types[0],
                "n_items": row["item_id_count_distinct"],
                "lps": sorted(row["lp_distinct"]),
            }
        )
    rows.sort(key=lambda r: r["system"])
    return _table(SCHEMAS["mt_systems"], rows)


def _read_challenge_sets(yaml_path: Path) -> list[dict[str, str]]:
    """The entries of ``challenge_sets.yaml``, each with exactly the text columns."""
    entries = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"{yaml_path}: expected a list of challenge sets")
    for entry in entries:
        label = entry.get("name") if isinstance(entry, dict) else entry
        if not isinstance(entry, dict) or entry.keys() != _CHALLENGE_TEXT_FIELDS:
            raise ValueError(
                f"{yaml_path}: entry {label!r} must have exactly the keys "
                f"{sorted(_CHALLENGE_TEXT_FIELDS)}"
            )
        if not all(isinstance(text, str) and text.strip() for text in entry.values()):
            raise ValueError(f"{yaml_path}: every field of entry {label!r} must be non-empty text")
    names = Counter(entry["name"] for entry in entries)
    repeated = sorted(name for name, n in names.items() if n > 1)
    if repeated:
        raise ValueError(f"{yaml_path}: challenge set(s) {repeated} appear twice")
    return entries


def build_challenge_sets(yaml_path: Path, pairs: pa.Table) -> pa.Table:
    """One row per entry of ``challenge_sets.yaml``, sorted by ``name``.

    ``n_items``, ``n_pairs`` and ``lps`` are counted from the ``pairs`` rows of that set, and are
    0 and an empty list for a set that has none.  A ``challenge_set`` in ``pairs`` that the YAML
    does not describe raises ``ValueError``.
    """
    entries = _read_challenge_sets(yaml_path)
    known = {entry["name"] for entry in entries}
    in_sets = pairs.filter(pc.is_valid(pairs.column("challenge_set")))
    grouped = in_sets.group_by("challenge_set").aggregate(
        [("item_id", "count_distinct"), ("item_id", "count"), ("lp", "distinct")]
    )
    counts = {row["challenge_set"]: row for row in grouped.to_pylist()}
    unknown = sorted(counts.keys() - known)
    if unknown:
        raise ValueError(
            f"pairs hold challenge set(s) missing from {Path(yaml_path).name}: {unknown}"
        )

    rows = []
    for entry in sorted(entries, key=lambda e: e["name"]):
        found = counts.get(entry["name"])
        rows.append(
            {
                **entry,
                "lps": [] if found is None else sorted(found["lp_distinct"]),
                "n_items": 0 if found is None else found["item_id_count_distinct"],
                "n_pairs": 0 if found is None else found["item_id_count"],
            }
        )
    return _table(SCHEMAS["challenge_sets"], rows)
