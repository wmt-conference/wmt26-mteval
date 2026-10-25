"""Validation gates for a staged release: nothing is uploaded unless all of them pass.

``validate`` runs the gates below over a build made by ``build_all`` and returns the failures as
plain messages (an empty list means every gate passed):

1. ``check_required_files``: the dataset card ``README.md`` and ``build_manifest.json`` exist;
2. ``check_row_counts``: the rows of each config on disk equal what is expected;
3. ``check_table2``: the official items and pairs per language pair equal Table 2 of the findings
   paper (``data/metadata/table2.tsv``);
4. ``find_pii``: the metadata-derived outputs hold no personal data;
5. ``check_schemas``: every Parquet file has exactly the columns of ``schemas.SCHEMAS``;
6. ``check_item_texts``: the text columns of ``items`` and ``pairs`` equal the grand file, item by
   item and pair by pair (``src``, ``ref_text``, ``hyp``, ``asr``, ``video_path``,
   ``screenshot_path`` and ``n_systems``);
7. ``check_keys``: every ``(item_id, system)`` of ``submissions`` and ``baselines`` is a row of
   ``pairs``, and every ``run_id`` of both is a row of ``systems``;
8. ``check_alt_task2``: the Task 2 scores of every ``alt`` baseline equal those of its ``official``
   counterpart in the grand file, so that the ``alt`` tables lose nothing by keeping Task 1 only;
9. ``check_round_trip``: every run file, folded back into records, equals its column of the grand
   file, so that the prediction tables lose nothing.

What is compared with the grand file, then, is everything the grand file holds that the typed
configs copy: the item fields (gate 6) and the predictions of every run (gate 9).  The columns that
the builder derives from them (``item_id`` fields, ``ref_type``, ``is_reference_candidate`` and the
like) are checked by Table 2, the row counts and the schemas.  ``systems``, ``mt_systems`` and
``challenge_sets`` are built from curated tables and scrubbed model cards, not from the grand file,
and are covered by the row counts, the schemas, the personal-data scan and the key check.

The gates read the Parquet files on disk, never the build manifest, and know the file naming of
the build only from the documented layout (``data/<config>/test-*.parquet``), so a mistake in the
builder cannot also hide itself here.  A failure message names the file and column concerned and
never quotes personal data or the text of a source, reference or hypothesis.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wmt26_mteval.build import MANIFEST_NAME, README_NAME
from wmt26_mteval.cards import EMAIL_RE
from wmt26_mteval.grand import Grand
from wmt26_mteval.runs import RunSpec, list_runs
from wmt26_mteval.schemas import SCHEMAS
from wmt26_mteval.tables.predictions import T1, T2, normalise_raw_record, roundtrip_record

DEFAULT_TABLE2_PATH = Path(__file__).resolve().parents[2] / "data" / "metadata" / "table2.tsv"

# The only address that may appear in the metadata-derived outputs: the organisers' mailing list.
ALLOWED_EMAILS = frozenset({"wmt-qe-metrics-organizers@googlegroups.com"})

# What the raw model cards call the personal data that the release must never carry.
PERSONAL_DATA_LABELS = ("Participant name", "Email", "Data Collection Consent")

# The metadata configs that the PII gate scans.  The text of the other configs is test data, in
# which e-mail addresses are legitimate (social-media posts).
_PII_CONFIGS = ("systems", "challenge_sets")

# Failing items shown per run before the rest are summarised in one line.
MAX_ITEM_FAILURES = 5


def _config_files(out_dir: Path, config: str) -> list[Path]:
    """The Parquet files of ``config`` in ``out_dir``, sorted by name."""
    return sorted((out_dir / "data" / config).glob("*.parquet"))


def _relative(out_dir: Path, path: Path) -> str:
    return path.relative_to(out_dir).as_posix()


class UnreadableFile(ValueError):
    """A Parquet file could not be read; the message names it."""


@contextmanager
def _readable(out_dir: Path, path: Path) -> Iterator[None]:
    """Turn an error raised while reading ``path`` into an ``UnreadableFile`` that names it."""
    try:
        yield
    except (OSError, pa.ArrowException) as error:
        name = _relative(out_dir, path)
        raise UnreadableFile(f"{name}: cannot be read: {type(error).__name__}: {error}") from error


def _read_columns(out_dir: Path, config: str, columns: list[str]) -> pa.Table:
    """Just ``columns`` of every file of ``config``, concatenated (empty without files)."""
    tables = []
    for path in _config_files(out_dir, config):
        with _readable(out_dir, path):
            tables.append(pq.read_table(path, columns=columns))
    if not tables:
        return pa.table({name: pa.array([], pa.string()) for name in columns})
    return pa.concat_tables(tables)


# --- gate 1: the files every build has -----------------------------------------------------------


def check_required_files(out_dir: Path) -> list[str]:
    """Failures for a build that lacks its dataset card or its manifest.

    ``build_all`` writes ``README.md`` and, last of all, ``build_manifest.json``; a directory
    without either is not a finished build.  The other gates skip a file that is not there (the
    personal-data scan does), so this one makes the absence a failure.
    """
    return [
        f"{name}: file is missing"
        for name in (README_NAME, MANIFEST_NAME)
        if not (Path(out_dir) / name).is_file()
    ]


# --- gate 2: row counts --------------------------------------------------------------------------


def check_row_counts(out_dir: Path, expected: dict) -> list[str]:
    """Compare the rows on disk with ``expected["rows"]`` and, if given, ``pairs_by_collection``."""
    failures = []
    for config, want in expected["rows"].items():
        got = 0
        for path in _config_files(out_dir, config):
            with _readable(out_dir, path):
                got += pq.ParquetFile(path).metadata.num_rows
        if got != want:
            failures.append(f"{config}: {got} rows on disk, expected {want}")

    by_collection = expected.get("pairs_by_collection")
    if by_collection is not None:
        found = Counter(_read_columns(out_dir, "pairs", ["collection"]).column(0).to_pylist())
        for collection in sorted(found.keys() | by_collection.keys()):
            got, want = found.get(collection, 0), by_collection.get(collection, 0)
            if got != want:
                failures.append(
                    f"pairs: {got} rows of the {collection!r} collection on disk, expected {want}"
                )
    return failures


# --- gate 3: Table 2 -----------------------------------------------------------------------------


def load_table2(path: Path) -> dict[str, tuple[int, int]]:
    """``lp -> (source items, pairs)`` from the tab-separated ``lp / src_items / pairs`` file."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["lp", "src_items", "pairs"]:
            raise ValueError(f"{path}: header must be 'lp<TAB>src_items<TAB>pairs'")
        table: dict[str, tuple[int, int]] = {}
        for row in reader:
            if row["lp"] in table:
                raise ValueError(f"{path}: duplicate language pair {row['lp']!r}")
            table[row["lp"]] = (int(row["src_items"]), int(row["pairs"]))
    return table


def check_table2(out_dir: Path, table2_path: Path = DEFAULT_TABLE2_PATH) -> list[str]:
    """Compare the official items and pair rows per ``lp`` with Table 2 of the paper.

    Items are the distinct official ``item_id`` values of ``items``; pairs are the official rows
    of ``pairs``.  Challenge sets are not part of Table 2.
    """
    want = load_table2(table2_path)
    items = _read_columns(out_dir, "items", ["collection", "lp", "item_id"])
    pairs = _read_columns(out_dir, "pairs", ["collection", "lp"])
    official_items = items.filter(pc.equal(items.column("collection"), "official"))
    official_pairs = pairs.filter(pc.equal(pairs.column("collection"), "official"))
    got_items = Counter(
        lp
        for lp, _ in {
            (row["lp"], row["item_id"])
            for row in official_items.select(["lp", "item_id"]).to_pylist()
        }
    )
    got_pairs = Counter(official_pairs.column("lp").to_pylist())

    failures = []
    for lp in sorted(want.keys() | got_items.keys() | got_pairs.keys()):
        if lp not in want:
            failures.append(
                f"table2 {lp}: official language pair is not in Table 2 "
                f"({got_items[lp]} items, {got_pairs[lp]} pairs)"
            )
            continue
        want_items, want_pairs = want[lp]
        if got_items[lp] != want_items:
            failures.append(f"table2 {lp}: {got_items[lp]} official items, expected {want_items}")
        if got_pairs[lp] != want_pairs:
            failures.append(f"table2 {lp}: {got_pairs[lp]} official pairs, expected {want_pairs}")
    return failures


# --- gate 4: personal data -----------------------------------------------------------------------


def _strings(value) -> Iterator[str]:
    """Every string in a cell: the cell itself, or the strings inside a list or struct."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)


def _has_forbidden_email(text: str) -> bool:
    return any(m.group(0).lower() not in ALLOWED_EMAILS for m in EMAIL_RE.finditer(text))


def _text_failures(name: str, text: str, labels: bool) -> list[str]:
    """Failures for one text file; ``labels`` also looks for the personal-data labels."""
    failures = []
    if _has_forbidden_email(text):
        failures.append(f"{name}: contains an e-mail address")
    if labels:
        failures += [
            f"{name}: contains the personal-data label {label!r}"
            for label in PERSONAL_DATA_LABELS
            if label in text
        ]
    return failures


def _parquet_failures(out_dir: Path, path: Path) -> list[str]:
    """Failures for one Parquet file: each column is reported once per kind of finding."""
    name = _relative(out_dir, path)
    with _readable(out_dir, path):
        table = pq.read_table(path)
    failures = []
    for column in table.column_names:
        emails = 0
        labels: Counter[str] = Counter()
        for text in (s for cell in table.column(column).to_pylist() for s in _strings(cell)):
            if _has_forbidden_email(text):
                emails += 1
            labels.update(label for label in PERSONAL_DATA_LABELS if label in text)
        if emails:
            failures.append(
                f"{name}: column {column!r}: {emails} value(s) contain an e-mail address"
            )
        for label in PERSONAL_DATA_LABELS:
            if labels[label]:
                failures.append(
                    f"{name}: column {column!r}: {labels[label]} value(s) contain the "
                    f"personal-data label {label!r}"
                )
    return failures


def find_pii(out_dir: Path) -> list[str]:
    """Personal data left in the metadata-derived outputs of ``out_dir``; empty means clean.

    Only ``data/systems/``, ``data/challenge_sets/``, ``README.md`` and ``build_manifest.json`` are
    scanned, and paths that do not exist are skipped.  The text of items and hypotheses is never
    scanned, since real e-mail addresses legitimately occur in the test data.

    - ``EMAIL_RE`` is checked in all four, apart from the addresses in ``ALLOWED_EMAILS``;
    - the labels in ``PERSONAL_DATA_LABELS`` are checked in the two Parquet configs and the
      manifest, but not in the README, whose prose may describe the model-card form.

    Every string of a Parquet cell is scanned, those in list cells included.  A message names the
    file and column and never repeats the text it found.
    """
    out_dir = Path(out_dir)
    failures = []
    for config in _PII_CONFIGS:
        for path in _config_files(out_dir, config):
            failures += _parquet_failures(out_dir, path)
    for name, labels in ((README_NAME, False), (MANIFEST_NAME, True)):
        path = out_dir / name
        if path.is_file():
            failures += _text_failures(name, path.read_text(encoding="utf-8"), labels)
    return failures


# --- gate 5: schemas -----------------------------------------------------------------------------


def _normalised(type_: pa.DataType) -> pa.DataType:
    """``type_`` with every list's child field called ``item``.

    Parquet hands list children back as ``element`` where the in-memory schemas say ``item``;
    the name carries no meaning, so both sides are compared with one name, recursively.
    """
    if pa.types.is_list(type_):
        child = type_.value_field
        return pa.list_(pa.field("item", _normalised(child.type), nullable=child.nullable))
    if pa.types.is_struct(type_):
        return pa.struct(
            [pa.field(f.name, _normalised(f.type), nullable=f.nullable) for f in type_]
        )
    return type_


def _description(field: pa.Field) -> bytes | None:
    return (field.metadata or {}).get(b"description")


def schema_failures(name: str, actual: pa.Schema, want: pa.Schema) -> list[str]:
    """How ``actual`` differs from ``want`` in column names, order, types, nullability and text.

    The comparison is field by field, because Parquet does not keep list child names and a
    plain ``Schema.equals`` would fail on them.  The ``description`` metadata is compared; any
    other field metadata is not.
    """
    missing = [n for n in want.names if n not in actual.names]
    unexpected = [n for n in actual.names if n not in want.names]
    failures = [f"{name}: column {n!r} is missing" for n in missing]
    failures += [f"{name}: column {n!r} is not in the schema" for n in unexpected]
    if not missing and not unexpected and actual.names != want.names:
        failures.append(f"{name}: columns are in a different order than the schema")
    for field in want:
        if field.name not in actual.names:
            continue
        found = actual.field(field.name)
        if _normalised(found.type) != _normalised(field.type):
            failures.append(
                f"{name}: column {field.name!r}: type is {found.type}, not {field.type}"
            )
        if found.nullable != field.nullable:
            failures.append(
                f"{name}: column {field.name!r}: nullable is {found.nullable}, not {field.nullable}"
            )
        if _description(found) != _description(field):
            failures.append(f"{name}: column {field.name!r}: description differs from the schema")
    return failures


def check_schemas(out_dir: Path) -> list[str]:
    """Check every Parquet file of every config against ``SCHEMAS`` (``schema_failures``)."""
    failures = []
    for config, want in SCHEMAS.items():
        files = _config_files(out_dir, config)
        if not files:
            failures.append(f"data/{config}: no Parquet files")
        for path in files:
            with _readable(out_dir, path):
                actual = pq.read_schema(path)
            failures += schema_failures(_relative(out_dir, path), actual, want)
    return failures


# --- gate 6: items and pairs against the grand file ----------------------------------------------

# The columns of ``items`` and of ``pairs`` that copy a value of the grand file.
_ITEM_VALUES = ("src", "ref_text", "asr", "video_path", "screenshot_path", "n_systems")
_PAIR_VALUES = ("src", "hyp", "ref_text")


def _grand_item_values(item: dict) -> dict:
    """The values of one grand item that ``items`` copies, under the names of its columns."""
    ref, resources = item["ref"], item["resources"] or {}
    return {
        "src": item["src"],
        "ref_text": None if ref is None else ref.get("text"),
        "asr": resources.get("asr"),
        "video_path": resources.get("video"),
        "screenshot_path": resources.get("screenshot"),
        "n_systems": len(item["hyps"]),
    }


def _digest(text: str | None) -> bytes | None:
    """An 8-byte fingerprint of ``text``, so that a pair is remembered without its text."""
    if text is None:
        return None
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest()


def _read_item_rows(out_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """``items`` as ``item_id -> its values``, and the items that have more than one row."""
    table = _read_columns(out_dir, "items", ["item_id", *_ITEM_VALUES])
    rows: dict[str, dict] = {}
    repeated: list[str] = []
    for row in table.to_pylist():
        if row["item_id"] in rows:
            repeated.append(row["item_id"])
        rows[row["item_id"]] = row
    return rows, repeated


def _read_pair_digests(out_dir: Path) -> tuple[dict[tuple[str, str], tuple], list[tuple[str, str]]]:
    """``pairs`` as ``(item_id, system) -> fingerprints of src, hyp and ref_text``.

    One file is held as Python objects at a time and only the fingerprints are kept.  The second
    result lists the keys that more than one row has.
    """
    digests: dict[tuple[str, str], tuple] = {}
    repeated: list[tuple[str, str]] = []
    for path in _config_files(out_dir, "pairs"):
        with _readable(out_dir, path):
            table = pq.read_table(path, columns=["item_id", "system", *_PAIR_VALUES])
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        for i, (item_id, system) in enumerate(zip(columns["item_id"], columns["system"])):
            key = (sys.intern(item_id), sys.intern(system))
            if key in digests:
                repeated.append(key)
            digests[key] = tuple(_digest(columns[name][i]) for name in _PAIR_VALUES)
    return digests, repeated


def check_item_texts(out_dir: Path, grand: Grand) -> list[str]:
    """``items`` and ``pairs`` against the grand file, for every item and every pair.

    - ``items`` must have exactly one row for each item of the grand file, and no other, with the
      grand values of ``src``, ``ref_text``, ``asr``, ``video_path``, ``screenshot_path`` and
      ``n_systems`` (the number of hypotheses);
    - ``pairs`` must have exactly one row for each hypothesis of each item, and no other, with the
      grand ``hyp``, ``src`` and ``ref_text``.

    Texts are compared to the character (nothing is trimmed) and a null is different from an
    empty text.  The grand file is read once, in a stream.  ``pairs`` is held as fingerprints of
    its texts, one file's rows as objects at a time, so the memory stays bounded.  A failure names
    the item, the system and the columns, and never quotes a text.
    """
    out_dir = Path(out_dir)
    rows, repeated_items = _read_item_rows(out_dir)
    digests, repeated_pairs = _read_pair_digests(out_dir)
    item_problems = [f"item {item_id!r}: listed more than once" for item_id in repeated_items]
    pair_problems = [
        f"item {item_id!r}, system {system!r}: listed more than once"
        for item_id, system in repeated_pairs
    ]

    for item in grand.items():
        item_id = item["item_id"]
        want = _grand_item_values(item)
        row = rows.pop(item_id, None)
        if row is None:
            item_problems.append(f"item {item_id!r}: no row")
        else:
            differ = [name for name in _ITEM_VALUES if row[name] != want[name]]
            if differ:
                item_problems.append(
                    f"item {item_id!r}: {', '.join(differ)} differ from the grand file"
                )
        src, ref = _digest(want["src"]), _digest(want["ref_text"])
        for system, hyp in item["hyps"].items():
            got = digests.pop((item_id, system), None)
            if got is None:
                pair_problems.append(f"item {item_id!r}, system {system!r}: no row")
                continue
            differ = [
                name
                for name, found, wanted in zip(_PAIR_VALUES, got, (src, _digest(hyp), ref))
                if found != wanted
            ]
            if differ:
                pair_problems.append(
                    f"item {item_id!r}, system {system!r}: {', '.join(differ)} differ from the "
                    "grand file"
                )

    item_problems += [
        f"item {item_id!r}: row of an item that is not in the grand file"
        for item_id in sorted(rows)
    ]
    pair_problems += [
        f"item {item_id!r}, system {system!r}: row of a pair that is not in the grand file"
        for item_id, system in sorted(digests)
    ]
    return _capped("data/items", item_problems) + _capped("data/pairs", pair_problems)


# --- gate 7: keys across configs -----------------------------------------------------------------

_KEY_COLUMNS = ["run_id", "item_id", "system"]


def check_keys(out_dir: Path) -> list[str]:
    """Every prediction points at a pair and at a run that the release lists.

    - each ``(item_id, system)`` of every ``submissions`` and ``baselines`` file is a row of
      ``pairs``;
    - each ``run_id`` of those files is a row of ``systems``.

    A file that lacks one of the three columns is skipped here, since the schema gate reports it.
    The files are read one at a time and only their key columns.
    """
    out_dir = Path(out_dir)
    pairs = _read_columns(out_dir, "pairs", ["item_id", "system"])
    known_runs = set(_read_columns(out_dir, "systems", ["run_id"]).column(0).to_pylist())
    failures = []
    for config in ("submissions", "baselines"):
        for path in _config_files(out_dir, config):
            with _readable(out_dir, path):
                if not set(_KEY_COLUMNS) <= set(pq.read_schema(path).names):
                    continue
                table = pq.read_table(path, columns=_KEY_COLUMNS)
            run_ids = pc.unique(table.column("run_id")).to_pylist()
            problems = [
                f"run_id {run_id!r} is not in systems"
                for run_id in sorted((r for r in run_ids if r not in known_runs), key=str)
            ]
            keys = table.select(["item_id", "system"])
            lost = keys.join(pairs, keys=["item_id", "system"], join_type="left anti")
            first = lost.sort_by([("item_id", "ascending"), ("system", "ascending")]).slice(
                0, MAX_ITEM_FAILURES
            )
            problems += [
                f"item {row['item_id']!r}, system {row['system']!r}: no such pair in pairs"
                for row in first.to_pylist()
            ]
            failures += _capped(
                _relative(out_dir, path), problems, total=len(problems) - len(first) + lost.num_rows
            )
    return failures


# --- gate 8: the Task 2 scores of the alt baselines ----------------------------------------------


def _task2_scores(grand: Grand, run: RunSpec) -> Iterator[tuple[str, dict[str, float]]]:
    """``(item_id, its Task 2 scores)`` for each item the run covers; an empty dict for none."""
    for item_id, raw in grand.iter_records(run.column):
        yield item_id, normalise_raw_record(raw).get(T2, {})


def _task2_differences(grand: Grand, alt: RunSpec, official: RunSpec) -> list[str]:
    scores = {item_id: s for item_id, s in _task2_scores(grand, alt) if s}
    problems = []
    for item_id, want in _task2_scores(grand, official):
        if scores.pop(item_id, {}) != want:
            problems.append(f"item {item_id!r}: Task 2 scores differ from the official baseline")
    problems += [
        f"item {item_id!r}: Task 2 scores that the official baseline does not have"
        for item_id in sorted(scores)
    ]
    return problems


def check_alt_task2(grand: Grand) -> list[str]:
    """For every ``alt`` baseline, its Task 2 scores in the grand file equal its official ones.

    An ``alt`` table keeps Task 1 only, because the Task 2 scores of the original snapshot are
    those of the official one.  This is the check that dropping them loses nothing: the two grand
    columns must have the same scores for every item and system (an item with no scores in one of
    them must have none in the other).  The run files are not read.
    """
    runs = list_runs(grand)
    official = {r.run_id: r for r in runs if r.kind == "baseline" and r.variant == "official"}
    failures = []
    for alt in (r for r in runs if r.variant == "alt"):
        name = run_file(alt)
        counterpart = official.get(alt.run_id)
        if counterpart is None:
            failures.append(f"{name}: there is no official baseline to compare its Task 2 with")
            continue
        failures += _capped(name, _task2_differences(grand, alt, counterpart))
    return failures


# --- gate 9: round trip --------------------------------------------------------------------------


def run_file(run: RunSpec) -> str:
    """The path, relative to the build, at which the table of ``run`` is expected."""
    if run.kind == "submission":
        return f"data/submissions/test-{run.task}-{run.run_id}.parquet"
    return f"data/baselines/test-{run.run_id}-{run.variant}.parquet"


def _label_problems(table: pa.Table, run: RunSpec) -> list[str]:
    """Problems with the constant columns, which must hold the run's own labels."""
    labels = {"run_id": run.run_id, "task": run.task}
    if run.kind == "baseline":
        labels["variant"] = run.variant
    problems = []
    for column, value in labels.items():
        if column not in table.column_names:
            continue  # the schema gate reports the missing column
        if pc.unique(table.column(column)).to_pylist() != [value]:
            problems.append(f"column {column!r} is not {value!r} in every row")
    return problems


def _round_trip_problems(table: pa.Table, grand: Grand, run: RunSpec) -> list[str]:
    """Items whose rows do not fold back into the record of the grand file, and vice versa."""
    needed = SCHEMAS["baselines" if run.kind == "baseline" else "submissions"].names
    absent = [name for name in needed if name not in table.column_names]
    if absent:
        return [f"cannot be compared with the grand file: no column {n!r}" for n in absent]

    # Row positions per item; only one item's rows are turned into Python objects at a time.
    positions: dict[str, list[int]] = {}
    for position, item_id in enumerate(table.column("item_id").to_pylist()):
        positions.setdefault(item_id, []).append(position)

    def rows_of(item_id: str) -> list[dict] | None:
        where = positions.get(item_id)
        if where is None:
            return None
        if where[-1] - where[0] == len(where) - 1:  # the build keeps an item's rows together
            return table.slice(where[0], len(where)).to_pylist()
        return table.take(where).to_pylist()

    problems = []
    covered = set()
    for item_id, raw in grand.iter_records(run.column):
        covered.add(item_id)
        record = normalise_raw_record(raw)
        if run.variant == "alt":  # an alt run keeps its Task 1 fields only
            record = {key: value for key, value in record.items() if key == T1}
        rows = rows_of(item_id)
        if rows is None:
            problems.append(f"item {item_id!r}: no rows")
            continue
        systems = sorted(set().union(*(predictions.keys() for predictions in record.values())))
        if sorted(row["system"] for row in rows) != systems:
            problems.append(
                f"item {item_id!r}: {len(rows)} rows for {len(systems)} systems in the grand file"
            )
        elif roundtrip_record(rows) != record:
            problems.append(f"item {item_id!r}: the predictions differ from the grand file")
    problems += [
        f"item {item_id!r}: rows for an item the run does not cover"
        for item_id in sorted(positions.keys() - covered)
    ]
    return problems


def _capped(name: str, problems: list[str], total: int | None = None) -> list[str]:
    """``problems`` as failures of ``name``, showing at most ``MAX_ITEM_FAILURES``.

    ``total`` is the number of problems when ``problems`` only holds the first few of them.
    """
    total = len(problems) if total is None else total
    shown = [f"{name}: {problem}" for problem in problems[:MAX_ITEM_FAILURES]]
    if total > MAX_ITEM_FAILURES:
        shown.append(f"{name}: ... and {total - MAX_ITEM_FAILURES} more")
    return shown


def check_run(out_dir: Path, grand: Grand, run: RunSpec) -> list[str]:
    """Round trip of one run: its file against its column of the grand file."""
    name = run_file(run)
    path = out_dir / name
    if not path.is_file():
        return [f"{name}: file is missing"]
    try:
        table = pq.read_table(path)
        problems = _label_problems(table, run)
        problems += _round_trip_problems(table, grand, run)
    except (OSError, ValueError, KeyError, pa.ArrowException) as error:
        return [f"{name}: cannot be checked: {type(error).__name__}: {error}"]
    return _capped(name, problems)


def check_round_trip(out_dir: Path, grand: Grand) -> list[str]:
    """Check every run's file against the grand file, one run at a time to bound memory.

    The rows of each file are grouped by item and folded back with ``roundtrip_record``; the
    result must equal ``normalise_raw_record`` of the grand cell (Task 1 only for an ``alt``
    run), with exactly the same systems.  Every item the run covers must have rows and no other
    item may.  Files that belong to no run are failures too.
    """
    runs = list_runs(grand)
    wanted = {run_file(run) for run in runs}
    failures = []
    for config in ("submissions", "baselines"):
        for path in _config_files(out_dir, config):
            name = _relative(out_dir, path)
            if name not in wanted:
                failures.append(f"{name}: file belongs to no run of the grand file")
    for run in runs:
        failures += check_run(out_dir, grand, run)
    return failures


# --- all gates -----------------------------------------------------------------------------------


def validate(
    out_dir: Path, grand: Grand, expected: dict, *, table2_path: Path = DEFAULT_TABLE2_PATH
) -> list[str]:
    """Run every gate on the build in ``out_dir`` and return the failure messages.

    An empty list means the build passes.  ``expected`` holds ``rows`` (rows per config),
    optionally ``pairs_by_collection``, and ``check_table2`` to switch the Table 2 gate on.  A
    gate that cannot read what it needs reports that as a failure instead of raising.  A build
    without its ``README.md`` or its ``build_manifest.json`` fails.
    """
    out_dir = Path(out_dir)
    gates = [(check_required_files, (out_dir,)), (check_row_counts, (out_dir, expected))]
    if expected.get("check_table2"):
        gates.append((check_table2, (out_dir, table2_path)))
    gates += [
        (find_pii, (out_dir,)),
        (check_schemas, (out_dir,)),
        (check_keys, (out_dir,)),
        (check_item_texts, (out_dir, grand)),
        (check_alt_task2, (grand,)),
        (check_round_trip, (out_dir, grand)),
    ]
    failures: list[str] = []
    for gate, args in gates:
        try:
            found = gate(*args)
        except UnreadableFile as error:
            found = [str(error)]
        except (OSError, ValueError, KeyError, pa.ArrowException) as error:
            found = [f"{gate.__name__}: could not run: {type(error).__name__}: {error}"]
        failures += [message for message in found if message not in failures]
    return failures
