"""Reader for the WMT26 grand Parquet file.

The grand file holds one row per test item.  ``item_id`` and ``src`` are plain strings; ``ref``,
``hyps`` and ``resources`` are ``arrow.json`` columns that pyarrow hands back as JSON strings.
Every other column belongs to one source run (a submission, a baseline or the ground truth) and
holds a JSON string encoding a one-element list of records, or null where the run does not cover
the item.  The file is large, so everything here streams in batches and reads only the columns it
needs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

GRAND_SHA256 = "a61675552cebb68fb8303cf08a23fbec31e5bb4e0bd26a1fdee1a9bfafc2fb7a"

MANIFEST_KEY = b"wmt26_source_manifest"
TEST_SET_KIND = "test_set"
ITEM_COLUMNS = ("item_id", "src", "ref", "hyps", "resources")

_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_BATCH_ROWS = 1024


class ChecksumError(Exception):
    """The grand file does not have the expected sha256."""


@dataclass(frozen=True)
class ManifestEntry:
    """One source file recorded in the ``wmt26_source_manifest`` schema metadata."""

    source_id: int
    column: str
    path: str
    kind: str
    task_namespace: str
    version: str
    run_id: str | None
    bytes: int
    sha256: str
    records: int
    canonical_source_id: int
    canonical_column: str


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loads(cell: str | None) -> Any:
    return None if cell is None else json.loads(cell)


class Grand:
    """Read-only view of the grand Parquet file, refusing any file with the wrong checksum."""

    def __init__(self, path: Path, expected_sha256: str | None = GRAND_SHA256):
        self.path = Path(path)
        if expected_sha256 is not None:
            actual = _sha256_of(self.path)
            if actual != expected_sha256:
                raise ChecksumError(
                    f"{self.path}: sha256 is {actual}, expected {expected_sha256}"
                )
        schema = pq.read_schema(self.path)
        self._column_names = frozenset(schema.names)
        raw = (schema.metadata or {}).get(MANIFEST_KEY)
        if raw is None:
            raise ValueError(f"{self.path}: schema metadata has no {MANIFEST_KEY.decode()}")
        names = [f.name for f in fields(ManifestEntry)]
        self._manifest = [
            ManifestEntry(**{name: entry[name] for name in names}) for entry in json.loads(raw)
        ]

    @property
    def manifest(self) -> list[ManifestEntry]:
        """All source entries, in ``source_id`` order, including the test set."""
        return list(self._manifest)

    def canonical_entries(self) -> list[ManifestEntry]:
        """Entries that are their own canonical column, without the test set (it is not a run)."""
        return [
            e
            for e in self._manifest
            if e.canonical_source_id == e.source_id and e.kind != TEST_SET_KIND
        ]

    def items(self) -> Iterator[dict]:
        """Yield every item as ``item_id, src, ref, hyps, resources`` with the JSON decoded.

        ``ref`` and ``resources`` are dicts, or ``None`` when the cell is null.
        """
        with pq.ParquetFile(self.path) as parquet:
            for batch in parquet.iter_batches(batch_size=_BATCH_ROWS, columns=list(ITEM_COLUMNS)):
                cells = {name: batch.column(name).to_pylist() for name in ITEM_COLUMNS}
                for row in range(batch.num_rows):
                    yield {
                        "item_id": cells["item_id"][row],
                        "src": cells["src"][row],
                        "ref": _loads(cells["ref"][row]),
                        "hyps": _loads(cells["hyps"][row]),
                        "resources": _loads(cells["resources"][row]),
                    }

    def iter_records(self, column: str) -> Iterator[tuple[str, dict]]:
        """Yield ``(item_id, record)`` for each item the run in ``column`` covers.

        The cell's one-element list is unwrapped.  A null cell is skipped; a list of any other
        length raises ``ValueError``.  An unknown column raises ``KeyError`` straight away.
        """
        if column not in self._column_names:
            raise KeyError(f"{self.path}: no column {column!r}")
        return self._iter_records(column)

    def _iter_records(self, column: str) -> Iterator[tuple[str, dict]]:
        with pq.ParquetFile(self.path) as parquet:
            for batch in parquet.iter_batches(batch_size=_BATCH_ROWS, columns=["item_id", column]):
                for item_id, cell in zip(
                    batch.column("item_id").to_pylist(), batch.column(column).to_pylist()
                ):
                    if cell is None:
                        continue
                    records = json.loads(cell)
                    if not isinstance(records, list) or len(records) != 1:
                        got = f"{len(records)} records" if isinstance(records, list) else "no list"
                        raise ValueError(
                            f"{column}: item {item_id!r}: expected one record, got {got}"
                        )
                    yield item_id, records[0]
