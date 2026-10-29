"""Read the integrated benchmark while preserving each Task 1 source's records."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from wmt26_mteval_task1.models import SubmissionDescriptor
from wmt26_mteval_task1.submissions import baseline_family_for_root


_CURRENT_SOURCE: ContextVar[Task1ParquetSource | None] = ContextVar(
    "task1_parquet_source", default=None
)


def current_source() -> Task1ParquetSource | None:
    """Return the parquet source active for the current command or test."""

    return _CURRENT_SOURCE.get()


class Task1ParquetSource:
    """Expose source-ordered Task 1 JSON records from the grand parquet file."""

    TEST_SET = Path("test-set-files/mteval-test26.jsonl")
    GROUND_TRUTH = Path("human_annotations/ground-truth.jsonl")

    def __init__(self, path: Path, model_cards_dir: Path) -> None:
        """Index Task 1 manifest entries and the location of copied model cards."""

        self.path = path
        self.model_cards_dir = model_cards_dir
        if not path.is_file():
            raise ValueError(f"{path}: integrated parquet data file is missing")
        if not model_cards_dir.is_dir():
            raise ValueError(f"{model_cards_dir}: Task 1 model-card directory is missing")
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        if metadata.get(b"wmt26_schema_version") != b"wmt26-grand-parquet-v1":
            raise ValueError(f"{path}: unsupported or missing WMT26 parquet schema")
        if b"wmt26_source_manifest" not in metadata:
            raise ValueError(f"{path}: source manifest is missing")
        manifest = json.loads(metadata[b"wmt26_source_manifest"])
        self.entries = {
            Path(entry["path"]): entry
            for entry in manifest
            if entry["kind"] in {"test_set", "ground_truth"}
            or entry["task_namespace"].startswith("task1")
        }
        self._parquet_digest: str | None = None
        for required in (self.TEST_SET, self.GROUND_TRUTH):
            if required not in self.entries:
                raise ValueError(f"{path}: required source {required} is absent")

    @contextmanager
    def activate(self) -> Iterator[Task1ParquetSource]:
        """Route existing Task 1 readers through this source for one command."""

        token = _CURRENT_SOURCE.set(self)
        try:
            yield self
        finally:
            _CURRENT_SOURCE.reset(token)

    def install(self) -> Token[Task1ParquetSource | None]:
        """Install this source for a command with an existing try/finally block."""

        return _CURRENT_SOURCE.set(self)

    @staticmethod
    def reset(token: Token[Task1ParquetSource | None]) -> None:
        """Restore the source that was active before one command."""

        _CURRENT_SOURCE.reset(token)

    def contains(self, path: Path) -> bool:
        """Check whether a logical JSONL path is represented by this parquet."""

        return path in self.entries

    def fingerprint(self, path: Path) -> str:
        """Fingerprint an integrated source and the physical parquet bytes."""

        entry = self.entries[path]
        return hashlib.sha256(
            bytes.fromhex(self.parquet_sha256) + bytes.fromhex(entry["sha256"])
        ).hexdigest()

    @property
    def parquet_sha256(self) -> str:
        """Return the physical file hash used to invalidate integrated-data caches."""

        if self._parquet_digest is None:
            digest = hashlib.sha256()
            with self.path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._parquet_digest = digest.hexdigest()
        return self._parquet_digest

    def logical_digest(self, path: Path) -> str:
        """Return the audited source hash independently of parquet packaging."""

        return self.entries[path]["sha256"]

    def descriptors(self, root: Path, *, baseline: bool) -> tuple[SubmissionDescriptor, ...] | None:
        """Discover source identities under a logical submission or baseline root."""

        expected_kind = "baseline" if baseline else "submission"
        entries = [
            entry for logical, entry in self.entries.items()
            if logical.parent.parent == root and entry["kind"] == expected_kind
        ]
        if not entries:
            return None
        family = baseline_family_for_root(root) if baseline else ""
        descriptors = (
            SubmissionDescriptor(
                path=Path(entry["path"]).parent,
                submission_id="" if baseline else str(entry["run_id"]),
                username="baseline" if baseline else "",
                phase_id="",
                suffix="",
                is_baseline=baseline,
                baseline_family=family,
                model_card_override=(
                    self.model_cards_dir / entry["path"].removesuffix("predictions.jsonl")
                    / "model_card.json"
                ),
            )
            for entry in entries
        )
        if baseline:
            return tuple(sorted(descriptors, key=lambda x: (x.path.name.casefold(), x.path.name)))
        return tuple(sorted(descriptors, key=lambda x: int(x.submission_id)))

    def iter_records(self, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield exact source JSON objects in original JSONL line order."""

        entry = self.entries[path]
        key = entry["column"]
        canonical = entry["canonical_column"]
        is_test_set = entry["kind"] == "test_set"
        columns = ["item_id", "source_provenance"]
        if is_test_set:
            columns.extend(("src", "ref", "hyps", "resources"))
        else:
            columns.append(canonical)
        records: list[tuple[int, dict[str, Any]]] = []
        parquet = pq.ParquetFile(self.path)
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            data = batch.to_pydict()
            for index, item_id in enumerate(data["item_id"]):
                provenance = json.loads(data["source_provenance"][index]).get(key)
                if provenance is None:
                    continue
                lines = provenance["line_numbers"]
                if is_test_set:
                    values = {
                        name: (
                            json.loads(data[name][index])
                            if name in {"ref", "hyps", "resources"}
                            and data[name][index] is not None
                            else data[name][index]
                        )
                        for name in ("src", "ref", "hyps", "resources")
                    }
                    values = {"item_id": item_id, **values, **provenance["extra_fields"][0]}
                    payloads = [
                        {name: values[name] for name in provenance["record_fields"][0]}
                    ]
                else:
                    cell = data[canonical][index]
                    payloads = json.loads(cell) if cell is not None else []
                    payloads = [{"item_id": item_id, **payload} for payload in payloads]
                if len(lines) != len(payloads):
                    raise ValueError(f"{self.path}: provenance mismatch for {path}")
                records.extend(zip(lines, payloads))
        records.sort(key=lambda pair: pair[0])
        if "records" in entry and len(records) != entry["records"]:
            raise ValueError(f"{self.path}: incomplete records for {path}")
        if any(line != number for number, (line, _) in enumerate(records, 1)):
            raise ValueError(f"{self.path}: invalid source line order for {path}")
        for line, record in records:
            yield line, record
