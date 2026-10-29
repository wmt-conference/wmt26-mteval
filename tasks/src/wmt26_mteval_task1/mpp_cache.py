"""Persistent per-assessment MPP components shared by scoring workflows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from wmt26_mteval_task1.constants import MATCHING_ALGORITHMS, mpp_metric_options
from wmt26_mteval_task1.audit_io import file_digest
from wmt26_mteval_task1.models import SubmissionDescriptor

if TYPE_CHECKING:
    from wmt26_mteval_task1.scoring import PreparedAssessment, ScoredAssessment


CACHE_SCHEMA_VERSION = "4"


@dataclass
class MppCacheStatistics:
    """Small run summary suitable for one concise INFO message."""

    reused_entries: int = 0
    reused_segments: int = 0
    computed_entries: int = 0
    computed_segments: int = 0
    invalidated_entries: int = 0


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 fingerprint for one authoritative input."""

    try:
        return file_digest(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{path}: cache fingerprint input is missing") from exc


def _entry_key(descriptor: SubmissionDescriptor) -> tuple[str, str]:
    """Return the stable cache namespace and ID for an entry."""

    if descriptor.is_baseline:
        return "baseline", descriptor.cache_name
    return "submission", descriptor.submission_id


class MppCache:
    """SQLite-backed cache of raw MPP components for aligned assessments."""

    def __init__(
        self,
        path: Path,
        *,
        test_set_path: Path,
        ground_truth_path: Path,
        matching_algorithm: str = "greedy",
    ) -> None:
        """Open one algorithm-specific cache and validate its global inputs."""

        if matching_algorithm not in MATCHING_ALGORITHMS:
            raise ValueError(
                f"unsupported MPP matching algorithm {matching_algorithm!r}"
            )
        self.path = path
        self.matching_algorithm = matching_algorithm
        self.statistics = MppCacheStatistics()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(path)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._initialize(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "test_set_sha256": _file_sha256(test_set_path),
                    "references_sha256": _file_sha256(ground_truth_path),
                    "metric_options": json.dumps(
                        mpp_metric_options(matching_algorithm),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        except (OSError, sqlite3.DatabaseError) as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise ValueError(f"{path}: cannot initialize MPP cache: {exc}") from exc

    def __enter__(self) -> "MppCache":
        """Return this open cache for use as a context manager."""

        return self

    def __exit__(self, *_: object) -> None:
        """Close the SQLite connection when leaving a context."""

        self.close()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._connection.close()

    def _initialize(self, expected_metadata: dict[str, str]) -> None:
        """Create the schema and clear entries incompatible with current inputs."""

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        existing = dict(
            self._connection.execute("SELECT key, value FROM cache_metadata")
        )
        if existing.get("schema_version") not in {None, CACHE_SCHEMA_VERSION}:
            self._connection.execute("DROP TABLE IF EXISTS segment_mpp")
            self._connection.execute("DROP TABLE IF EXISTS entry_cache")
            self._connection.execute("DELETE FROM cache_metadata")
            existing = {}

        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS entry_cache (
                entry_type TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                matching_algorithm TEXT NOT NULL,
                predictions_sha256 TEXT NOT NULL,
                PRIMARY KEY (entry_type, entry_id, matching_algorithm)
            );

            CREATE TABLE IF NOT EXISTS segment_mpp (
                entry_type TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                matching_algorithm TEXT NOT NULL,
                item_id TEXT NOT NULL,
                system TEXT NOT NULL,
                language_pair TEXT NOT NULL,
                reference_index INTEGER NOT NULL,
                source_length INTEGER,
                target_length INTEGER NOT NULL,
                segment_weight REAL NOT NULL,
                precision_numerator REAL NOT NULL,
                precision_denominator REAL NOT NULL,
                recall_numerator REAL NOT NULL,
                recall_denominator REAL NOT NULL,
                PRIMARY KEY (
                    entry_type, entry_id, matching_algorithm,
                    item_id, system, reference_index
                ),
                FOREIGN KEY (entry_type, entry_id, matching_algorithm)
                    REFERENCES entry_cache (entry_type, entry_id, matching_algorithm)
                    ON DELETE CASCADE
            );
            """
        )

        if existing and any(
            existing.get(key) != value for key, value in expected_metadata.items()
        ):
            self.statistics.invalidated_entries = int(
                self._connection.execute("SELECT COUNT(*) FROM entry_cache").fetchone()[
                    0
                ]
            )
            self._connection.execute("DELETE FROM entry_cache")
            self._connection.execute("DELETE FROM cache_metadata")

        self._connection.executemany(
            "INSERT OR REPLACE INTO cache_metadata (key, value) VALUES (?, ?)",
            expected_metadata.items(),
        )
        self._connection.commit()

    def load(
        self,
        descriptor: SubmissionDescriptor,
        expected_assessments: Iterable["PreparedAssessment"],
        *,
        matching_algorithm: str = "greedy",
    ) -> tuple[tuple[float, float, float, float, float] | None, ...] | None:
        """Return available cached components aligned to expected assessments."""

        self._validate_matching_algorithm(matching_algorithm)
        assessments = tuple(expected_assessments)
        entry_type, entry_id = _entry_key(descriptor)
        predictions_hash = _file_sha256(descriptor.predictions_path)
        try:
            entry = self._connection.execute(
                """
                SELECT predictions_sha256 FROM entry_cache
                WHERE entry_type = ? AND entry_id = ? AND matching_algorithm = ?
                """,
                (entry_type, entry_id, matching_algorithm),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"{self.path}: cannot read MPP cache: {exc}") from exc
        if entry is None or entry["predictions_sha256"] != predictions_hash:
            return None

        try:
            rows = self._connection.execute(
                """
                SELECT * FROM segment_mpp
                WHERE entry_type = ? AND entry_id = ? AND matching_algorithm = ?
                """,
                (entry_type, entry_id, matching_algorithm),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"{self.path}: cannot read MPP cache: {exc}") from exc
        by_key = {
            (row["item_id"], row["system"], row["reference_index"]): row for row in rows
        }
        values: list[tuple[float, float, float, float, float] | None] = []
        for assessment in assessments:
            key = (
                getattr(assessment, "item_id"),
                getattr(assessment, "system"),
                getattr(assessment, "reference_index"),
            )
            row = by_key.get(key)
            if row is None:
                values.append(None)
                continue
            if (
                row["language_pair"] != getattr(assessment, "language_pair")
                or row["source_length"] != getattr(assessment, "source_length")
                or row["target_length"] != getattr(assessment, "target_length")
            ):
                values.append(None)
                continue
            values.append(
                (
                    float(row["segment_weight"]),
                    float(row["precision_numerator"]),
                    float(row["precision_denominator"]),
                    float(row["recall_numerator"]),
                    float(row["recall_denominator"]),
                )
            )
        reused = sum(value is not None for value in values)
        if reused:
            self.statistics.reused_entries += 1
            self.statistics.reused_segments += reused
        return tuple(values)

    def store(
        self,
        descriptor: SubmissionDescriptor,
        scored_assessments: Iterable["ScoredAssessment"],
        *,
        matching_algorithm: str = "greedy",
    ) -> None:
        """Merge scored assessments, clearing rows only for changed predictions."""

        self._validate_matching_algorithm(matching_algorithm)
        values = tuple(scored_assessments)
        if not values:
            return
        entry_type, entry_id = _entry_key(descriptor)
        predictions_hash = _file_sha256(descriptor.predictions_path)
        try:
            with self._connection:
                existing = self._connection.execute(
                    "SELECT predictions_sha256 FROM entry_cache WHERE entry_type = ? "
                    "AND entry_id = ? AND matching_algorithm = ?",
                    (entry_type, entry_id, matching_algorithm),
                ).fetchone()
                if (
                    existing is not None
                    and existing["predictions_sha256"] != predictions_hash
                ):
                    self._connection.execute(
                        "DELETE FROM entry_cache WHERE entry_type = ? AND entry_id = ? "
                        "AND matching_algorithm = ?",
                        (entry_type, entry_id, matching_algorithm),
                    )
                    existing = None
                if existing is None:
                    self._connection.execute(
                        """
                        INSERT INTO entry_cache (
                            entry_type, entry_id, matching_algorithm, predictions_sha256
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (entry_type, entry_id, matching_algorithm, predictions_hash),
                    )
                self._connection.executemany(
                    """
                    INSERT OR REPLACE INTO segment_mpp (
                        entry_type, entry_id, matching_algorithm, item_id, system,
                        language_pair,
                        reference_index, source_length, target_length,
                        segment_weight, precision_numerator,
                        precision_denominator, recall_numerator,
                        recall_denominator
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            entry_type,
                            entry_id,
                            matching_algorithm,
                            value.assessment.item_id,
                            value.assessment.system,
                            value.assessment.language_pair,
                            value.assessment.reference_index,
                            value.assessment.source_length,
                            value.assessment.target_length,
                            value.segment_weight,
                            value.components.precision_numerator,
                            value.components.precision_denominator,
                            value.components.recall_numerator,
                            value.components.recall_denominator,
                        )
                        for value in values
                    ),
                )
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"{self.path}: cannot update MPP cache: {exc}") from exc
        self.statistics.computed_entries += 1
        self.statistics.computed_segments += len(values)

    def _validate_matching_algorithm(self, matching_algorithm: str) -> None:
        """Reject attempts to mix algorithms in one physical cache file."""

        if matching_algorithm != self.matching_algorithm:
            raise ValueError(
                f"MPP cache {self.path} is bound to {self.matching_algorithm!r}, "
                f"not {matching_algorithm!r}"
            )

    def summary(self) -> str:
        """Format cache hit, computation, and invalidation counts for logging."""

        stats = self.statistics
        invalidated = (
            f"; invalidated {stats.invalidated_entries} stale "
            f"{'entry' if stats.invalidated_entries == 1 else 'entries'}"
            if stats.invalidated_entries
            else ""
        )
        return (
            f"MPP cache {self.path}: reused {stats.reused_entries} entries / "
            f"{stats.reused_segments} segments; computed {stats.computed_entries} "
            f"entries / {stats.computed_segments} segments{invalidated}"
        )
