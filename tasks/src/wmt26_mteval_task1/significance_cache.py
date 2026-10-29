"""Transactional, resumable storage of Task 1 permutation difference draws."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

CHUNK_SIZE = 256
CACHE_VERSION = 1


def fingerprint(value: object) -> str:
    """Return a stable SHA-256 identity for JSON-compatible configuration."""
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


class SignificanceCache:
    """Store float64 draw chunks by immutable input and method fingerprints."""

    def __init__(self, path: Path, *, data_fingerprint: str | None = None) -> None:
        """Open a cache and invalidate saved draws if the parquet file changed."""
        self.path = path
        self.reused_chunks = 0
        self.computed_chunks = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        # WAL retains atomic chunk transactions without forcing a disk sync for
        # each small cache write. After power loss, recent chunks may need to be
        # regenerated; their deterministic seeds make this safe.
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS cache_environment (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS permutation_series (
                fingerprint TEXT PRIMARY KEY, metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS permutation_chunks (
                fingerprint TEXT NOT NULL REFERENCES permutation_series(fingerprint),
                chunk_index INTEGER NOT NULL, draws BLOB NOT NULL,
                PRIMARY KEY (fingerprint, chunk_index)
            );
        """)
        if data_fingerprint is not None:
            row = self.connection.execute(
                "SELECT value FROM cache_environment WHERE key = 'data_fingerprint'"
            ).fetchone()
            if row is None or row[0] != data_fingerprint:
                with self.connection:
                    self.connection.execute("DELETE FROM permutation_chunks")
                    self.connection.execute("DELETE FROM permutation_series")
                    self.connection.execute(
                        "INSERT OR REPLACE INTO cache_environment VALUES (?, ?)",
                        ("data_fingerprint", data_fingerprint),
                    )

    def __enter__(self) -> SignificanceCache:
        """Return the open cache for a context-managed computation."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the database after successful or interrupted computation."""
        self.connection.close()

    def register(self, metadata: dict, *, recompute: bool = False) -> str:
        """Identify a draw stream and optionally discard all its saved chunks."""
        metadata = {**metadata, "cache_version": CACHE_VERSION, "chunk_size": CHUNK_SIZE}
        key = fingerprint(metadata)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO permutation_series VALUES (?, ?)",
                (key, json.dumps(metadata, sort_keys=True)),
            )
            if recompute:
                self.connection.execute(
                    "DELETE FROM permutation_chunks WHERE fingerprint = ?", (key,),
                )
        return key

    def load(self, key: str, chunk_index: int) -> np.ndarray | None:
        """Return a complete saved chunk, or None when it has not been computed."""
        row = self.connection.execute(
            "SELECT draws FROM permutation_chunks WHERE fingerprint = ? AND chunk_index = ?",
            (key, chunk_index),
        ).fetchone()
        if row is None:
            return None
        values = np.frombuffer(row[0], dtype="<f8")
        if values.shape != (CHUNK_SIZE,) or not np.isfinite(values).all():
            raise ValueError(f"{self.path}: invalid permutation chunk {key}/{chunk_index}")
        self.reused_chunks += 1
        return values

    def store(self, key: str, chunk_index: int, draws: np.ndarray) -> None:
        """Atomically save a complete finite chunk for later prefix reuse."""
        values = np.asarray(draws, dtype="<f8")
        if values.shape != (CHUNK_SIZE,) or not np.isfinite(values).all():
            raise ValueError("permutation cache requires a complete finite chunk")
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO permutation_chunks VALUES (?, ?, ?)",
                (key, chunk_index, values.tobytes()),
            )
        self.computed_chunks += 1

    def summary(self) -> str:
        """Return concise chunk hit and computation counts for run logging."""
        return (
            f"Significance cache {self.path}: reused {self.reused_chunks} chunks; "
            f"computed {self.computed_chunks} chunks"
        )
