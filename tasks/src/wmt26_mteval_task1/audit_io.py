"""Small deterministic serialization helpers for Task 1 audit artifacts."""

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file without loading it into memory."""

    from wmt26_mteval_task1.parquet_source import current_source

    source = current_source()
    if source is not None and source.contains(path):
        return source.fingerprint(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_digest(value: Any) -> str:
    """Return a stable hash of a JSON-compatible value, ignoring object-key order."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Serialize a report value as indented UTF-8 JSON with a trailing newline."""

    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(
    path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    """Stream report dictionaries into a CSV with the supplied stable columns."""

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_errors(
    errors: list[dict[str, Any]],
    *,
    identity_fields: tuple[str, ...],
    preferred_severity: str | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate errors by selected fields, optionally preferring one severity.

    The returned groups retain the order of their first occurrences. Within a
    duplicate group, the first error with ``preferred_severity`` replaces an
    earlier non-preferred error; otherwise the first error is retained.
    """

    retained_positions: dict[str, int] = {}
    retained: list[dict[str, Any]] = []
    for error in errors:
        key = [error.get(field) for field in identity_fields]
        signature = json.dumps(key, sort_keys=True, ensure_ascii=False)
        if signature not in retained_positions:
            retained_positions[signature] = len(retained)
            retained.append(error)
            continue
        position = retained_positions[signature]
        if (
            preferred_severity is not None
            and retained[position].get("severity") != preferred_severity
            and error.get("severity") == preferred_severity
        ):
            retained[position] = error
    return retained


def duplicate_error_examples(
    errors: list[dict[str, Any]], *, identity_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Return the first raw-error pair sharing the selected identity fields."""

    seen: dict[str, dict[str, Any]] = {}
    for error in errors:
        key = [error.get(field) for field in identity_fields]
        signature = json.dumps(key, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            return [seen[signature], error]
        seen[signature] = error
    return []
