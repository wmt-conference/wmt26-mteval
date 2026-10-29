"""Strict input helpers for JSONL files and submission archives."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .models import ParsedItemId, SubmissionSpec


class DataError(ValueError):
    """An analysis input is missing or malformed."""


class DuplicateKeyError(ValueError):
    """A JSON object contains a duplicate key."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    """Reject nonstandard JSON numeric constants such as NaN and Infinity."""
    raise ValueError(f"non-standard JSON constant {value!r}")


def parse_json(text: str, source: str) -> Any:
    """Parse strict JSON and attach the source name to any validation error."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, DuplicateKeyError, ValueError) as error:
        raise DataError(f"{source}: invalid JSON: {error}") from error


def load_json(path: Path) -> Any:
    """Read and strictly parse one UTF-8 JSON file."""
    try:
        return parse_json(path.read_text(encoding="utf-8"), str(path))
    except UnicodeDecodeError as error:
        raise DataError(f"{path}: not valid UTF-8: {error}") from error


def iter_jsonl_text(text: str, source: str) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield numbered object records from nonempty strict JSONL text."""
    found = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        found = True
        if not line.strip():
            raise DataError(f"{source}:{line_number}: blank line")
        value = parse_json(line, f"{source}:{line_number}")
        if not isinstance(value, dict):
            raise DataError(f"{source}:{line_number}: expected a JSON object")
        yield line_number, value
    if not found:
        raise DataError(f"{source}: empty JSONL file")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield numbered object records from a nonempty UTF-8 JSONL file."""
    found = False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                found = True
                if not line.strip():
                    raise DataError(f"{path}:{line_number}: blank line")
                value = parse_json(line, f"{path}:{line_number}")
                if not isinstance(value, dict):
                    raise DataError(f"{path}:{line_number}: expected a JSON object")
                yield line_number, value
    except UnicodeDecodeError as error:
        raise DataError(f"{path}: not valid UTF-8: {error}") from error
    if not found:
        raise DataError(f"{path}: empty JSONL file")


def parse_item_id(item_id: Any, source: str) -> ParsedItemId:
    """Validate and split a six-component item identifier."""
    if not isinstance(item_id, str):
        raise DataError(f"{source}: item_id must be a string")
    parts = item_id.split("_###_")
    if len(parts) != 6 or any(not part for part in parts):
        raise DataError(
            f"{source}: invalid item_id {item_id!r}; expected "
            "set_###_source_###_target_###_domain_###_document_###_segment"
        )
    return ParsedItemId(*parts)


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of supplied bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 using bounded-memory streaming."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_file_by_sha256(
    directory: Path, pattern: str, expected_sha256: str, description: str
) -> Path:
    """Find exactly one matching payload inside the explicitly supplied directory."""
    matches = [
        path
        for path in sorted(directory.glob(pattern))
        if path.is_file() and sha256_file(path) == expected_sha256
    ]
    if len(matches) != 1:
        raise DataError(
            f"Expected exactly one {description} with SHA-256 "
            f"{expected_sha256}, found {len(matches)}"
        )
    return matches[0]


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 of a canonical compact sorted JSON representation."""
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(data)


def load_submission_specs(path: Path) -> list[SubmissionSpec]:
    """Read and validate a frozen original-archive selection catalog."""
    value = load_json(path)
    if not isinstance(value, list) or not value:
        raise DataError(f"{path}: expected a non-empty JSON list")
    specs = []
    seen_ids: set[int] = set()
    for index, row in enumerate(value):
        expected_fields = {
            "submission_id",
            "archive_sha256",
            "prediction_sha256",
        }
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise DataError(f"{path}: entry {index} has invalid fields")
        submission_id = row["submission_id"]
        if type(submission_id) is not int or submission_id <= 0:
            raise DataError(f"{path}: entry {index} has invalid submission_id")
        if submission_id in seen_ids:
            raise DataError(f"{path}: duplicate submission_id {submission_id}")
        archive_sha256 = row["archive_sha256"]
        prediction_sha256 = row["prediction_sha256"]
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (archive_sha256, prediction_sha256)
        ):
            raise DataError(f"{path}: entry {index} has invalid SHA-256")
        seen_ids.add(submission_id)
        specs.append(
            SubmissionSpec(
                submission_id,
                archive_sha256,
                prediction_sha256,
            )
        )
    return sorted(specs, key=lambda spec: spec.submission_id)


def find_submission_archive(directory: Path, spec: SubmissionSpec) -> Path:
    """Resolve exactly one original submission ZIP by stable numeric ID."""
    matches = sorted(directory.glob(f"{spec.submission_id}_*.zip"))
    if len(matches) != 1:
        raise DataError(
            f"Expected exactly one archive for {spec.key}, found {len(matches)}: "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def read_archive_member(archive_path: Path, filename: str) -> bytes:
    """Read one unique basename member after validating ZIP integrity."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise DataError(
                    f"{archive_path.name}: CRC failure in {bad_member!r}"
                )
            candidates = []
            for member in archive.infolist():
                member_path = PurePosixPath(member.filename)
                if (
                    member.is_dir()
                    or "__MACOSX" in member_path.parts
                    or member_path.name.startswith("._")
                ):
                    continue
                if member_path.name == filename:
                    candidates.append(member)
            if len(candidates) != 1:
                raise DataError(
                    f"{archive_path.name}: expected exactly one {filename}, "
                    f"found {[member.filename for member in candidates]}"
                )
            return archive.read(candidates[0])
    except zipfile.BadZipFile as error:
        raise DataError(f"{archive_path}: invalid ZIP archive") from error
