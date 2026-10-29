"""Shared model-card parsing helpers for automatic task conversions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .io import DataError, load_json


SUBMISSION_NAME_QUESTION = "Submission name"
ORGANIZATION_NAME_QUESTION = "Organization name"
CLOSED_SOURCE_MODEL_QUESTION = "Closed-source model name (e.g. GPT-5.6-luna)"
SYSTEM_LINK_QUESTION = "Provide link to the submitted system if you have any"
BASE_MODEL_OVERRIDE_TRACKS = ("task2_participant", "task3_direct")
REFERENCE_QUESTION = (
    "Does your system use the provided reference translations in any form? "
    "(true|false)"
)


def load_answers(path: Path) -> dict[str, Any]:
    """Read a model-card list into unique question-to-answer entries."""
    card = load_json(path)
    if not isinstance(card, list):
        raise DataError(f"{path}: model card must be a JSON list")
    answers = {}
    for index, entry in enumerate(card, start=1):
        if not isinstance(entry, dict):
            raise DataError(f"{path}: model-card entry {index} must be an object")
        question = entry.get("question")
        if not isinstance(question, str) or "answer" not in entry:
            raise DataError(f"{path}: model-card entry {index} is malformed")
        if question in answers:
            raise DataError(f"{path}: duplicate model-card question {question!r}")
        answers[question] = entry["answer"]
    return answers


def required_text(answers: dict[str, Any], question: str, source: str) -> str:
    """Return a stripped required text answer or reject an empty/non-text value."""
    answer = answers.get(question)
    if not isinstance(answer, str) or not answer.strip():
        raise DataError(f"{source}: missing text answer for {question!r}")
    return answer.strip()


def parse_boolean(value: Any) -> bool | None:
    """Return a declared Boolean, accepting JSON Booleans and true/false strings."""
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def raw_answer(value: Any) -> str:
    """Serialize a card answer for diagnostics without changing string answers."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def declared_base_model(answers: dict[str, Any]) -> str:
    """Return explicit model metadata without inferring from free-form prose."""
    closed_source_model = answers.get(CLOSED_SOURCE_MODEL_QUESTION)
    if isinstance(closed_source_model, str) and closed_source_model.strip():
        return closed_source_model.strip()

    system_link = answers.get(SYSTEM_LINK_QUESTION)
    if not isinstance(system_link, str) or not system_link.strip():
        return ""
    parsed = urlparse(system_link.strip())
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    return "/".join(parts[:2]) if len(parts) >= 2 else ""


def load_base_model_overrides(path: Path) -> dict[str, dict[int, str]]:
    """Load explicit base-model corrections keyed by stable submission ID."""
    value = load_json(path)
    expected_fields = {"schema_version", *BASE_MODEL_OVERRIDE_TRACKS}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise DataError(f"{path}: invalid base-model override fields")
    if value["schema_version"] != 1:
        raise DataError(f"{path}: unsupported base-model override schema")

    overrides = {}
    for track in BASE_MODEL_OVERRIDE_TRACKS:
        entries = value[track]
        if not isinstance(entries, dict):
            raise DataError(f"{path}: {track} overrides must be an object")
        parsed = {}
        for raw_id, model in entries.items():
            if (
                not isinstance(raw_id, str)
                or not raw_id.isdigit()
                or str(int(raw_id)) != raw_id
                or int(raw_id) <= 0
            ):
                raise DataError(f"{path}: invalid {track} submission ID {raw_id!r}")
            if not isinstance(model, str) or not model.strip():
                raise DataError(f"{path}: invalid base model for {track} {raw_id}")
            parsed[int(raw_id)] = model.strip()
        if list(parsed) != sorted(parsed):
            raise DataError(f"{path}: {track} overrides must be sorted by ID")
        overrides[track] = parsed
    return overrides


def resolved_base_model(
    answers: dict[str, Any],
    submission_id: int,
    overrides: dict[int, str] | None = None,
) -> str:
    """Prefer reviewed ID-keyed metadata over incomplete model-card fields."""
    if overrides and submission_id in overrides:
        return overrides[submission_id]
    return declared_base_model(answers)
