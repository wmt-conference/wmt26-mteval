"""Convert opted-in Task 1 error annotations to Task 3 binary labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import DataError, iter_jsonl, load_json, parse_item_id, sha256_file
from .metrics import PairObservations
from .model_cards import (
    ORGANIZATION_NAME_QUESTION,
    REFERENCE_QUESTION,
    SUBMISSION_NAME_QUESTION,
    declared_base_model,
    load_answers,
    parse_boolean,
    raw_answer,
    required_text,
)
from .models import HumanReference, OfficialInventory


PERMISSION_QUESTION = "Allow automatic conversion from Task 1 to Task 3 (true|false)"


@dataclass(frozen=True)
class Task1Decision:
    """Store typed task1 decision state shared by the original analysis routines."""
    submission_id: int | str
    directory_name: str
    organization_name: str
    submission_name: str
    uses_reference: bool | None
    permission: bool | None
    permission_raw: str
    status: str
    reason: str
    predictions_file: str
    predictions_sha256: str
    model_card_file: str
    model_card_sha256: str
    archive_file: str = ""
    archive_sha256: str = ""
    base_model: str = ""


@dataclass(frozen=True)
class Task1ConversionResult:
    """Store typed task1 conversion result state shared by the original analysis routines."""
    decision: Task1Decision
    display_name: str
    attempted_pairs: tuple[str, ...]
    official_labels: int
    converted_error_free: int
    converted_error: int
    submitted_error_spans: int
    zero_length_spans: int
    omission_marker_spans: int
    instruction_error_marker_spans: int
    nonstandard_zero_length_spans: int
    top_level_omission_labels: int
    top_level_instruction_fault_labels: int
    converted_prediction_sha256: str
    metric_rows: tuple[dict[str, Any], ...]


def _find_submission_directory(task1_dir: Path, submission_id: int) -> Path:
    """Find one numeric-ID release directory or original ID-prefixed directory."""
    matches = [path for path in [task1_dir / str(submission_id),
                                *sorted(task1_dir.glob(f"{submission_id}_*"))]
               if path.is_dir()]
    if len(matches) != 1:
        raise DataError(
            f"Expected one Task 1 directory for {submission_id}, found "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def task1_input_digest(submission_ids: list[int], task1_dir: Path) -> str:
    """Hash original Task 1 relative filenames and payload bytes in sorted order."""
    digest = hashlib.sha256()
    paths = []
    for submission_id in submission_ids:
        directory = _find_submission_directory(task1_dir, submission_id)
        paths.extend([directory / "model_card.json", directory / "predictions.jsonl"])
    for path in sorted(paths, key=lambda candidate: str(candidate.relative_to(task1_dir))):
        if not path.is_file():
            raise DataError(f"Missing retained Task 1 input: {path}")
        relative_path = str(path.relative_to(task1_dir))
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _local_task1_submission_ids(task1_dir: Path) -> list[int]:
    """List unique numeric IDs from original ID-prefixed source directories."""
    ids = []
    for path in task1_dir.iterdir():
        prefix, separator, _ = path.name.partition("_")
        if path.is_dir() and separator and prefix.isdigit():
            ids.append(int(prefix))
    if len(ids) != len(set(ids)):
        raise DataError(f"{task1_dir}: duplicate Task 1 submission ID directories")
    return sorted(ids)


def load_task1_submission_ids(
    path: Path, task1_dir: Path | None = None
) -> list[int]:
    """Validate the original frozen Task 1 selection and optional source digest."""
    value = load_json(path)
    expected_fields = {"input_digest_algorithm", "input_sha256", "submission_ids"}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise DataError(f"{path}: invalid Task 1 selection fields")
    submission_ids = value["submission_ids"]
    if not isinstance(submission_ids, list) or not submission_ids:
        raise DataError(f"{path}: expected a non-empty list of submission IDs")
    if any(
        type(submission_id) is not int or submission_id <= 0
        for submission_id in submission_ids
    ):
        raise DataError(f"{path}: every submission ID must be a positive integer")
    if len(set(submission_ids)) != len(submission_ids):
        raise DataError(f"{path}: duplicate submission ID")
    ordered_ids = sorted(submission_ids)
    if task1_dir is not None:
        local_ids = _local_task1_submission_ids(task1_dir)
        if ordered_ids != local_ids:
            missing = sorted(set(local_ids) - set(ordered_ids))
            absent = sorted(set(ordered_ids) - set(local_ids))
            raise DataError(
                f"{path}: selection does not match local Task 1 directories; "
                f"unselected local IDs={missing}, configured IDs not found={absent}"
            )
        actual_digest = task1_input_digest(ordered_ids, task1_dir)
        if actual_digest != value["input_sha256"]:
            raise DataError(
                f"{path}: retained Task 1 inputs changed; expected aggregate "
                f"SHA-256 {value['input_sha256']}, found {actual_digest}"
            )
    return ordered_ids


def inspect_task1_decision(submission_id: int, task1_dir: Path) -> Task1Decision:
    """Read source-card permission and return a conversion decision with file hashes."""
    directory = _find_submission_directory(task1_dir, submission_id)
    predictions_path = directory / "predictions.jsonl"
    model_card_path = directory / "model_card.json"
    if not predictions_path.is_file() or not model_card_path.is_file():
        raise DataError(f"Missing Task 1 files in {directory}")
    answers = load_answers(model_card_path)
    organization_name = required_text(
        answers, ORGANIZATION_NAME_QUESTION, str(model_card_path)
    )
    submission_name = required_text(
        answers, SUBMISSION_NAME_QUESTION, str(model_card_path)
    )
    permission_value = answers.get(PERMISSION_QUESTION)
    permission = parse_boolean(permission_value)
    if permission is True:
        status = "converted"
        reason = "Task 1 to Task 3 conversion permission is true"
    elif permission is False:
        status = "opted_out"
        reason = "Task 1 to Task 3 conversion permission is false"
    else:
        status = "invalid_permission"
        reason = "Conversion permission is missing or not true/false"
    return Task1Decision(
        submission_id=submission_id,
        directory_name=directory.name,
        organization_name=organization_name,
        submission_name=submission_name,
        base_model=declared_base_model(answers),
        uses_reference=parse_boolean(answers.get(REFERENCE_QUESTION)),
        permission=permission,
        permission_raw=raw_answer(permission_value),
        status=status,
        reason=reason,
        predictions_file=predictions_path.name,
        predictions_sha256=sha256_file(predictions_path),
        model_card_file=model_card_path.name,
        model_card_sha256=sha256_file(model_card_path),
    )


def build_task1_display_names(
    decisions: list[Task1Decision],
) -> dict[int | str, str]:
    """Return public display names, disambiguating repeated organizations/systems."""
    organization_counts = Counter(decision.organization_name for decision in decisions)
    base_names = {
        decision.submission_id: (
            decision.organization_name
            if organization_counts[decision.organization_name] == 1
            else f"{decision.organization_name}: {decision.submission_name}"
        )
        for decision in decisions
    }
    base_counts = Counter(base_names.values())
    return {
        submission_id: (
            f"{base_name} [{submission_id}]"
            if base_counts[base_name] > 1
            else base_name
        )
        for submission_id, base_name in base_names.items()
    }


def _is_clean_task1_judgment(
    value: Any, target: str, source: str
) -> tuple[bool, int, int, int, int, bool, bool]:
    """Validate Task 1 spans/markers and return cleanliness plus diagnostic counts."""
    if not isinstance(value, dict):
        raise DataError(f"{source}: Task 1 judgment must be an object")
    errors = value.get("errors")
    if "omission" not in value:
        raise DataError(f"{source}: missing omission")
    omission = value["omission"]
    instruction_fault = value.get("instruction_fault")
    if not isinstance(errors, list):
        raise DataError(f"{source}: errors must be a list")
    if omission not in (None, "minor", "major"):
        raise DataError(f"{source}: invalid omission value {omission!r}")
    if instruction_fault not in (None, "minor", "major"):
        raise DataError(
            f"{source}: invalid instruction_fault value {instruction_fault!r}"
        )

    zero_length_spans = 0
    omission_marker_spans = 0
    instruction_error_marker_spans = 0
    nonstandard_zero_length_spans = 0
    target_length = len(target)
    for span_index, span in enumerate(errors):
        if not isinstance(span, dict):
            raise DataError(f"{source}: error {span_index} must be an object")
        start = span.get("start")
        end = span.get("end")
        severity = span.get("severity")
        if type(start) is not int or type(end) is not int:
            raise DataError(f"{source}: error {span_index} offsets must be integers")
        if start == end == target_length:
            omission_marker_spans += 1
        elif start == end == target_length + 1:
            instruction_error_marker_spans += 1
        elif start < 0 or end < start or end > target_length:
            raise DataError(
                f"{source}: invalid error offsets {start}:{end} for target "
                f"length {target_length}"
            )
        elif start == end:
            nonstandard_zero_length_spans += 1
        if severity not in {"minor", "major"}:
            raise DataError(f"{source}: invalid error severity {severity!r}")
        zero_length_spans += start == end
    clean = not errors and omission is None and instruction_fault is None
    return (
        clean,
        len(errors),
        omission_marker_spans,
        instruction_error_marker_spans,
        nonstandard_zero_length_spans,
        omission is not None,
        instruction_fault is not None,
    )


def convert_task1_and_evaluate(
    decision: Task1Decision,
    display_name: str,
    task1_dir: Path,
    inventory: OfficialInventory,
    human_reference: HumanReference,
) -> Task1ConversionResult:
    """Convert clean Task 1 assessments and score every retained human judgment."""
    if decision.status != "converted":
        raise ValueError(f"Submission {decision.submission_id} is not convertible")
    directory = task1_dir / decision.directory_name
    predictions_path = directory / decision.predictions_file
    submitted_ids: set[str] = set()
    submitted_by_pair: defaultdict[str, set[str]] = defaultdict(set)
    observations: defaultdict[str, PairObservations] = defaultdict(PairObservations)
    converted_digest = hashlib.sha256()
    official_labels = 0
    converted_error_free = 0
    submitted_error_spans = 0
    zero_length_spans = 0
    omission_marker_spans = 0
    instruction_error_marker_spans = 0
    nonstandard_zero_length_spans = 0
    top_level_omission_labels = 0
    top_level_instruction_fault_labels = 0

    for line_number, record in iter_jsonl(predictions_path):
        source = f"{predictions_path}:{line_number}"
        if "item_id" not in record:
            raise DataError(f"{source}: missing item_id")
        item_id = record["item_id"]
        parsed_id = parse_item_id(item_id, source)
        if parsed_id.set_id != "official":
            continue
        if "task1_pred" not in record:
            raise DataError(f"{source}: official record is missing task1_pred")
        predictions = record["task1_pred"]
        if not isinstance(predictions, dict) or not predictions:
            raise DataError(f"{source}: task1_pred must be a non-empty object")
        if item_id in submitted_ids:
            raise DataError(f"{source}: duplicate official item_id {item_id!r}")
        submitted_ids.add(item_id)

        test_item = inventory.items.get(item_id)
        if test_item is None:
            raise DataError(f"{source}: unknown official item_id {item_id!r}")
        if set(predictions) != set(test_item.hypotheses):
            raise DataError(f"{source}: official system coverage mismatch")
        pair = parsed_id.language_pair
        submitted_by_pair[pair].add(item_id)
        converted_predictions = {}
        for system, value in predictions.items():
            judgment_source = f"{source} system {system!r}"
            (
                clean,
                span_count,
                omission_marker_count,
                instruction_marker_count,
                nonstandard_zero_count,
                has_top_level_omission,
                has_top_level_instruction_fault,
            ) = _is_clean_task1_judgment(
                value, test_item.hypotheses[system], judgment_source
            )
            converted_predictions[system] = clean
            official_labels += 1
            converted_error_free += clean
            submitted_error_spans += span_count
            zero_length_spans += (
                omission_marker_count
                + instruction_marker_count
                + nonstandard_zero_count
            )
            omission_marker_spans += omission_marker_count
            instruction_error_marker_spans += instruction_marker_count
            nonstandard_zero_length_spans += nonstandard_zero_count
            top_level_omission_labels += has_top_level_omission
            top_level_instruction_fault_labels += has_top_level_instruction_fault
            for human in human_reference.judgments_for((item_id, system)):
                observations[pair].add(item_id, human.error_free, clean)

        converted_line = json.dumps(
            {"item_id": item_id, "task3_pred": converted_predictions},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        converted_digest.update(converted_line.encode("utf-8") + b"\n")

    if not submitted_by_pair:
        raise DataError(f"{predictions_path}: no official predictions")
    for pair, actual_ids in submitted_by_pair.items():
        expected_ids = inventory.item_ids_by_pair.get(pair)
        if expected_ids is None or actual_ids != expected_ids:
            expected_count = 0 if expected_ids is None else len(expected_ids)
            raise DataError(
                f"{predictions_path}: incomplete {pair} coverage; expected "
                f"{expected_count} items, found {len(actual_ids)}"
            )

    metric_rows = []
    for pair in sorted(submitted_by_pair):
        pair_observations = observations.get(pair)
        if pair_observations is None:
            raise DataError(f"{predictions_path}: no human labels for {pair}")
        metric_rows.append(
            {
                "submission_id": decision.submission_id,
                "display_name": display_name,
                "organization_name": decision.organization_name,
                "submission_name": decision.submission_name,
                "language_pair": pair,
                **pair_observations.metrics(),
            }
        )

    return Task1ConversionResult(
        decision=decision,
        display_name=display_name,
        attempted_pairs=tuple(sorted(submitted_by_pair)),
        official_labels=official_labels,
        converted_error_free=converted_error_free,
        converted_error=official_labels - converted_error_free,
        submitted_error_spans=submitted_error_spans,
        zero_length_spans=zero_length_spans,
        omission_marker_spans=omission_marker_spans,
        instruction_error_marker_spans=instruction_error_marker_spans,
        nonstandard_zero_length_spans=nonstandard_zero_length_spans,
        top_level_omission_labels=top_level_omission_labels,
        top_level_instruction_fault_labels=top_level_instruction_fault_labels,
        converted_prediction_sha256=converted_digest.hexdigest(),
        metric_rows=tuple(metric_rows),
    )
