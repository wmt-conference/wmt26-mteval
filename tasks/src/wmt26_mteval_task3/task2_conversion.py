"""Convert retained Task 2 scores to Task 3 labels using model-card rules."""

from __future__ import annotations

import hashlib
import json
import math
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
    load_answers,
    parse_boolean,
    raw_answer,
    resolved_base_model,
    required_text,
)
from .models import HumanReference, OfficialInventory


PERMISSION_QUESTION = "Allow automatic conversion from Task 2 to Task 3 (true|false)"
THRESHOLD_QUESTION = (
    "Task 2 to 3 conversion threshold (everything above this score will be "
    "regarded as error-free)"
)
INCLUSIVITY_QUESTION = "Should the above threshold be inclusive (≥) or exclusive (>)?"


@dataclass(frozen=True)
class ConversionDecision:
    """Store typed conversion decision state shared by the original analysis routines."""
    submission_id: int | str
    organization_name: str
    submission_name: str
    uses_reference: bool | None
    permission: bool | None
    permission_raw: str
    threshold: float | None
    threshold_raw: str
    inclusive: bool | None
    inclusivity_raw: str
    comparison: str
    status: str
    reason: str
    segments_file: str
    segments_sha256: str
    model_card_file: str
    model_card_sha256: str
    directory_name: str = ""
    archive_file: str = ""
    archive_sha256: str = ""
    base_model: str = ""


@dataclass(frozen=True)
class ConversionResult:
    """Store typed conversion result state shared by the original analysis routines."""
    decision: ConversionDecision
    display_name: str
    attempted_pairs: tuple[str, ...]
    source_score_min: float
    source_score_max: float
    official_labels: int
    converted_error_free: int
    converted_error: int
    converted_prediction_sha256: str
    metric_rows: tuple[dict[str, Any], ...]


def task2_input_digest(submission_ids: list[int], task2_dir: Path) -> str:
    """Hash original Task 2 filenames and payload bytes in sorted order."""
    digest = hashlib.sha256()
    paths = []
    for submission_id in submission_ids:
        paths.extend(
            [
                task2_dir / f"{submission_id}_model_card.json",
                task2_dir / f"{submission_id}_segments.jsonl",
            ]
        )
    for path in sorted(paths, key=lambda candidate: candidate.name):
        if not path.is_file():
            raise DataError(f"Missing retained Task 2 input: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def load_task2_submission_ids(
    path: Path, task2_dir: Path | None = None
) -> list[int]:
    """Validate the original frozen Task 2 selection and optional source digest."""
    value = load_json(path)
    expected_fields = {"input_digest_algorithm", "input_sha256", "submission_ids"}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise DataError(f"{path}: invalid Task 2 selection fields")
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
    if task2_dir is not None:
        actual_digest = task2_input_digest(ordered_ids, task2_dir)
        if actual_digest != value["input_sha256"]:
            raise DataError(
                f"{path}: retained Task 2 inputs changed; expected aggregate "
                f"SHA-256 {value['input_sha256']}, found {actual_digest}"
            )
    return ordered_ids


def _parse_threshold(value: Any) -> float | None:
    """Return a finite numeric threshold, or None for an invalid/missing answer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        threshold = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            threshold = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return threshold if math.isfinite(threshold) else None


def _parse_inclusive(value: Any) -> bool | None:
    """Return the declared inclusive/exclusive rule, or None when unrecognized."""
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().lower().split())
    if normalized in {"inclusive", "inclusive (≥)", "≥", ">="}:
        return True
    if normalized in {"exclusive", "exclusive (>)", ">"}:
        return False
    return None


def inspect_conversion_decision(
    submission_id: int,
    task2_dir: Path,
    base_model_overrides: dict[int, str] | None = None,
) -> ConversionDecision:
    """Parse permission, finite threshold and comparison into a conversion decision."""
    directory = task2_dir / str(submission_id)
    segments_path = directory / "predictions.jsonl"
    model_card_path = directory / "model_card.json"
    if not directory.exists():
        # Retain the original explicit-directory API for upstream callers/tests.
        directory = task2_dir
        segments_path = directory / f"{submission_id}_segments.jsonl"
        model_card_path = directory / f"{submission_id}_model_card.json"
    if not segments_path.is_file() or not model_card_path.is_file():
        raise DataError(f"Missing retained Task 2 files for submission {submission_id}")
    answers = load_answers(model_card_path)
    organization_name = required_text(
        answers, ORGANIZATION_NAME_QUESTION, str(model_card_path)
    )
    submission_name = required_text(
        answers, SUBMISSION_NAME_QUESTION, str(model_card_path)
    )
    permission_value = answers.get(PERMISSION_QUESTION)
    threshold_value = answers.get(THRESHOLD_QUESTION)
    inclusivity_value = answers.get(INCLUSIVITY_QUESTION)
    permission = parse_boolean(permission_value)
    threshold = _parse_threshold(threshold_value)
    inclusive = _parse_inclusive(inclusivity_value)

    if permission is False:
        status = "opted_out"
        reason = "Task 2 to Task 3 conversion permission is false"
    elif permission is None:
        status = "invalid_permission"
        reason = "Conversion permission is missing or not true/false"
    elif threshold is None:
        status = "missing_threshold"
        reason = "Conversion threshold is missing or non-numeric"
    elif inclusive is None:
        status = "missing_inclusivity"
        reason = "Inclusive/exclusive conversion rule is missing or unrecognized"
    else:
        status = "converted"
        reason = "Complete opt-in conversion rule"

    reference_value = answers.get(REFERENCE_QUESTION)
    return ConversionDecision(
        submission_id=submission_id,
        organization_name=organization_name,
        submission_name=submission_name,
        base_model=resolved_base_model(answers, submission_id, base_model_overrides),
        uses_reference=parse_boolean(reference_value),
        permission=permission,
        permission_raw=raw_answer(permission_value),
        threshold=threshold,
        threshold_raw=raw_answer(threshold_value),
        inclusive=inclusive,
        inclusivity_raw=raw_answer(inclusivity_value),
        comparison=(">=" if inclusive is True else ">" if inclusive is False else ""),
        status=status,
        reason=reason,
        directory_name="" if directory == task2_dir else directory.name,
        segments_file=segments_path.name,
        segments_sha256=sha256_file(segments_path),
        model_card_file=model_card_path.name,
        model_card_sha256=sha256_file(model_card_path),
    )


def build_conversion_display_names(
    decisions: list[ConversionDecision],
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


def _numeric_scores(value: Any, source: str) -> dict[str, float]:
    """Validate finite non-Boolean numeric scores and return a float mapping."""
    if not isinstance(value, dict) or not value:
        raise DataError(f"{source}: task2_pred must be a non-empty object")
    scores = {}
    for system, score in value.items():
        if not isinstance(system, str) or not system:
            raise DataError(f"{source}: invalid system name")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise DataError(f"{source}: score for {system!r} must be numeric")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise DataError(f"{source}: score for {system!r} must be finite")
        scores[system] = numeric_score
    return scores


def convert_and_evaluate(
    decision: ConversionDecision,
    display_name: str,
    task2_dir: Path,
    inventory: OfficialInventory,
    human_reference: HumanReference,
) -> ConversionResult:
    """Apply the declared Task 2 comparison and score every retained human judgment."""
    if decision.status != "converted" or decision.threshold is None:
        raise ValueError(f"Submission {decision.submission_id} is not convertible")
    if decision.inclusive is None:
        raise ValueError(f"Submission {decision.submission_id} has no comparison rule")

    segments_path = task2_dir / decision.directory_name / decision.segments_file
    submitted_ids: set[str] = set()
    submitted_by_pair: defaultdict[str, set[str]] = defaultdict(set)
    observations: defaultdict[str, PairObservations] = defaultdict(PairObservations)
    converted_digest = hashlib.sha256()
    score_min = math.inf
    score_max = -math.inf
    official_labels = 0
    converted_error_free = 0

    for line_number, record in iter_jsonl(segments_path):
        source = f"{segments_path}:{line_number}"
        if "item_id" not in record:
            raise DataError(f"{source}: missing item_id")
        item_id = record["item_id"]
        parsed_id = parse_item_id(item_id, source)
        if parsed_id.set_id != "official":
            continue
        if "task2_pred" not in record:
            raise DataError(f"{source}: official record is missing task2_pred")
        scores = _numeric_scores(record["task2_pred"], source)
        if item_id in submitted_ids:
            raise DataError(f"{source}: duplicate official item_id {item_id!r}")
        submitted_ids.add(item_id)

        test_item = inventory.items.get(item_id)
        if test_item is None:
            raise DataError(f"{source}: unknown official item_id {item_id!r}")
        if set(scores) != set(test_item.hypotheses):
            raise DataError(f"{source}: official system coverage mismatch")
        pair = parsed_id.language_pair
        submitted_by_pair[pair].add(item_id)
        predictions = {}
        for system, score in scores.items():
            prediction = (
                score >= decision.threshold
                if decision.inclusive
                else score > decision.threshold
            )
            predictions[system] = prediction
            score_min = min(score_min, score)
            score_max = max(score_max, score)
            official_labels += 1
            converted_error_free += prediction
            for human in human_reference.judgments_for((item_id, system)):
                observations[pair].add(item_id, human.error_free, prediction)

        converted_line = json.dumps(
            {"item_id": item_id, "task3_pred": predictions},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        converted_digest.update(converted_line.encode("utf-8") + b"\n")

    if not submitted_by_pair:
        raise DataError(f"{segments_path}: no official predictions")
    for pair, actual_ids in submitted_by_pair.items():
        expected_ids = inventory.item_ids_by_pair.get(pair)
        if expected_ids is None or actual_ids != expected_ids:
            expected_count = 0 if expected_ids is None else len(expected_ids)
            raise DataError(
                f"{segments_path}: incomplete {pair} coverage; expected "
                f"{expected_count} items, found {len(actual_ids)}"
            )

    metric_rows = []
    for pair in sorted(submitted_by_pair):
        pair_observations = observations.get(pair)
        if pair_observations is None:
            raise DataError(f"{segments_path}: no human labels for {pair}")
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

    return ConversionResult(
        decision=decision,
        display_name=display_name,
        attempted_pairs=tuple(sorted(submitted_by_pair)),
        source_score_min=score_min,
        source_score_max=score_max,
        official_labels=official_labels,
        converted_error_free=converted_error_free,
        converted_error=official_labels - converted_error_free,
        converted_prediction_sha256=converted_digest.hexdigest(),
        metric_rows=tuple(metric_rows),
    )
