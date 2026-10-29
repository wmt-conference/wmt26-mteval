"""Prediction discovery and strict finite-score validation."""
from __future__ import annotations
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from .constants import ITEM_ID_SEPARATOR, OFFICIAL_SET_ID
from .models import PredictionLoadResult, SubmissionDescriptor

def _parse_item_id(value: Any, context: str) -> tuple[str, str]:
    """Validate a six-field WMT26 item ID and return set ID and direction."""

    if not isinstance(value, str):
        raise ValueError(f"{context}: item_id must be a string")
    fields = value.split(ITEM_ID_SEPARATOR)
    if len(fields) != 6 or any(not field for field in fields):
        raise ValueError(f"{context}: invalid six-field item_id {value!r}")
    return fields[0], f"{fields[1]}-{fields[2]}"



def _submission_name(path: Path) -> str | None:
    """Read only the non-sensitive display name from an optional model card."""

    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    for entry in value:
        if not isinstance(entry, dict) or entry.get("question") != "Submission name":
            continue
        answer = entry.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    return None



def discover_submissions(
    directories: Path | Iterable[Path], selectors: set[str] | None = None
) -> list[SubmissionDescriptor]:
    """Discover native flat and conversion-generated nested Task 2 entries.

    A root may contain native ``<id>_segments.jsonl`` files or immediate child
    directories containing ``segments.jsonl``. Multiple roots make it possible
    to rank native entries together with ``converted/task2`` without copying or
    renaming either corpus.
    """

    roots = (directories,) if isinstance(directories, Path) else tuple(directories)
    if not roots:
        raise ValueError("At least one Task 2 submissions directory is required")
    by_id: dict[str, SubmissionDescriptor] = {}
    for directory in roots:
        if not directory.is_dir():
            raise ValueError(f"{directory}: Task 2 submissions directory is missing")
        candidates: list[tuple[str, Path, Path]] = []
        for scores_path in directory.glob("*_segments.jsonl"):
            if scores_path.name.endswith("_converted_segments.jsonl"):
                submission_id = scores_path.name[: -len("_converted_segments.jsonl")]
            else:
                submission_id = scores_path.name[: -len("_segments.jsonl")]
            candidates.append(
                (
                    submission_id,
                    scores_path,
                    directory / f"{submission_id}_model_card.json",
                )
            )
        for child in directory.iterdir():
            scores_path = child / "segments.jsonl"
            if not scores_path.is_file():
                scores_path = child / "predictions.jsonl"
            if child.is_dir() and scores_path.is_file():
                identity = child.name.split("_", 1)[0] if child.name[:1].isdigit() else child.name
                candidates.append((identity, scores_path, child / "model_card.json"))
        for submission_id, scores_path, card_path in sorted(
            candidates, key=lambda value: (value[0], str(value[1]))
        ):
            if selectors is not None and submission_id not in selectors:
                continue
            descriptor = SubmissionDescriptor(
                submission_id,
                scores_path,
                card_path,
                _submission_name(card_path),
            )
            previous = by_id.get(submission_id)
            if previous is not None:
                raise ValueError(
                    f"Duplicate Task 2 submission ID {submission_id!r}: "
                    f"{previous.scores_path} and {scores_path}"
                )
            by_id[submission_id] = descriptor
    if selectors is not None:
        missing = sorted(selectors - set(by_id))
        if missing:
            raise ValueError(f"Unknown Task 2 submissions: {', '.join(missing)}")
    if not by_id:
        roots_label = ", ".join(str(root) for root in roots)
        raise ValueError(f"{roots_label}: no selected Task 2 submissions found")
    return [by_id[submission_id] for submission_id in sorted(by_id)]



def _prediction_score(value: Any, context: str) -> float:
    """Validate and return one finite automatic segment quality score."""

    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{context}: prediction must be a finite JSON number")
    return float(value)



def load_predictions_for_directions(
    path: Path,
    test_sets: Mapping[str, dict[str, dict[str, str]]],
) -> dict[str, PredictionLoadResult]:
    """Validate several directions from one submission in one file pass.

    A direction with no official rows is reported as ``not_submitted`` rather
    than invalid.  A partially covered or malformed attempted direction is
    ``invalid`` without preventing other directions in the same file from
    being used.  Automatic-score range counts are diagnostic only: the public
    Task 2 submission contract requires numerical predictions but does not
    explicitly impose a [0, 100] input rejection rule.
    """

    markers: dict[str, str] = {}
    for direction in test_sets:
        source, target = direction.split("-", 1)
        markers[direction] = (
            f"{ITEM_ID_SEPARATOR}{source}{ITEM_ID_SEPARATOR}{target}"
            f"{ITEM_ID_SEPARATOR}"
        )
    predictions: dict[str, dict[str, dict[str, float]]] = {
        direction: {} for direction in test_sets
    }
    errors: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            direction = next(
                (value for value, marker in markers.items() if marker in line), None
            )
            if direction is None or direction in errors:
                continue
            context = f"{path}:{line_number}"
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{context}: each line must be an object")
                item_id = record.get("item_id")
                set_id, item_direction = _parse_item_id(item_id, context)
                if set_id != OFFICIAL_SET_ID or item_direction != direction:
                    continue
                expected_systems = test_sets[direction].get(item_id)
                if expected_systems is None:
                    raise ValueError(
                        f"{context}: unsupported official item {item_id!r}"
                    )
                direction_predictions = predictions[direction]
                if item_id in direction_predictions:
                    raise ValueError(f"{context}: duplicate item_id {item_id!r}")
                values = record.get("task2_pred")
                if not isinstance(values, dict) or not values:
                    raise ValueError(
                        f"{context}: task2_pred must be a non-empty object"
                    )
                unknown = sorted(set(values) - set(expected_systems))
                if unknown:
                    raise ValueError(
                        f"{context}: task2_pred contains unsupported systems "
                        f"{unknown[:3]}"
                    )
                missing_systems = sorted(set(expected_systems) - set(values))
                if missing_systems:
                    raise ValueError(
                        f"{context}: task2_pred is missing "
                        f"{len(missing_systems)} system(s)"
                    )
                direction_predictions[item_id] = {
                    system: _prediction_score(
                        values[system], f"{context}, system {system!r}"
                    )
                    for system in expected_systems
                }
            except json.JSONDecodeError as exc:
                errors[direction] = f"{context}: invalid JSON: {exc.msg}"
            except ValueError as exc:
                errors[direction] = str(exc)

    results: dict[str, PredictionLoadResult] = {}
    for direction, test_set in test_sets.items():
        if direction in errors:
            results[direction] = PredictionLoadResult(
                direction, "invalid", None, errors[direction]
            )
            continue
        values = predictions[direction]
        if not values:
            results[direction] = PredictionLoadResult(
                direction,
                "not_submitted",
                None,
                "no official predictions supplied for this direction",
            )
            continue
        missing_items = sorted(set(test_set) - set(values))
        if missing_items:
            results[direction] = PredictionLoadResult(
                direction,
                "invalid",
                None,
                f"direction {direction} is missing {len(missing_items)} official "
                "item(s)",
            )
            continue
        flat_scores = [
            score for item_scores in values.values() for score in item_scores.values()
        ]
        results[direction] = PredictionLoadResult(
            direction=direction,
            status="valid",
            predictions=values,
            message="complete and finite",
            score_count=len(flat_scores),
            minimum_score=min(flat_scores),
            maximum_score=max(flat_scores),
            scores_below_zero=sum(score < 0 for score in flat_scores),
            scores_above_hundred=sum(score > 100 for score in flat_scores),
        )
    return results
