"""Load predictions on precisely the retained human-supported translations."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from .io import _parse_item_id
from .conversion import task1_to_task2_score

@dataclass(frozen=True)
class PredictionOutcome:
    status: str
    predictions: dict[str, dict[str, float]] | None
    message: str
    required_scores: int



def _load_human_supported_predictions(
    path: Path,
    requirements: dict[str, dict[str, set[str]]],
    *,
    hypothesis_lengths: dict[str, dict[str, int]] | None = None,
) -> dict[str, PredictionOutcome]:
    """Read exactly the predictions needed by retained human evidence.

    A system must provide every human-supported item/system score for a
    direction to enter that direction's ranking. Scores for unannotated
    translations are intentionally irrelevant to a Task 2 score and neither
    help nor prevent eligibility here.
    """

    directions = tuple(requirements)
    markers = {
        direction: f"_###_{direction.split('-', 1)[0]}_###_{direction.split('-', 1)[1]}_###"
        for direction in directions
    }
    found: dict[str, dict[str, dict[str, float]]] = {direction: {} for direction in directions}
    errors: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            direction = next((key for key, marker in markers.items() if marker in line), None)
            if direction is None or direction in errors:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("row must be a JSON object")
                item_id = record.get("item_id")
                set_id, parsed_direction = _parse_item_id(item_id, f"{path}:{line_number}")
                if set_id != "official" or parsed_direction != direction:
                    continue
                required_systems = requirements[direction].get(item_id)
                if required_systems is None:
                    continue
                if item_id in found[direction]:
                    raise ValueError(f"duplicate human-supported item {item_id!r}")
                values = record.get("task2_pred")
                if hypothesis_lengths is not None:
                    raw = record.get("task1_pred")
                    if not isinstance(raw, dict):
                        raise ValueError("task1_pred must be an object")
                    values = {
                        system: task1_to_task2_score(raw[system], hypothesis_lengths[item_id][system])
                        for system in required_systems if system in raw
                    }
                if not isinstance(values, dict):
                    raise ValueError("task2_pred must be an object")
                missing = sorted(required_systems - set(values))
                if missing:
                    raise ValueError(f"missing {len(missing)} human-supported system score(s)")
                scored: dict[str, float] = {}
                for system in required_systems:
                    value = values[system]
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError(f"non-finite score for system {system!r}")
                    scored[system] = float(value)
                found[direction][item_id] = scored
            except (json.JSONDecodeError, ValueError) as exc:
                errors[direction] = f"{path}:{line_number}: {exc}"

    outcomes: dict[str, PredictionOutcome] = {}
    for direction, required in requirements.items():
        if direction in errors:
            outcomes[direction] = PredictionOutcome("invalid", None, errors[direction], 0)
            continue
        missing_items = sorted(set(required) - set(found[direction]))
        if missing_items:
            status = "not_submitted" if not found[direction] else "invalid"
            outcomes[direction] = PredictionOutcome(
                status, None,
                f"missing {len(missing_items)} human-supported item(s)", 0,
            )
            continue
        outcomes[direction] = PredictionOutcome(
            "valid", found[direction], "", sum(len(systems) for systems in required.values())
        )
    return outcomes
