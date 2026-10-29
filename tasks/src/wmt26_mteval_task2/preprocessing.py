"""Load shared ground truth and aggregate contrastive scores for Task 2.

Raw Pearmut conversion is shared with Task 1; use convert_annotation_dump.py.
"""
from __future__ import annotations
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from .constants import ITEM_ID_SEPARATOR, OFFICIAL_SET_ID
from .models import Screen, HumanEvidence, DirectionStatistics

def _finite_score(value: Any, context: str) -> float:
    """Validate and return one exact numeric, finite human quality score."""

    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{context}: score must be a finite JSON number")
    score = float(value)
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"{context}: human cESA score must be between 0 and 100")
    return score



def aggregate_human_evidence(
    test_set: Mapping[str, Mapping[str, str]],
    screens: Iterable[Screen],
    direction: str,
    *,
    excluded_systems: Iterable[str] = (),
) -> HumanEvidence:
    """Average segment/co-screen evidence after optional system exclusion.

    Excluded systems remain part of blindset and screen validation, but their
    human scores and every comparison involving them are absent from the
    returned scoring evidence.
    """

    direction_items = {
        item_id: hypotheses
        for item_id, hypotheses in test_set.items()
        if _item_direction(item_id) == direction
    }
    if not direction_items:
        raise ValueError(f"Direction {direction!r} is absent from the blindset")
    excluded = frozenset(excluded_systems)
    direction_systems = {
        system for hypotheses in direction_items.values() for system in hypotheses
    }
    unknown_exclusions = sorted(excluded - direction_systems)
    if unknown_exclusions:
        raise ValueError(
            f"Direction {direction!r} cannot exclude absent system(s) "
            f"{unknown_exclusions}"
        )
    incomplete_exclusions = sorted(
        system
        for system in excluded
        if any(system not in hypotheses for hypotheses in direction_items.values())
    )
    if incomplete_exclusions:
        raise ValueError(
            f"Direction {direction!r} excluded system(s) are not present for "
            f"every item: {incomplete_exclusions}"
        )
    retained_systems = direction_systems - excluded
    if len(retained_systems) < 2:
        raise ValueError(
            f"Direction {direction!r} must retain at least two systems"
        )
    selected_screens = [screen for screen in screens if screen.direction == direction]
    if not selected_screens:
        raise ValueError(f"Direction {direction!r} has no retained Task 2 screens")

    item_system_values: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(
        list
    )
    co_screen_values: dict[tuple[str, str, str], list[tuple[float, float, str]]] = (
        defaultdict(list)
    )
    for screen in selected_screens:
        hypotheses = direction_items.get(screen.item_id)
        if hypotheses is None:
            raise ValueError(
                f"Screen {screen.screen_id!r} item is absent from direction {direction}"
            )
        if len(screen.systems) != 3 or tuple(screen.scores) != screen.systems:
            raise ValueError(
                f"Screen {screen.screen_id!r} must preserve exactly three ordered scores"
            )
        for system in screen.systems:
            if system not in hypotheses:
                raise ValueError(
                    f"Screen {screen.screen_id!r} system {system!r} is absent "
                    "from its blindset item"
                )
            if system in excluded:
                continue
            item_system_values[(screen.item_id, system)].append(
                (screen.scores[system], screen.annotator_id)
            )
        scoring_systems = tuple(
            system for system in screen.systems if system not in excluded
        )
        for first_index, first in enumerate(scoring_systems):
            for second in scoring_systems[first_index + 1 :]:
                system_a, system_b = sorted((first, second))
                co_screen_values[(screen.item_id, system_a, system_b)].append(
                    (
                        screen.scores[system_a],
                        screen.scores[system_b],
                        screen.annotator_id,
                    )
                )

    segment_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for (item_id, system), values in item_system_values.items():
        segment_scores[item_id][system] = sum(value for value, _ in values) / len(
            values
        )

    co_screen_scores = {
        key: (
            sum(value[0] for value in values) / len(values),
            sum(value[1] for value in values) / len(values),
        )
        for key, values in co_screen_values.items()
    }

    expected_pairs = {
        (item_id, system)
        for item_id, hypotheses in direction_items.items()
        for system in hypotheses
        if system not in excluded
    }
    observed_pairs = set(item_system_values)
    item_counts = [len(values) for values in item_system_values.values()]
    pair_counts = [len(values) for values in co_screen_values.values()]
    statistics = DirectionStatistics(
        direction=direction,
        blindset_items=len(direction_items),
        blindset_item_systems=len(expected_pairs),
        retained_screens=len(selected_screens),
        human_score_observations=sum(
            system not in excluded
            for screen in selected_screens
            for system in screen.systems
        ),
        missing_item_systems=len(expected_pairs - observed_pairs),
        item_systems=len(item_system_values),
        repeated_item_systems=sum(count > 1 for count in item_counts),
        extra_item_system_scores=sum(count - 1 for count in item_counts),
        maximum_item_system_scores=max(item_counts, default=0),
        multiple_distinct_annotators=sum(
            len({annotator for _, annotator in values}) > 1
            for values in item_system_values.values()
        ),
        same_annotator_repeated=sum(
            any(count > 1 for count in Counter(a for _, a in values).values())
            for values in item_system_values.values()
        ),
        co_screen_item_pairs=len(co_screen_values),
        repeated_co_screen_pairs=sum(count > 1 for count in pair_counts),
        extra_pair_screens=sum(count - 1 for count in pair_counts),
        maximum_pair_screens=max(pair_counts, default=0),
    )
    systems = tuple(sorted(retained_systems))
    return HumanEvidence(
        direction,
        systems,
        tuple(sorted(direction_items)),
        {item: dict(values) for item, values in segment_scores.items()},
        co_screen_scores,
        statistics,
    )



def _item_direction(item_id: str) -> str:
    """Validate a six-field official item ID and return its direction."""

    if not isinstance(item_id, str):
        raise ValueError("Task 2 evidence item_id must be a string")
    fields = item_id.split(ITEM_ID_SEPARATOR)
    if len(fields) != 6 or any(not field for field in fields):
        raise ValueError(f"Invalid blindset item_id: {item_id!r}")
    if fields[0] != OFFICIAL_SET_ID:
        raise ValueError(f"Task 2 evidence item is not official: {item_id!r}")
    return f"{fields[1]}-{fields[2]}"



def _screen_from_payload(
    item_id: Any,
    payload: Any,
    context: str,
    seen_ids: set[str],
) -> Screen:
    """Validate one legacy or unified screen payload and return its model."""

    if not isinstance(payload, dict):
        raise ValueError(f"{context}: screen must be an object")
    screen_id = payload.get("screen_id")
    annotator_id = payload.get("annotator_id")
    unified = "task1_pred" in payload or "task2_pred" in payload
    score_mapping = payload.get("task2_pred") if unified else payload.get("scores")
    assessments = payload.get("task1_pred") if unified else None
    if not isinstance(screen_id, str) or not screen_id:
        raise ValueError(f"{context}: screen_id must be a non-empty string")
    if screen_id in seen_ids:
        raise ValueError(f"{context}: duplicate screen_id {screen_id!r}")
    seen_ids.add(screen_id)
    direction = _item_direction(item_id)
    if not isinstance(annotator_id, str) or not annotator_id:
        raise ValueError(f"{context}: annotator_id must be a non-empty string")
    if not isinstance(score_mapping, dict) or len(score_mapping) != 3:
        raise ValueError(
            f"{context}: task2_pred must contain exactly three systems"
            if unified
            else f"{context}: scores must contain exactly three systems"
        )
    normalized: dict[str, float] = {}
    for system, score in score_mapping.items():
        if not isinstance(system, str) or not system:
            raise ValueError(f"{context}: system names must be non-empty strings")
        normalized[system] = _finite_score(score, f"{context}, {system!r}")
    normalized_assessments: dict[str, Mapping[str, Any]] | None = None
    if unified:
        if not isinstance(assessments, dict) or tuple(assessments) != tuple(normalized):
            raise ValueError(
                f"{context}: task1_pred and task2_pred must have the same "
                "three systems in the same order"
            )
        normalized_assessments = {}
        for system, assessment in assessments.items():
            if not isinstance(assessment, dict):
                raise ValueError(
                    f"{context}, system {system!r}: task1_pred must be an object"
                )
            if "annotator_id" in assessment or "context_models" in assessment:
                raise ValueError(
                    f"{context}, system {system!r}: annotator and context belong "
                    "to the enclosing screen"
                )
            enriched = dict(assessment)
            enriched["annotator_id"] = annotator_id
            enriched["context_models"] = [
                other for other in normalized if other != system
            ]
            normalized_assessments[system] = enriched
    return Screen(
        screen_id,
        item_id,
        direction,
        annotator_id,
        tuple(normalized),
        normalized,
        normalized_assessments,
    )



def load_screens(path: Path) -> list[Screen]:
    """Load unified references or legacy Task 2 screen JSONL.

    One file must use one layout throughout. Unified files contain one item per
    line with nested screens; legacy files contain one top-level screen per
    line. Both layouts are normalized to the same public Screen list.
    """

    screens: list[Screen] = []
    seen_ids: set[str] = set()
    seen_items: set[str] = set()
    layout: str | None = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            context = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{context}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{context}: each line must be an object")
            record_layout = (
                "unified"
                if "screens" in record
                else "legacy"
                if "screen_id" in record and "scores" in record
                else "unsupported"
            )
            if record_layout == "unsupported":
                raise ValueError(
                    f"{context}: expected unified item screens or a legacy Task 2 screen"
                )
            if layout is None:
                layout = record_layout
            elif layout != record_layout:
                raise ValueError(
                    f"{context}: human-reference layouts must not be mixed"
                )

            item_id = record.get("item_id")
            if record_layout == "legacy":
                screens.append(_screen_from_payload(item_id, record, context, seen_ids))
                continue
            _item_direction(item_id)
            if item_id in seen_items:
                raise ValueError(f"{context}: duplicate item_id {item_id!r}")
            seen_items.add(item_id)
            values = record.get("screens")
            if not isinstance(values, list) or not values:
                raise ValueError(f"{context}: screens must be a non-empty list")
            for index, payload in enumerate(values):
                screens.append(
                    _screen_from_payload(
                        item_id,
                        payload,
                        f"{context}, screens[{index}]",
                        seen_ids,
                    )
                )
    if not screens:
        raise ValueError(f"{path}: human-reference screen file is empty")
    return screens
