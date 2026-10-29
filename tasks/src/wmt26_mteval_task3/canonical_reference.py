"""Load canonical multi-screen PEAR-MUT references for binary Task 3 scoring."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import DataError, iter_jsonl, parse_item_id
from .models import (
    HumanJudgment,
    HumanReference,
    NormalizedSpan,
    OfficialInventory,
)


def _severity(value: Any, source: str) -> str:
    """Validate and return a canonical minor/major severity value."""
    if value not in ("minor", "major"):
        raise DataError(f"{source}: severity must be minor or major")
    return value


def _assessment(
    value: Any,
    score: Any,
    target: str,
    pair: str,
    item_id: str,
    system: str,
    annotator_id: str,
    screen_id: str,
    line_number: int,
    screen_index: int,
    source: str,
) -> HumanJudgment:
    """Validate one canonical Task 1/Task 2 assessment and binarize it."""
    if not isinstance(value, dict):
        raise DataError(f"{source}: task1_pred must be an object")
    if set(value) != {"errors", "omission", "instruction_fault"}:
        raise DataError(f"{source}: unexpected Task 1 assessment fields")
    errors = value["errors"]
    omission = value["omission"]
    instruction_fault = value["instruction_fault"]
    if not isinstance(errors, list):
        raise DataError(f"{source}: errors must be a list")
    if omission not in (None, "minor", "major"):
        raise DataError(f"{source}: invalid omission value {omission!r}")
    if instruction_fault not in (None, "minor", "major"):
        raise DataError(
            f"{source}: invalid instruction_fault value {instruction_fault!r}"
        )
    spans = []
    for span_index, span in enumerate(errors):
        span_source = f"{source}, error {span_index}"
        if not isinstance(span, dict):
            raise DataError(f"{span_source}: expected an object")
        start = span.get("start")
        end = span.get("end")
        if type(start) is not int or type(end) is not int:
            raise DataError(f"{span_source}: offsets must be integers")
        if start < 0 or end <= start or end > len(target):
            raise DataError(
                f"{span_source}: invalid code-point offsets {start}:{end} for "
                f"target length {len(target)}"
            )
        spans.append(
            NormalizedSpan(
                start,
                end,
                span.get("category"),
                _severity(span.get("severity"), span_source),
                "ordinary",
                "unicode_code_point",
            )
        )
    if omission is not None:
        spans.append(
            NormalizedSpan(
                len(target),
                len(target),
                None,
                omission,
                "omission",
                "unicode_code_point",
            )
        )
    if instruction_fault is not None:
        spans.append(
            NormalizedSpan(
                len(target) + 1,
                len(target) + 1,
                None,
                instruction_fault,
                "instruction_error",
                "unicode_code_point",
            )
        )
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise DataError(f"{source}: task2_pred must be numeric")
    numeric_score = float(score)
    if not math.isfinite(numeric_score) or not 0 <= numeric_score <= 100:
        raise DataError(f"{source}: task2_pred must be finite and in [0, 100]")
    clean = not errors and omission is None and instruction_fault is None
    return HumanJudgment(
        language_pair=pair,
        item_id=item_id,
        system=system,
        annotator_id=annotator_id,
        submit_time=0.0,
        score=numeric_score,
        error_free=numeric_score >= 85 and clean,
        spans=tuple(spans),
        group_name=f"canonical:{screen_id}",
        record_index=line_number,
        item_index=screen_index,
    )


def load_canonical_human_reference(
    path: Path, inventory: OfficialInventory
) -> HumanReference:
    """Load canonical item/screens JSONL and retain every human assessment."""
    collected: defaultdict[tuple[str, str], list[HumanJudgment]] = defaultdict(list)
    seen_items: set[str] = set()
    seen_screens: set[str] = set()
    pairs: set[str] = set()
    stats: Counter[str] = Counter()

    for line_number, record in iter_jsonl(path):
        source = f"{path}:{line_number}"
        if set(record) != {"item_id", "screens"}:
            raise DataError(f"{source}: expected item_id and screens")
        item_id = record["item_id"]
        parsed = parse_item_id(item_id, source)
        if parsed.set_id != "official" or item_id not in inventory.items:
            raise DataError(f"{source}: unknown official item_id {item_id!r}")
        if item_id in seen_items:
            raise DataError(f"{source}: duplicate item_id {item_id!r}")
        seen_items.add(item_id)
        pair = parsed.language_pair
        pairs.add(pair)
        test_item = inventory.items[item_id]
        screens = record["screens"]
        if not isinstance(screens, list) or not screens:
            raise DataError(f"{source}: screens must be a nonempty list")
        stats["records"] += 1

        for screen_index, screen in enumerate(screens):
            screen_source = f"{source}, screen {screen_index}"
            if not isinstance(screen, dict) or set(screen) != {
                "screen_id",
                "annotator_id",
                "task1_pred",
                "task2_pred",
            }:
                raise DataError(f"{screen_source}: invalid screen fields")
            screen_id = screen["screen_id"]
            annotator_id = screen["annotator_id"]
            if not isinstance(screen_id, str) or not screen_id or screen_id in seen_screens:
                raise DataError(f"{screen_source}: invalid or duplicate screen_id")
            if not isinstance(annotator_id, str) or not annotator_id:
                raise DataError(f"{screen_source}: invalid annotator_id")
            seen_screens.add(screen_id)
            task1 = screen["task1_pred"]
            task2 = screen["task2_pred"]
            if (
                not isinstance(task1, dict)
                or not isinstance(task2, dict)
                or len(task1) != 3
                or tuple(task1) != tuple(task2)
            ):
                raise DataError(f"{screen_source}: inconsistent three-system mappings")
            if any(system not in test_item.hypotheses for system in task1):
                raise DataError(f"{screen_source}: system absent from official item")
            stats["screens"] += 1
            for system in task1:
                judgment = _assessment(
                    task1[system],
                    task2[system],
                    test_item.hypotheses[system],
                    pair,
                    item_id,
                    system,
                    annotator_id,
                    screen_id,
                    line_number,
                    screen_index,
                    f"{screen_source}, system {system!r}",
                )
                collected[(item_id, system)].append(judgment)
                stats["selected_labels"] += 1
                stats[
                    "selected_error_free" if judgment.error_free else "selected_error"
                ] += 1

    assessments = {key: tuple(values) for key, values in collected.items()}
    labels = {key: values[0] for key, values in assessments.items()}
    judgment_counts = {key: len(values) for key, values in assessments.items()}
    annotator_counts = {
        key: len({judgment.annotator_id for judgment in values})
        for key, values in assessments.items()
    }
    stats["groups"] = len(pairs)
    stats["selected_keys"] = len(labels)
    stats["repeated_keys"] = sum(count > 1 for count in judgment_counts.values())
    stats["extra_assessments"] = sum(count - 1 for count in judgment_counts.values())
    return HumanReference(
        labels,
        set(),
        judgment_counts,
        annotator_counts,
        dict(stats),
        assessments,
    )
