"""Load the current official test inventory used to align all inputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .io import DataError, iter_jsonl, parse_item_id
from .models import OfficialInventory, TestItem


def load_official_inventory(path: Path) -> OfficialInventory:
    """Read the authoritative test set into official/challenge coverage indexes."""
    items: dict[str, TestItem] = {}
    item_ids_by_pair: defaultdict[str, set[str]] = defaultdict(set)
    label_counts_by_pair: Counter[str] = Counter()
    challenge_items: dict[str, TestItem] = {}
    challenge_item_ids_by_pair: defaultdict[str, set[str]] = defaultdict(set)

    for line_number, record in iter_jsonl(path):
        source = f"{path}:{line_number}"
        item_id = record.get("item_id")
        parsed_id = parse_item_id(item_id, source)
        if parsed_id.set_id not in {"official", "challenge"}:
            raise DataError(f"{source}: unknown set {parsed_id.set_id!r}")
        destination = items if parsed_id.set_id == "official" else challenge_items
        if item_id in destination:
            raise DataError(f"{source}: duplicate item_id {item_id!r}")

        text = record.get("src")
        hypotheses = record.get("hyps")
        if not isinstance(text, str):
            raise DataError(f"{source}: src must be a string")
        if not isinstance(hypotheses, dict) or not hypotheses:
            raise DataError(f"{source}: hyps must be a non-empty object")
        if any(
            not isinstance(system, str)
            or not system
            or not isinstance(hypothesis, str)
            for system, hypothesis in hypotheses.items()
        ):
            raise DataError(f"{source}: invalid hypothesis map")

        item = TestItem(item_id, parsed_id, text, hypotheses)
        destination[item_id] = item
        if parsed_id.set_id == "official":
            item_ids_by_pair[parsed_id.language_pair].add(item_id)
            label_counts_by_pair[parsed_id.language_pair] += len(hypotheses)
        else:
            challenge_item_ids_by_pair[parsed_id.language_pair].add(item_id)

    if not items:
        raise DataError(f"{path}: no official items found")
    return OfficialInventory(
        items=items,
        item_ids_by_pair=dict(item_ids_by_pair),
        label_counts_by_pair=dict(label_counts_by_pair),
        challenge_items=challenge_items,
        challenge_item_ids_by_pair=dict(challenge_item_ids_by_pair),
    )
