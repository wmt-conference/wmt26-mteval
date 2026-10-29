"""Read the original Task 1 inputs without materializing converted submissions."""

from __future__ import annotations

import json
from pathlib import Path

from .conversion_labels import CONVERSION_ENTRIES
from .models import SubmissionDescriptor


def discover_conversion_inputs(submissions: Path, dedup: Path, original: Path):
    """Resolve conversion identities in the evaluator catalog against input roots.

    Participant directory names must be exact catalog IDs. No extra error
    normalization or deduplication is applied. The published comparison used
    two supplied baseline snapshots, which must stay distinct. GPT Sol
    medium/xhigh conversions are not in the eligible paper catalog.
    """
    found = {}
    for root, prefix, suffix in (
        (submissions, "", ""), (dedup, "baseline_", "__deduplicated"),
        (original, "baseline_nondedup_", "__pre_dedup_pre_enhy_fix"),
    ):
        if not root.is_dir():
            raise ValueError(f"Missing Task 1 input directory: {root}")
        for path in sorted(root.glob("*/predictions.jsonl")):
            identity = prefix + path.parent.name.removesuffix(suffix)
            if identity not in CONVERSION_ENTRIES:
                continue
            if identity in found:
                raise ValueError(f"Duplicate raw Task 1 identity: {identity}")
            found[identity] = SubmissionDescriptor(identity, path, path.with_name("model_card.json"))
    missing = set(CONVERSION_ENTRIES) - set(found)
    if missing:
        raise ValueError(f"Missing conversion inputs: {', '.join(sorted(missing))}")
    return [found[identity] for identity in sorted(found)]


def load_hypothesis_lengths(path: Path) -> dict[str, dict[str, int]]:
    """Return exact code-point lengths needed by the historical span conversion."""
    result = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item = row["item_id"]
            if item in result:
                raise ValueError(f"{path}:{number}: duplicate item ID")
            result[item] = {system: len(text) for system, text in row["hyps"].items()}
    return result
