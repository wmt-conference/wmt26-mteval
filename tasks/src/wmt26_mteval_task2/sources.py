"""Structural source identities shared across target-language directions."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from .significance import source_key

def validate_sources(path, items):
    """Check shared structural source IDs against text without text deduplication."""
    texts, counts, seen = {}, Counter(), set()
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            item = record["item_id"]
            if item not in items:
                continue
            if item in seen:
                raise ValueError(f"Duplicate blindset item: {item}")
            seen.add(item)
            source = source_key(item)
            value = record.get("src", record.get("source"))
            text = value.get("text") if isinstance(value, dict) else value
            if not isinstance(text, str) or source in texts and texts[source] != text:
                raise ValueError(f"Inconsistent source identity: {source}")
            texts[source] = text
            counts[source] += 1
    if seen != items:
        raise ValueError("Some retained items are absent from the blindset")
    return {key: i for i, key in enumerate(sorted(texts))}, dict(
        direction_items=len(items), source_units=len(texts),
        shared_source_units=sum(n > 1 for n in counts.values()),
        documents=len({key[:-1] for key in texts}),
        source_multiplicities=dict(Counter(counts.values())))
