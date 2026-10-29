"""Upstream analysis regression tests with release-package imports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wmt26_mteval_task3.canonical_reference import load_canonical_human_reference
from wmt26_mteval_task3.models import (
    OfficialInventory,
    ParsedItemId,
    TestItem as InventoryItem,
)


ITEM_ID = "official_###_eng_###_deu_###_news_###_doc_###_1"


def inventory() -> OfficialInventory:
    """Build or process the synthetic inventory fixture for the supplied test inputs."""
    parsed = ParsedItemId("official", "eng", "deu", "news", "doc", "1")
    item = InventoryItem(
        ITEM_ID,
        parsed,
        "source",
        {"one": "first", "two": "second", "three": "third"},
    )
    return OfficialInventory(
        items={ITEM_ID: item},
        item_ids_by_pair={"eng---deu": {ITEM_ID}},
        label_counts_by_pair={"eng---deu": 3},
        challenge_items={},
        challenge_item_ids_by_pair={},
    )


def assessment(*, errors=None, omission=None, instruction_fault=None) -> dict:
    """Build or process the synthetic assessment fixture for the supplied test inputs."""
    return {
        "errors": [] if errors is None else errors,
        "omission": omission,
        "instruction_fault": instruction_fault,
    }


class CanonicalReferenceTest(unittest.TestCase):
    """Group original canonical reference test regression scenarios."""
    def test_retains_every_screen_assessment(self) -> None:
        """Verify retains every screen assessment using synthetic upstream fixtures."""
        clean = {system: assessment() for system in ("one", "two", "three")}
        erroneous = {
            "one": assessment(
                errors=[
                    {
                        "start": 0,
                        "end": 1,
                        "severity": "major",
                        "category": None,
                    }
                ]
            ),
            "two": assessment(omission="minor"),
            "three": assessment(),
        }
        record = {
            "item_id": ITEM_ID,
            "screens": [
                {
                    "screen_id": "eng_###_deu_###_0",
                    "annotator_id": "first-user",
                    "task1_pred": clean,
                    "task2_pred": {"one": 90, "two": 90, "three": 90},
                },
                {
                    "screen_id": "eng_###_deu_###_1",
                    "annotator_id": "second-user",
                    "task1_pred": erroneous,
                    "task2_pred": {"one": 90, "two": 90, "three": 80},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "references.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            reference = load_canonical_human_reference(path, inventory())

        judgments = reference.judgments_for((ITEM_ID, "one"))
        self.assertEqual([value.error_free for value in judgments], [True, False])
        self.assertEqual(reference.stats["selected_labels"], 6)
        self.assertEqual(reference.stats["selected_keys"], 3)
        self.assertEqual(reference.judgment_counts[(ITEM_ID, "one")], 2)
        self.assertEqual(judgments[1].spans[0].offset_unit, "unicode_code_point")


if __name__ == "__main__":
    unittest.main()
