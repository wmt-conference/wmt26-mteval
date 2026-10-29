"""Upstream analysis regression tests with release-package imports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wmt26_mteval_task3.models import (
    HumanJudgment,
    HumanReference,
    OfficialInventory,
    ParsedItemId,
    TestItem as InventoryItem,
)
from wmt26_mteval_task3.task2_conversion import (
    convert_and_evaluate,
    inspect_conversion_decision,
    task2_input_digest,
)


ITEM_ID = "official_###_eng_###_deu_###_news_###_doc_###_1"


def model_card(permission, threshold, inclusivity) -> list[dict]:
    """Build or process the synthetic model card fixture for the supplied test inputs."""
    return [
        {"question": "Organization name", "answer": "Organization"},
        {"question": "Submission name", "answer": "Metric"},
        {
            "question": (
                "Does your system use the provided reference translations in "
                "any form? (true|false)"
            ),
            "answer": "false",
        },
        {
            "question": (
                "Allow automatic conversion from Task 2 to Task 3 (true|false)"
            ),
            "answer": permission,
        },
        {
            "question": (
                "Task 2 to 3 conversion threshold (everything above this score "
                "will be regarded as error-free)"
            ),
            "answer": threshold,
        },
        {
            "question": (
                "Should the above threshold be inclusive (≥) or exclusive (>)?"
            ),
            "answer": inclusivity,
        },
    ]


def write_submission(
    directory: Path,
    submission_id: int,
    permission,
    threshold,
    inclusivity,
    score: float = 85,
) -> None:
    """Build or process the synthetic write submission fixture for the supplied test inputs."""
    (directory / f"{submission_id}_model_card.json").write_text(
        json.dumps(model_card(permission, threshold, inclusivity)),
        encoding="utf-8",
    )
    record = {"item_id": ITEM_ID, "task2_pred": {"system": score}}
    (directory / f"{submission_id}_segments.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )


def inventory() -> OfficialInventory:
    """Build or process the synthetic inventory fixture for the supplied test inputs."""
    parsed = ParsedItemId("official", "eng", "deu", "news", "doc", "1")
    item = InventoryItem(ITEM_ID, parsed, "source", {"system": "target"})
    return OfficialInventory(
        items={ITEM_ID: item},
        item_ids_by_pair={"eng---deu": {ITEM_ID}},
        label_counts_by_pair={"eng---deu": 1},
        challenge_items={},
        challenge_item_ids_by_pair={},
    )


def reference() -> HumanReference:
    """Build or process the synthetic reference fixture for the supplied test inputs."""
    judgment = HumanJudgment(
        language_pair="eng---deu",
        item_id=ITEM_ID,
        system="system",
        annotator_id="user",
        submit_time=1,
        score=100,
        error_free=True,
        spans=(),
        group_name="eng---deu v3",
        record_index=0,
        item_index=0,
    )
    key = (ITEM_ID, "system")
    return HumanReference({key: judgment}, set(), {key: 1}, {key: 1}, {})


class ConversionDecisionTest(unittest.TestCase):
    """Group original conversion decision test regression scenarios."""
    def test_accepts_json_boolean_and_inclusive_rule(self) -> None:
        """Verify accepts json boolean and inclusive rule using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            write_submission(directory, 1, True, "85", "inclusive (≥)")

            decision = inspect_conversion_decision(1, directory)

        self.assertEqual(decision.status, "converted")
        self.assertTrue(decision.permission)
        self.assertTrue(decision.inclusive)
        self.assertEqual(decision.threshold, 85)
        self.assertEqual(decision.comparison, ">=")

    def test_applies_reviewed_base_model_override(self) -> None:
        """Verify applies reviewed base model override using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            write_submission(directory, 1, True, "85", "inclusive")

            decision = inspect_conversion_decision(
                1, directory, {1: "reviewed-model"}
            )

        self.assertEqual(decision.base_model, "reviewed-model")

    def test_opt_out_and_missing_rule_are_not_converted(self) -> None:
        """Verify opt out and missing rule are not converted using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            write_submission(directory, 1, "false", "", "")
            write_submission(directory, 2, "true", "", "")

            opted_out = inspect_conversion_decision(1, directory)
            missing = inspect_conversion_decision(2, directory)

        self.assertEqual(opted_out.status, "opted_out")
        self.assertEqual(missing.status, "missing_threshold")

    def test_inclusive_and_exclusive_thresholds_differ_at_boundary(self) -> None:
        """Verify inclusive and exclusive thresholds differ at boundary using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            write_submission(directory, 1, "true", "85", "inclusive", score=85)
            write_submission(directory, 2, "true", "85", "exclusive", score=85)
            inclusive = inspect_conversion_decision(1, directory)
            exclusive = inspect_conversion_decision(2, directory)

            inclusive_result = convert_and_evaluate(
                inclusive, "Inclusive", directory, inventory(), reference()
            )
            exclusive_result = convert_and_evaluate(
                exclusive, "Exclusive", directory, inventory(), reference()
            )

        self.assertEqual(inclusive_result.converted_error_free, 1)
        self.assertEqual(exclusive_result.converted_error_free, 0)
        self.assertEqual(inclusive_result.metric_rows[0]["f1"], 1.0)
        self.assertEqual(exclusive_result.metric_rows[0]["f1"], 0.0)

    def test_challenge_scores_are_ignored_before_validation(self) -> None:
        """Verify challenge scores are ignored before validation using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            write_submission(directory, 1, "true", "85", "inclusive", score=85)
            segments_path = directory / "1_segments.jsonl"
            official = segments_path.read_text(encoding="utf-8")
            challenge = json.dumps(
                {
                    "item_id": (
                        "challenge_###_x_###_y_###_domain_###_doc_###_1"
                    ),
                    "task2_pred": {"system": "not numeric"},
                }
            )
            segments_path.write_text(challenge + "\n" + official, encoding="utf-8")
            decision = inspect_conversion_decision(1, directory)

            result = convert_and_evaluate(
                decision, "System", directory, inventory(), reference()
            )

        self.assertEqual(result.official_labels, 1)

    def test_aggregate_input_digest_changes_with_source(self) -> None:
        """Verify aggregate input digest changes with source using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            write_submission(directory, 1, "true", "85", "inclusive")
            original = task2_input_digest([1], directory)
            (directory / "1_segments.jsonl").write_text("changed\n", encoding="utf-8")

            changed = task2_input_digest([1], directory)

        self.assertNotEqual(original, changed)


if __name__ == "__main__":
    unittest.main()
