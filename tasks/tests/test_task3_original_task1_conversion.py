"""Upstream analysis regression tests with release-package imports."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wmt26_mteval_task3.io import DataError
from wmt26_mteval_task3.models import (
    HumanJudgment,
    HumanReference,
    OfficialInventory,
    ParsedItemId,
    TestItem as InventoryItem,
)
from wmt26_mteval_task3.task1_conversion import (
    convert_task1_and_evaluate,
    inspect_task1_decision,
    load_task1_submission_ids,
    task1_input_digest,
)


ITEM_ID = "official_###_eng_###_deu_###_news_###_doc_###_1"


def model_card(permission) -> list[dict]:
    """Build or process the synthetic model card fixture for the supplied test inputs."""
    return [
        {"question": "Organization name", "answer": "Organization"},
        {"question": "Submission name", "answer": "Span Metric"},
        {
            "question": (
                "Does your system use the provided reference translations in "
                "any form? (true|false)"
            ),
            "answer": "false",
        },
        {
            "question": (
                "Allow automatic conversion from Task 1 to Task 3 (true|false)"
            ),
            "answer": permission,
        },
    ]


def write_submission(
    root: Path, submission_id: int, permission, judgment: dict
) -> None:
    """Build or process the synthetic write submission fixture for the supplied test inputs."""
    directory = root / f"{submission_id}_account_submission"
    directory.mkdir()
    (directory / "model_card.json").write_text(
        json.dumps(model_card(permission)), encoding="utf-8"
    )
    record = {"item_id": ITEM_ID, "task1_pred": {"system": judgment}}
    (directory / "predictions.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )


def inventory(target: str = "target") -> OfficialInventory:
    """Build or process the synthetic inventory fixture for the supplied test inputs."""
    parsed = ParsedItemId("official", "eng", "deu", "news", "doc", "1")
    item = InventoryItem(ITEM_ID, parsed, "source", {"system": target})
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


class Task1ConversionTest(unittest.TestCase):
    """Group original task1 conversion test regression scenarios."""
    def test_requires_no_errors_or_omission(self) -> None:
        """Verify requires no errors or omission using synthetic upstream fixtures."""
        cases = [
            ({"errors": [], "omission": None}, 1),
            ({"errors": [], "omission": "major"}, 0),
            ({"errors": [], "omission": None, "instruction_fault": "minor"}, 0),
        ]
        for index, (judgment, expected_clean) in enumerate(cases, start=1):
            with self.subTest(judgment=judgment):
                with tempfile.TemporaryDirectory() as temporary_dir:
                    root = Path(temporary_dir)
                    write_submission(root, index, "true", judgment)
                    decision = inspect_task1_decision(index, root)
                    result = convert_task1_and_evaluate(
                        decision, "System", root, inventory(), reference()
                    )
                self.assertEqual(result.converted_error_free, expected_clean)

    def test_counts_explicit_instruction_fault_extension(self) -> None:
        """Verify counts explicit instruction fault extension using synthetic upstream fixtures."""
        judgment = {
            "errors": [],
            "omission": None,
            "instruction_fault": "major",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_submission(root, 1, True, judgment)
            decision = inspect_task1_decision(1, root)
            result = convert_task1_and_evaluate(
                decision, "System", root, inventory(), reference()
            )

        self.assertEqual(result.top_level_instruction_fault_labels, 1)
        self.assertEqual(result.converted_error_free, 0)

    def test_rejects_composite_instruction_fault(self) -> None:
        """Verify rejects composite instruction fault using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_submission(
                root,
                1,
                True,
                {"errors": [], "omission": None, "instruction_fault": []},
            )
            decision = inspect_task1_decision(1, root)

            with self.assertRaisesRegex(DataError, "invalid instruction_fault"):
                convert_task1_and_evaluate(
                    decision, "System", root, inventory(), reference()
                )

    def test_decodes_special_marker_spans_as_nonempty_errors(self) -> None:
        """Verify decodes special marker spans as nonempty errors using synthetic upstream fixtures."""
        cases = [
            (6, "omission_marker_spans"),
            (7, "instruction_error_marker_spans"),
        ]
        for submission_id, (offset, count_field) in enumerate(cases, start=1):
            judgment = {
                "errors": [
                    {
                        "start": offset,
                        "end": offset,
                        "severity": "major",
                        "category": None,
                    }
                ],
                "omission": None,
            }
            with self.subTest(count_field=count_field):
                with tempfile.TemporaryDirectory() as temporary_dir:
                    root = Path(temporary_dir)
                    write_submission(root, submission_id, True, judgment)
                    decision = inspect_task1_decision(submission_id, root)
                    result = convert_task1_and_evaluate(
                        decision, "System", root, inventory(), reference()
                    )

                self.assertEqual(result.zero_length_spans, 1)
                self.assertEqual(result.submitted_error_spans, 1)
                self.assertEqual(getattr(result, count_field), 1)
                self.assertEqual(result.converted_error_free, 0)

    def test_participant_marker_offsets_use_code_points(self) -> None:
        """Verify participant marker offsets use code points using synthetic upstream fixtures."""
        target = "a😀"
        judgment = {
            "errors": [
                {"start": 2, "end": 2, "severity": "major", "category": None}
            ],
            "omission": None,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_submission(root, 1, True, judgment)
            decision = inspect_task1_decision(1, root)
            result = convert_task1_and_evaluate(
                decision, "System", root, inventory(target), reference()
            )

        self.assertEqual(len(target), 2)
        self.assertEqual(result.omission_marker_spans, 1)
        self.assertEqual(result.instruction_error_marker_spans, 0)

    def test_preserves_nonstandard_zero_length_error_as_nonempty(self) -> None:
        """Verify preserves nonstandard zero length error as nonempty using synthetic upstream fixtures."""
        judgment = {
            "errors": [
                {"start": 2, "end": 2, "severity": "major", "category": "omission"}
            ],
            "omission": None,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_submission(root, 1, True, judgment)
            decision = inspect_task1_decision(1, root)
            result = convert_task1_and_evaluate(
                decision, "System", root, inventory(), reference()
            )

        self.assertEqual(result.zero_length_spans, 1)
        self.assertEqual(result.submitted_error_spans, 1)
        self.assertEqual(result.nonstandard_zero_length_spans, 1)
        self.assertEqual(result.converted_error_free, 0)

    def test_requires_explicit_omission_field(self) -> None:
        """Verify requires explicit omission field using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_submission(root, 1, True, {"errors": []})
            decision = inspect_task1_decision(1, root)

            with self.assertRaisesRegex(DataError, "missing omission"):
                convert_task1_and_evaluate(
                    decision, "System", root, inventory(), reference()
                )

    def test_opt_out_is_not_convertible(self) -> None:
        """Verify opt out is not convertible using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_submission(root, 1, "false", {"errors": [], "omission": None})
            decision = inspect_task1_decision(1, root)

        self.assertEqual(decision.status, "opted_out")

    def test_aggregate_digest_changes_with_prediction(self) -> None:
        """Verify aggregate digest changes with prediction using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_submission(root, 1, "true", {"errors": [], "omission": None})
            original = task1_input_digest([1], root)
            directory = next(root.iterdir())
            (directory / "predictions.jsonl").write_text("changed\n", encoding="utf-8")

            changed = task1_input_digest([1], root)

        self.assertNotEqual(original, changed)

    def test_selection_must_match_local_directories(self) -> None:
        """Verify selection must match local directories using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_submission(root, 1, "true", {"errors": [], "omission": None})
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "input_digest_algorithm": "test",
                        "input_sha256": task1_input_digest([1], root),
                        "submission_ids": [1],
                    }
                ),
                encoding="utf-8",
            )
            write_submission(root, 2, "false", {"errors": [], "omission": None})

            with self.assertRaisesRegex(DataError, r"unselected local IDs=\[2\]"):
                load_task1_submission_ids(selection, root)


if __name__ == "__main__":
    unittest.main()
