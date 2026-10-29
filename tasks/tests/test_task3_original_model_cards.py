"""Upstream analysis regression tests with release-package imports."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from wmt26_mteval_task3.model_cards import (
    CLOSED_SOURCE_MODEL_QUESTION,
    SYSTEM_LINK_QUESTION,
    declared_base_model,
    load_base_model_overrides,
    resolved_base_model,
)
from wmt26_mteval_task3.io import DataError


class DeclaredBaseModelTest(unittest.TestCase):
    """Group original declared base model test regression scenarios."""
    def test_prefers_explicit_closed_source_model(self) -> None:
        """Verify prefers explicit closed source model using synthetic upstream fixtures."""
        answers = {
            CLOSED_SOURCE_MODEL_QUESTION: " model-v1 ",
            SYSTEM_LINK_QUESTION: "https://huggingface.co/owner/fallback",
        }

        self.assertEqual(declared_base_model(answers), "model-v1")

    def test_extracts_hugging_face_repository(self) -> None:
        """Verify extracts hugging face repository using synthetic upstream fixtures."""
        answers = {
            CLOSED_SOURCE_MODEL_QUESTION: "",
            SYSTEM_LINK_QUESTION: "https://huggingface.co/owner/model/tree/main",
        }

        self.assertEqual(declared_base_model(answers), "owner/model")

    def test_reviewed_override_takes_precedence(self) -> None:
        """Verify reviewed override takes precedence using synthetic upstream fixtures."""
        answers = {CLOSED_SOURCE_MODEL_QUESTION: "declared-model"}

        self.assertEqual(
            resolved_base_model(answers, 123, {123: "reviewed-model"}),
            "reviewed-model",
        )

    def test_loads_strict_id_keyed_overrides(self) -> None:
        """Verify loads strict id keyed overrides using synthetic upstream fixtures."""
        value = {
            "schema_version": 1,
            "task2_participant": {"123": " model-a "},
            "task3_direct": {"456": "model-b"},
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "overrides.json"
            path.write_text(json.dumps(value), encoding="utf-8")

            overrides = load_base_model_overrides(path)

        self.assertEqual(overrides["task2_participant"], {123: "model-a"})
        self.assertEqual(overrides["task3_direct"], {456: "model-b"})

    def test_rejects_invalid_override_id(self) -> None:
        """Verify rejects invalid override id using synthetic upstream fixtures."""
        value = {
            "schema_version": 1,
            "task2_participant": {"0123": "model-a"},
            "task3_direct": {},
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "overrides.json"
            path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaises(DataError):
                load_base_model_overrides(path)


if __name__ == "__main__":
    unittest.main()
