"""Upstream analysis regression tests with release-package imports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wmt26_mteval_task3.io import DataError, iter_jsonl


class JsonlIterationTest(unittest.TestCase):
    """Group original jsonl iteration test regression scenarios."""
    def test_streams_strict_jsonl_records(self) -> None:
        """Verify streams strict jsonl records using synthetic upstream fixtures."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "records.jsonl"
            path.write_text('{"value":1}\r\n{"value":2}', encoding="utf-8")

            self.assertEqual(
                list(iter_jsonl(path)),
                [(1, {"value": 1}), (2, {"value": 2})],
            )

    def test_rejects_blank_lines_and_duplicate_keys(self) -> None:
        """Verify rejects blank lines and duplicate keys using synthetic upstream fixtures."""
        cases = [
            ('{"value":1}\n\n', "blank line"),
            ('{"value":1,"value":2}\n', "duplicate JSON key"),
        ]
        for content, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary_dir:
                    path = Path(temporary_dir) / "records.jsonl"
                    path.write_text(content, encoding="utf-8")

                    with self.assertRaisesRegex(DataError, message):
                        list(iter_jsonl(path))


if __name__ == "__main__":
    unittest.main()
