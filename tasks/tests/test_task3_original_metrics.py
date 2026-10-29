"""Upstream analysis regression tests with release-package imports."""

from __future__ import annotations

import unittest

from wmt26_mteval_task3.metrics import PairObservations


class PairObservationsTest(unittest.TestCase):
    """Group original pair observations test regression scenarios."""
    def test_metrics_and_counts(self) -> None:
        """Verify metrics and counts using synthetic upstream fixtures."""
        observations = PairObservations()
        observations.add("item-1", True, True)
        observations.add("item-1", True, False)
        observations.add("item-2", False, True)
        observations.add("item-2", False, False)

        metrics = observations.metrics()

        self.assertEqual(metrics["annotated_items"], 2)
        self.assertEqual(metrics["annotated_labels"], 4)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["mcc"], 0.0)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_degenerate_mcc_is_zero(self) -> None:
        """Verify degenerate mcc is zero using synthetic upstream fixtures."""
        observations = PairObservations()
        observations.add("item-1", False, False)
        observations.add("item-2", False, False)

        self.assertEqual(observations.metrics()["mcc"], 0.0)


if __name__ == "__main__":
    unittest.main()
