"""Upstream analysis regression tests with release-package imports."""

from __future__ import annotations

import unittest

from wmt26_mteval_task3.threshold.core import (
    AnalysisResult,
    CurvePoint,
    Observation,
    downsample_curve,
    mcc_from_counts,
    optimize_threshold,
    optimize_per_language,
    pooled_metrics_from_official_rows,
)
from wmt26_mteval_task3.threshold.core import LanguageAnalysis
from wmt26_mteval_task3.threshold.cli import (PAPER_EXCLUDED_PARTICIPANT_IDS, PAPER_SYSTEM_GROUPS, PRIORITY_LANGUAGE_PAIRS)
from wmt26_mteval_task3.threshold.plotting import (
    paper_plot_legend,
    plottable_curve_points,
    select_language_pairs,
    select_results_by_id,
    short_pair_label,
)
from wmt26_mteval_task3.io import DataError


def observation(score: float, gold: bool, item: int) -> Observation:
    """Build or process the synthetic observation fixture for the supplied test inputs."""
    return Observation(score, gold, "eng_Latn---deu_Latn", str(item), "system")


class ThresholdSweepTest(unittest.TestCase):
    """Group original threshold sweep test regression scenarios."""
    def test_paper_system_selection_excludes_ineligible_participants(self) -> None:
        """Verify paper system selection excludes ineligible participants using synthetic upstream fixtures."""
        self.assertEqual(
            set(PAPER_EXCLUDED_PARTICIPANT_IDS),
            {"870257", "870554", "871298", "875949", "876231"},
        )
        participant_ids = next(
            entry_ids
            for _, track, entry_ids, _ in PAPER_SYSTEM_GROUPS
            if track == "participant"
        )
        self.assertEqual(participant_ids, ("867652", "872328", "876271"))
        self.assertTrue(
            set(participant_ids).isdisjoint(PAPER_EXCLUDED_PARTICIPANT_IDS)
        )
        baseline_ids = next(
            entry_ids
            for _, track, entry_ids, _ in PAPER_SYSTEM_GROUPS
            if track == "baseline"
        )
        self.assertEqual(
            baseline_ids,
            ("gemma4-reasoning", "gpt-5.6-sol-medium", "xcomet-xxl-no-reference"),
        )

    def test_uses_official_priority_language_pairs(self) -> None:
        """Verify uses official priority language pairs using synthetic upstream fixtures."""
        self.assertEqual(
            PRIORITY_LANGUAGE_PAIRS,
            (
                "ces_Latn---deu_Latn",
                "ces_Latn---vie_Latn",
                "eng_Latn---arz_Arab",
                "eng_Latn---ekk_Latn",
                "eng_Latn---ind_Latn",
                "eng_Latn---isl_Latn",
                "eng_Latn---sme_Latn",
                "zho_Hans---jpn_Jpan",
            ),
        )

    def test_finds_exact_boundary_without_splitting_ties(self) -> None:
        """Verify finds exact boundary without splitting ties using synthetic upstream fixtures."""
        observations = [
            observation(3.0, True, 1),
            observation(3.0, True, 2),
            observation(2.0, False, 3),
            observation(1.0, False, 4),
        ]

        result = optimize_threshold(observations, anchor_threshold=2.0)

        self.assertEqual(result.best.mcc, 1.0)
        self.assertGreater(result.best.threshold, 2.0)
        self.assertLessEqual(result.best.threshold, 3.0)
        self.assertEqual(
            (
                result.best.true_positive,
                result.best.false_positive,
                result.best.false_negative,
                result.best.true_negative,
            ),
            (2, 0, 0, 2),
        )

    def test_degenerate_mcc_matches_evaluation_convention(self) -> None:
        """Verify degenerate mcc matches evaluation convention using synthetic upstream fixtures."""
        self.assertEqual(mcc_from_counts(2, 2, 0, 0), 0.0)
        self.assertEqual(mcc_from_counts(0, 0, 2, 2), 0.0)

    def test_precision_and_recall_use_error_free_as_positive(self) -> None:
        """Verify precision and recall use error free as positive using synthetic upstream fixtures."""
        point = CurvePoint(0.5, 0.0, 3, 1, 2, 4)

        self.assertEqual(point.precision, 0.75)
        self.assertEqual(point.recall, 0.6)
        self.assertEqual(CurvePoint(1.0, 0.0, 0, 0, 5, 4).precision, 0.0)
        no_predictions = CurvePoint(1.0, 0.0, 0, 0, 5, 4)
        terminal = CurvePoint(0.9, -0.1, 0, 1, 5, 3)
        final_positive = CurvePoint(0.8, 0.1, 1, 1, 4, 3)
        predicted = CurvePoint(0.5, 0.2, 3, 1, 2, 4)
        self.assertEqual(
            plottable_curve_points(
                (no_predictions, terminal, final_positive, predicted)
            ),
            (final_positive, predicted),
        )

    def test_task1_reference_metrics_are_pooled_from_counts(self) -> None:
        """Verify task1 reference metrics are pooled from counts using synthetic upstream fixtures."""
        rows = [
            {"true_positive": "2", "false_positive": "1", "false_negative": "1", "true_negative": "2"},
            {"true_positive": "1", "false_positive": "0", "false_negative": "2", "true_negative": "3"},
        ]

        metrics = pooled_metrics_from_official_rows(rows)

        self.assertAlmostEqual(metrics["mcc"], mcc_from_counts(3, 1, 3, 5))
        self.assertEqual(metrics["precision"], 0.75)
        self.assertEqual(metrics["recall"], 0.5)

    def test_downsampling_preserves_precision_and_recall_extrema(self) -> None:
        """Verify downsampling preserves precision and recall extrema using synthetic upstream fixtures."""
        points = tuple(
            CurvePoint(float(index), 0.1, *counts)
            for index, counts in enumerate(
                (
                    (0, 0, 5, 5),
                    (1, 0, 4, 5),
                    (1, 4, 4, 1),
                    (4, 1, 1, 4),
                    (5, 0, 0, 5),
                    (5, 5, 0, 0),
                    (4, 0, 1, 5),
                    (0, 0, 5, 5),
                )
            )
        )

        sampled = downsample_curve(points, set(), limit=7)

        self.assertIn(points[0], sampled)
        self.assertIn(points[-1], sampled)
        self.assertEqual(max(point.precision for point in sampled), 1.0)
        self.assertEqual(max(point.recall for point in sampled), 1.0)

    def test_optimizes_each_language_before_macro_averaging(self) -> None:
        """Verify optimizes each language before macro averaging using synthetic upstream fixtures."""
        observations = [
            Observation(score, gold, language, str(index), "system")
            for language, values in (
                ("a---b", ((4.0, True), (3.0, True), (2.0, False), (1.0, False))),
                ("c---d", ((10.0, True), (9.0, True), (8.0, False), (7.0, False))),
            )
            for index, (score, gold) in enumerate(values)
        ]

        rows = optimize_per_language(
            observations,
            anchor_threshold=5.0,
            declared_predicate=lambda score: score >= 5.0,
            pooled_optimal_threshold=5.0,
        )

        self.assertEqual([row["language_pair"] for row in rows], ["a---b", "c---d"])
        self.assertEqual([row["language_optimal_mcc"] for row in rows], [1.0, 1.0])
        self.assertNotEqual(
            rows[0]["language_optimal_threshold"],
            rows[1]["language_optimal_threshold"],
        )

    def test_plot_language_selection_defaults_to_all_and_accepts_subset(self) -> None:
        """Verify plot language selection defaults to all and accepts subset using synthetic upstream fixtures."""
        result = AnalysisResult(
            row={},
            plot_points=(),
            declared_pair_counts={},
            languages=(
                LanguageAnalysis({"language_pair": "z---q"}, ()),
                LanguageAnalysis({"language_pair": "a---b"}, ()),
            ),
        )

        self.assertEqual(select_language_pairs([result], None), ["a---b", "z---q"])
        self.assertEqual(
            select_language_pairs([result], ["z---q", "z---q"]),
            ["z---q"],
        )
        with self.assertRaises(DataError):
            select_language_pairs([result], ["missing---pair"])

    def test_paper_plot_helpers_preserve_requested_system_order(self) -> None:
        """Verify paper plot helpers preserve requested system order using synthetic upstream fixtures."""
        results = [
            AnalysisResult(
                row={"track": "participant", "entry_id": entry_id},
                plot_points=(),
                declared_pair_counts={},
                languages=(),
            )
            for entry_id in ("second", "first")
        ]

        selected = select_results_by_id(
            results, "participant", ["first", "second"]
        )

        self.assertEqual(
            [result.row["entry_id"] for result in selected],
            ["first", "second"],
        )
        self.assertEqual(short_pair_label("eng_Latn---arz_Arab"), "eng-arz")
        with self.assertRaises(DataError):
            select_results_by_id(results, "baseline", ["first"])

        self.assertNotIn(
            "Task 1->3 MCC",
            [handle.get_label() for handle in paper_plot_legend(False)],
        )
        self.assertIn(
            "Task 1->3 MCC",
            [handle.get_label() for handle in paper_plot_legend(True)],
        )
        task1_handle = next(
            handle
            for handle in paper_plot_legend(True)
            if handle.get_label() == "Task 1->3 MCC"
        )
        self.assertEqual(task1_handle.get_linestyle(), ":")


if __name__ == "__main__":
    unittest.main()
