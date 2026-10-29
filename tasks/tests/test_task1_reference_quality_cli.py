"""CLI contract tests for the RefQuality analysis script."""

from __future__ import annotations

from scripts.analyze_task1_reference_quality import build_parser


def test_reference_quality_cli_uses_shared_cache_root_and_report_defaults():
    """Keep reports separate while sharing algorithm-specific MPP caches."""

    arguments = build_parser().parse_args([])

    assert str(arguments.mpp_cache_root) == "outputs/task1"
    assert str(arguments.output_dir) == "outputs/task1/reference_quality"
    assert arguments.matching_algorithm == "greedy"
    assert arguments.submission_selectors is None
    assert arguments.baseline_selectors is None


def test_reference_quality_cli_accepts_empty_and_subset_selectors():
    """Mirror the official scorer's all, none, and explicit selection behavior."""

    empty = build_parser().parse_args(["--submission", "--baseline"])
    selected = build_parser().parse_args(
        ["--submission", "123", "--baseline", "gemma_ref_task1"]
    )

    assert empty.submission_selectors == []
    assert empty.baseline_selectors == []
    assert selected.submission_selectors == ["123"]
    assert selected.baseline_selectors == ["gemma_ref_task1"]
