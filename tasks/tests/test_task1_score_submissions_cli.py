from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze_task1_gpt_sol_stop import build_parser as build_gpt_sol_parser
from scripts.analyze_task1_reference_quality import build_parser as build_reference_quality_parser
from scripts.analyze_task1_reference_thinking_effects import build_parser as build_reference_thinking_parser
from scripts.analyze_task1_score_statistics import build_parser as build_score_statistics_parser
from scripts.score_task1_submissions import build_parser
from wmt26_mteval_task1.cli import mpp_cache_path, run_output_dir


def test_category_selectors_accept_explicit_empty_sets():
    arguments = build_parser().parse_args(["--submission", "--baseline"])

    assert arguments.submission_selectors == []
    assert arguments.baseline_selectors == []


def test_category_selectors_accept_explicit_subsets():
    arguments = build_parser().parse_args(
        [
            "--submission",
            "12",
            "--submission",
            "34",
            "--baseline",
            "Gemini",
            "--baseline",
            "Qwen_ref",
        ]
    )

    assert arguments.submission_selectors == [
        "12",
        "34",
    ]
    assert arguments.baseline_selectors == ["Gemini", "Qwen_ref"]


def test_omitted_category_selectors_retain_select_all_semantics():
    arguments = build_parser().parse_args([])

    assert arguments.submission_selectors is None
    assert arguments.baseline_selectors is None


def test_default_paths_are_task1_scoped():
    """Keep the official scorer pointed at the Task 1 data and output roots."""

    arguments = build_parser().parse_args([])

    assert str(arguments.submissions_dir) == "submissions/task1"
    assert str(arguments.baselines_dir) == "baselines/task1_deduplicated"
    assert str(arguments.mpp_cache_root) == "outputs/task1"
    assert str(arguments.output_dir) == "outputs/task1/scores"
    assert arguments.ground_truth == Path("human_annotations/ground-truth.jsonl")
    assert arguments.run_specific_info == "default"
    assert arguments.data_file == Path("wmt26-grand.parquet")
    assert arguments.model_cards_dir == Path("model_cards/task1")
    assert arguments.baseline_snapshot == "task1_deduplicated"


def test_all_paper_scripts_share_no_argument_baseline_and_namespace_defaults():
    """Keep every paper entry point on the same baseline snapshot and run label."""

    for parser_factory in (
        build_parser,
        build_reference_thinking_parser,
        build_reference_quality_parser,
        build_score_statistics_parser,
        build_gpt_sol_parser,
    ):
        arguments = parser_factory().parse_args([])
        assert arguments.baselines_dir == Path("baselines/task1_deduplicated")
        assert arguments.ground_truth == Path("human_annotations/ground-truth.jsonl")
        assert arguments.run_specific_info == "default"
        assert arguments.matching_algorithm == "greedy"


def test_removed_scorer_options_are_rejected():
    """Reject CLI switches removed from the paper scorer."""

    for option in (
        "--require-complete-references",
        "--allow-partial-predictions",
        "--exclude-file",
        "--exclude-submission",
        "--exclude-submissions",
        "--references",
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([option])


def test_ground_truth_override_is_shared_by_all_paper_scripts():
    """Accept one consistently named ground-truth input switch."""

    for parser_factory in (
        build_parser,
        build_reference_thinking_parser,
        build_reference_quality_parser,
        build_score_statistics_parser,
        build_gpt_sol_parser,
    ):
        arguments = parser_factory().parse_args(
            ["--ground-truth", "custom-ground-truth.jsonl"]
        )
        assert arguments.ground_truth == Path("custom-ground-truth.jsonl")


def test_shared_mpp_cache_options():
    arguments = build_parser().parse_args(
        ["--mpp-cache-root", "custom-cache", "--recompute-mpp", "--verbose"]
    )

    assert str(arguments.mpp_cache_root) == "custom-cache"
    assert arguments.recompute_mpp
    assert arguments.verbose


def test_run_specific_info_namespaces_reports_and_caches():
    """Place one named scoring run below its matching-algorithm directory."""

    arguments = build_parser().parse_args(["--run-specific-info", "release_candidate"])

    assert arguments.run_specific_info == "release_candidate"
    assert run_output_dir(
        arguments.output_dir, "greedy", arguments.run_specific_info
    ) == (arguments.output_dir / "greedy" / "release_candidate")
    assert mpp_cache_path(
        arguments.mpp_cache_root, "greedy", arguments.run_specific_info
    ) == Path("outputs/task1/greedy/release_candidate/mpp_cache.sqlite3")


def test_matching_algorithm_and_direction_arguments():
    arguments = build_parser().parse_args(
        [
            "--direction",
            "eng_Latn-deu_Latn",
            "--direction",
            "zho_Hans-jpn_Jpan",
        ]
    )

    assert arguments.matching_algorithm == "greedy"
    assert arguments.directions == [
        "eng_Latn-deu_Latn",
        "zho_Hans-jpn_Jpan",
    ]
