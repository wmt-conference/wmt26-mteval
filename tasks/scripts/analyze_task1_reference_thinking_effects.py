#!/usr/bin/env python3
"""Analyze matched reference-use and thinking effects in Task 1 baselines."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmt26_mteval_task1.cli import (
    DEFAULT_TASK1_BASELINES_DIR,
    add_parquet_arguments,
    add_ground_truth_argument,
    add_mpp_cache_arguments,
    add_run_specific_info_argument,
    mpp_cache_path,
    prepare_parquet_source,
    resolve_baseline_selection,
    run_output_dir,
)
from wmt26_mteval_task1.io import (
    load_model_card_template,
    load_ground_truth,
    load_test_set,
)
from wmt26_mteval_task1.mpp_cache import MppCache
from wmt26_mteval_task1.mpp_effect_analysis import (
    BASELINE_PAIR_NAMES,
    analyze_mpp_effects,
)
from wmt26_mteval_task1.mpp_effect_analysis_reporting import (
    write_paper_mpp_effect_reports,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the baseline-only matched MPP-effects command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_parquet_arguments(parser)
    add_mpp_cache_arguments(parser)
    parser.set_defaults(matching_algorithm="greedy")
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=DEFAULT_TASK1_BASELINES_DIR,
        help="directory containing deduplicated organizer baseline directories",
    )
    parser.add_argument(
        "--baseline",
        action="extend",
        nargs="*",
        dest="baseline_selectors",
        metavar="BASELINE",
        help="exact baseline directory basename; use only complete matched pairs",
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("test-set-files/mteval-test26.jsonl"),
        help="authoritative WMT26 test-set JSONL",
    )
    add_ground_truth_argument(parser)
    parser.add_argument(
        "--model-card-template",
        type=Path,
        help="optional legacy model-card validation template",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/task1/reference_thinking_effects"),
        help="paper reference/thinking report root",
    )
    add_run_specific_info_argument(parser)
    parser.add_argument(
        "--require-complete-ground-truth",
        action="store_true",
        help="reject ground-truth annotations lacking any policy-scorable official pair",
    )
    return parser


def _run_algorithm(
    arguments: argparse.Namespace,
    algorithm: str,
    baselines: tuple,
):
    """Score selected baselines and return the result after writing paper reports."""

    test_set = load_test_set(arguments.test_set)
    template = (
        load_model_card_template(arguments.model_card_template)
        if arguments.model_card_template is not None else None
    )
    ground_truth = load_ground_truth(
        arguments.ground_truth,
        test_set,
        require_complete=arguments.require_complete_ground_truth,
    )
    with MppCache(
        mpp_cache_path(
            arguments.mpp_cache_root, algorithm, arguments.run_specific_info
        ),
        test_set_path=arguments.test_set,
        ground_truth_path=arguments.ground_truth,
        matching_algorithm=algorithm,
    ) as cache:
        result = analyze_mpp_effects(
            baselines,
            test_set,
            ground_truth,
            template,
            mpp_cache=cache,
            recompute_mpp=arguments.recompute_mpp,
            matching_algorithm=algorithm,
        )
        logging.info(cache.summary())
        paths = write_paper_mpp_effect_reports(
            run_output_dir(arguments.output_dir, algorithm, arguments.run_specific_info),
            result,
        )
    for path in paths:
        logging.debug("Wrote %s", path)
    return result


def main(argv: list[str] | None = None) -> int:
    """Discover baselines, score matched pairs, and write effect reports."""

    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    source = None
    token = None
    try:
        source = prepare_parquet_source(arguments)
        token = source.install() if source is not None else None
        baselines, discovery_issues = resolve_baseline_selection(arguments)
        baselines = tuple(
            baseline
            for baseline in baselines
            if baseline.source_baseline_name in BASELINE_PAIR_NAMES
        )
        result = _run_algorithm(arguments, arguments.matching_algorithm, baselines)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("Reference/thinking effect analysis setup failed: %s", exc)
        return 2
    finally:
        if token is not None:
            source.reset(token)
    logging.info(
        "Wrote reference/thinking paper reports",
    )
    return (
        1
        if result.failed
        or any(issue.severity == "error" for issue in discovery_issues)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
