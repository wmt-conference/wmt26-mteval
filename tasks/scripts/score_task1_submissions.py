#!/usr/bin/env python3
"""Batch-score WMT26 Task 1 submissions with MPP."""

from __future__ import annotations

import argparse
import logging
import sqlite3
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
    add_selection_arguments,
    mpp_cache_path,
    prepare_parquet_source,
    resolve_baseline_selection,
    resolve_selection,
    run_output_dir,
)
from wmt26_mteval_task1.mpp_cache import MppCache
from wmt26_mteval_task1.models import BatchScoreResult
from wmt26_mteval_task1.significance import (
    SignificanceConfig,
    SignificanceResult,
    compute_significance,
)
from wmt26_mteval_task1.io import (
    load_model_card_template,
    load_ground_truth,
    load_test_set,
)
from wmt26_mteval_task1.reporting import write_paper_score_reports
from wmt26_mteval_task1.scoring import score_submissions


def build_parser() -> argparse.ArgumentParser:
    """Build the official Task 1 scoring command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_parquet_arguments(parser)
    add_selection_arguments(
        parser, allow_empty_submissions=True, allow_exclusions=False
    )
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
        help=(
            "exact baseline directory basename; repeat for a subset, or pass "
            "without values to select no baselines"
        ),
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
        default=Path("outputs/task1/scores"),
        help="paper ranking report root",
    )
    add_run_specific_info_argument(parser)
    parser.add_argument(
        "--direction",
        action="append",
        dest="directions",
        metavar="LANGUAGE_PAIR",
        help=(
            "translation direction to score; repeat for a subset (default: all "
            "official directions)"
        ),
    )
    parser.add_argument(
        "--no-significance", action="store_true",
        help="skip permutations and leave ranking clusters blank",
    )
    parser.add_argument(
        "--significance-permutations", type=int, default=10000,
        help="number of fixed permutation draws (default: 10000)",
    )
    parser.add_argument(
        "--significance-workers", type=int, default=1,
        help="parallel evaluator-pair processes (default: 1; each worker uses one numerical thread)",
    )
    parser.add_argument(
        "--significance-seed", type=int, default=4,
        help="nonnegative permutation seed (default: 4)",
    )
    parser.add_argument(
        "--significance-alpha", type=float, default=0.05,
        help="unadjusted significance threshold (default: 0.05)",
    )
    parser.add_argument(
        "--significance-alternative", choices=("one-sided", "two-sided"),
        default="two-sided", help="permutation test alternative (default: two-sided)",
    )
    parser.add_argument(
        "--recompute-significance", action="store_true",
        help="refresh selected permutation streams while reusing MPP components",
    )
    return parser


def significance_for_result(
    arguments: argparse.Namespace, result: BatchScoreResult,
) -> SignificanceResult | None:
    """Compute one matching algorithm's ranking tests unless explicitly disabled."""
    if arguments.no_significance:
        return None
    cache_path = mpp_cache_path(
        arguments.mpp_cache_root, result.matching_algorithm, arguments.run_specific_info,
    ).with_name("significance_cache.sqlite3")
    return compute_significance(
        result,
        SignificanceConfig(
            arguments.significance_permutations,
            arguments.significance_seed,
            arguments.significance_alpha,
            workers=arguments.significance_workers,
            alternative=arguments.significance_alternative,
        ),
        cache_path,
        test_set_path=arguments.test_set,
        ground_truth_path=arguments.ground_truth,
        recompute=arguments.recompute_significance or arguments.recompute_mpp,
        paper_only=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Score the selected cohort and write algorithm-specific reports."""

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
        SignificanceConfig(
            arguments.significance_permutations,
            arguments.significance_seed,
            arguments.significance_alpha,
            workers=arguments.significance_workers,
            alternative=arguments.significance_alternative,
        )
        submissions, submission_discovery_issues = resolve_selection(arguments)
        baselines, baseline_discovery_issues = resolve_baseline_selection(arguments)
        scoring_inputs = submissions + baselines
        discovery_issues = submission_discovery_issues + baseline_discovery_issues
        test_set = load_test_set(arguments.test_set)
        template = (
            load_model_card_template(arguments.model_card_template)
            if arguments.model_card_template is not None else None
        )
        ground_truth = load_ground_truth(arguments.ground_truth, test_set)
        if ground_truth.zero_length_span_count:
            logging.warning(
                "Ignored %d zero-length reference spans",
                ground_truth.zero_length_span_count,
            )
        logging.info(
            "Scoring %d submissions and %d baselines",
            len(submissions),
            len(baselines),
        )
        with MppCache(
            mpp_cache_path(
                arguments.mpp_cache_root, arguments.matching_algorithm,
                arguments.run_specific_info,
            ),
            test_set_path=arguments.test_set,
            ground_truth_path=arguments.ground_truth,
            matching_algorithm=arguments.matching_algorithm,
        ) as cache:
            result = score_submissions(
                scoring_inputs, test_set, ground_truth, template,
                require_complete_predictions=True,
                mpp_cache=cache,
                recompute_mpp=arguments.recompute_mpp,
                matching_algorithm=arguments.matching_algorithm,
                directions=arguments.directions,
                collect_significance_inputs=not arguments.no_significance,
            )
            significance = significance_for_result(arguments, result)
            logging.info(cache.summary())
            paths = write_paper_score_reports(
                run_output_dir(
                    arguments.output_dir, arguments.matching_algorithm,
                    arguments.run_specific_info,
                ),
                result, significance,
            )
    except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as exc:
        logging.error("Batch setup failed: %s", exc)
        return 2
    finally:
        if token is not None:
            source.reset(token)

    for path in paths:
        logging.debug("Wrote %s", path)
    logging.info("Wrote %d scoring reports to %s", len(paths), arguments.output_dir)
    return (
        1
        if (
            result.failed
            or any(issue.severity == "error" for issue in discovery_issues)
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
