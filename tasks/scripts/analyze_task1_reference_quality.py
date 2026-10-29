#!/usr/bin/env python3
"""Compare reference-based evaluators across RefQuality reference variants."""

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
    add_selection_arguments,
    mpp_cache_path,
    prepare_parquet_source,
    resolve_baseline_selection,
    resolve_selection,
    run_output_dir,
)
from wmt26_mteval_task1.io import (
    load_model_card_template,
    load_ground_truth,
    load_test_set,
)
from wmt26_mteval_task1.mpp_cache import MppCache
from wmt26_mteval_task1.reference_quality import (
    analyze_reference_quality,
    load_reference_quality_index,
)
from wmt26_mteval_task1.reference_quality_reporting import (
    write_paper_reference_quality_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the RefQuality analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_parquet_arguments(parser)
    add_selection_arguments(parser, allow_empty_submissions=True)
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
        help="authoritative WMT26 test-set JSONL containing RefQuality rows",
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
        default=Path("outputs/task1/reference_quality"),
        help="paper reference-quality figure root",
    )
    add_run_specific_info_argument(parser)
    parser.add_argument(
        "--require-complete-ground-truth",
        action="store_true",
        help=(
            "reject ground-truth annotations that do not cover every policy-scorable "
            "official item-system pair"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run cohort discovery, matched MPP scoring, and report generation."""

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
        submissions, submission_discovery_issues = resolve_selection(arguments)
        baselines, baseline_discovery_issues = resolve_baseline_selection(arguments)
        entries = submissions + baselines
        discovery_issues = submission_discovery_issues + baseline_discovery_issues
        test_set = load_test_set(arguments.test_set)
        reference_quality = load_reference_quality_index(arguments.test_set, test_set)
        template = (
            load_model_card_template(arguments.model_card_template)
            if arguments.model_card_template is not None else None
        )
        ground_truth = load_ground_truth(
            arguments.ground_truth,
            test_set,
            require_complete=arguments.require_complete_ground_truth,
        )
        logging.info(
            "Analyzing %d submissions and %d baselines across %d RefQuality rows",
            len(submissions),
            len(baselines),
            reference_quality.refquality_row_count,
        )
        with MppCache(
            mpp_cache_path(
                arguments.mpp_cache_root,
                arguments.matching_algorithm,
                arguments.run_specific_info,
            ),
            test_set_path=arguments.test_set,
                ground_truth_path=arguments.ground_truth,
            matching_algorithm=arguments.matching_algorithm,
        ) as mpp_cache:
            result = analyze_reference_quality(
                entries,
                test_set,
                ground_truth,
                template,
                reference_quality,
                mpp_cache=mpp_cache,
                recompute_mpp=arguments.recompute_mpp,
                matching_algorithm=arguments.matching_algorithm,
            )
            logging.info(mpp_cache.summary())
        paper_path = write_paper_reference_quality_report(
            run_output_dir(
                arguments.output_dir,
                arguments.matching_algorithm,
                arguments.run_specific_info,
            ),
            result,
        )
        paths = (paper_path,)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("Batch setup failed: %s", exc)
        return 2
    finally:
        if token is not None:
            source.reset(token)

    for path in paths:
        logging.debug("Wrote %s", path)
    logging.info(
        "Wrote %d reference-quality reports to %s",
        len(paths),
        run_output_dir(
            arguments.output_dir,
            arguments.matching_algorithm,
            arguments.run_specific_info,
        ),
    )
    return (
        1
        if result.failed or any(issue.severity == "error" for issue in discovery_issues)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
