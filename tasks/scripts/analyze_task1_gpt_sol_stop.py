#!/usr/bin/env python3
"""Compare GPT Sol Medium and xhigh using pooled micro MPP on shared-stop gold pairs."""

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
    add_run_specific_info_argument,
    prepare_parquet_source,
    resolve_baseline_selection,
    run_output_dir,
)
from wmt26_mteval_task1.gpt_sol_stop import analyze_gpt_sol_stop
from wmt26_mteval_task1.gpt_sol_stop_reporting import write_paper_gpt_sol_stop_report
from wmt26_mteval_task1.io import (
    load_model_card_template,
    load_ground_truth,
    load_test_set,
)


def build_parser() -> argparse.ArgumentParser:
    """Expose shared input, matcher, and report namespace controls for the two-baseline analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_parquet_arguments(parser)
    for option, default in {
        "baselines-dir": DEFAULT_TASK1_BASELINES_DIR,
        "test-set": "test-set-files/mteval-test26.jsonl",
        "output-dir": "outputs/task1/gpt_sol_stop",
    }.items():
        parser.add_argument(f"--{option}", type=Path, default=Path(default))
    parser.add_argument("--model-card-template", type=Path)
    add_ground_truth_argument(parser)
    parser.set_defaults(baseline_selectors=None)
    parser.set_defaults(matching_algorithm="greedy")
    add_run_specific_info_argument(parser)
    parser.add_argument("--require-complete-ground-truth", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load inputs and write the medium-versus-xhigh paper table."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    source = None
    token = None
    try:
        source = prepare_parquet_source(args)
        token = source.install() if source is not None else None
        baselines, issues = resolve_baseline_selection(args)
        for issue in issues:
            logging.warning(issue.message)
        test_set = load_test_set(args.test_set)
        ground_truth = load_ground_truth(
            args.ground_truth, test_set, require_complete=args.require_complete_ground_truth
        )
        template = (
            load_model_card_template(args.model_card_template)
            if args.model_card_template is not None else None
        )
        result = analyze_gpt_sol_stop(
            baselines, test_set, ground_truth, template,
            matching_algorithm=args.matching_algorithm,
        )
        output = run_output_dir(
            args.output_dir, args.matching_algorithm, args.run_specific_info,
        )
        write_paper_gpt_sol_stop_report(output, result)
        logging.info("Wrote %s: %d shared pairs", output, len(result.pairs))
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("GPT Sol shared-stop analysis failed: %s", exc)
        return 2
    finally:
        if token is not None:
            source.reset(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
