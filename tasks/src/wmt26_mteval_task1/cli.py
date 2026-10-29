"""Shared argument and selection helpers for repository scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from wmt26_mteval_task1.constants import MATCHING_ALGORITHMS
from wmt26_mteval_task1.models import Issue, SubmissionDescriptor
from wmt26_mteval_task1.submissions import (
    discover_baselines,
    discover_submissions,
    load_exclusions,
    merge_exclusions,
    select_baselines,
    select_submissions,
)


DEFAULT_TASK1_BASELINES_DIR = Path("baselines/task1_deduplicated")
DEFAULT_TASK1_GROUND_TRUTH = Path("human_annotations/ground-truth.jsonl")
DEFAULT_RUN_SPECIFIC_INFO = "default"
BASELINE_SNAPSHOTS = (
    "task1",
    "task1_deduplicated",
    "task1_prededup_and_enhy_fix",
)


def add_parquet_arguments(parser: argparse.ArgumentParser) -> None:
    """Add integrated-data and copied-card paths to a Task 1 command."""

    parser.add_argument(
        "--data-file", type=Path, default=Path("wmt26-grand.parquet"),
        help="integrated WMT26 parquet data file",
    )
    parser.add_argument(
        "--model-cards-dir", type=Path, default=Path("model_cards/task1"),
        help="root containing the copied Task 1 model cards",
    )
    parser.add_argument(
        "--baseline-snapshot", choices=BASELINE_SNAPSHOTS,
        default="task1_deduplicated",
        help="organizer baseline snapshot (default: task1_deduplicated)",
    )


def prepare_parquet_source(arguments: argparse.Namespace):
    """Select the integrated snapshot or retain explicit legacy test inputs."""

    from wmt26_mteval_task1.parquet_source import Task1ParquetSource

    old_root = getattr(arguments, "baselines_dir", DEFAULT_TASK1_BASELINES_DIR)
    if old_root != DEFAULT_TASK1_BASELINES_DIR:
        if old_root.name not in BASELINE_SNAPSHOTS:
            if old_root.is_dir():
                return None
            raise ValueError(f"{old_root}: unsupported baseline snapshot")
        if arguments.baseline_snapshot != "task1_deduplicated" and arguments.baseline_snapshot != old_root.name:
            raise ValueError("--baselines-dir and --baseline-snapshot select different snapshots")
        arguments.baseline_snapshot = old_root.name
    arguments.baselines_dir = Path("baselines") / arguments.baseline_snapshot
    return Task1ParquetSource(arguments.data_file, arguments.model_cards_dir)


def _run_specific_info(value: str) -> str:
    """Validate one portable, single-directory run namespace."""

    path = Path(value)
    if not value or value in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise argparse.ArgumentTypeError(
            "run-specific info must be one non-empty directory name"
        )
    return value


def add_run_specific_info_argument(parser: argparse.ArgumentParser) -> None:
    """Add the report namespace shared by Task 1 reporting commands."""

    parser.add_argument(
        "--run-specific-info",
        type=_run_specific_info,
        default=DEFAULT_RUN_SPECIFIC_INFO,
        help=(
            "single-directory report namespace below the command-specific "
            "output root (default: default)"
        ),
    )


def add_ground_truth_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared path to human ground-truth quality annotations."""

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_TASK1_GROUND_TRUTH,
        help="ground-truth human translation-quality annotations JSONL",
    )


def run_output_dir(root: Path, matching_algorithm: str, run_specific_info: str) -> Path:
    """Return the namespaced report/cache directory for one matching algorithm."""

    return root / matching_algorithm / run_specific_info


def add_mpp_cache_arguments(parser: argparse.ArgumentParser) -> None:
    """Add algorithm-cache-root and logging controls to a Task 1 command."""

    parser.add_argument(
        "--mpp-cache-root",
        type=Path,
        default=Path("outputs/task1"),
        help=(
            "root containing greedy/RUN_SPECIFIC_INFO/mpp_cache.sqlite3 and "
            "optimal/RUN_SPECIFIC_INFO/mpp_cache.sqlite3"
        ),
    )
    parser.add_argument(
        "--recompute-mpp",
        action="store_true",
        help="ignore valid cache entries and recompute MPP for every selected entry",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show per-entry and per-direction DEBUG progress messages",
    )


def add_matching_algorithm_argument(
    parser: argparse.ArgumentParser, *, allow_both: bool = False
) -> None:
    """Add a greedy/optimal matcher selector, optionally including both."""

    choices = (*MATCHING_ALGORITHMS, "both") if allow_both else MATCHING_ALGORITHMS
    parser.add_argument(
        "--matching-algorithm",
        choices=choices,
        default="greedy",
        help=(
            "MPP matching algorithm"
            + ("; both also writes a paired comparison report" if allow_both else "")
        ),
    )


def mpp_cache_path(
    root: Path, matching_algorithm: str, run_specific_info: str = "default"
) -> Path:
    """Return the enforced physical cache path for one matching algorithm."""

    if matching_algorithm not in MATCHING_ALGORITHMS:
        raise ValueError(f"unsupported MPP matching algorithm {matching_algorithm!r}")
    return (
        run_output_dir(root, matching_algorithm, run_specific_info)
        / "mpp_cache.sqlite3"
    )


def add_selection_arguments(
    parser: argparse.ArgumentParser,
    *,
    allow_empty_submissions: bool = False,
    allow_exclusions: bool = True,
) -> None:
    """Add participant selectors and optional exclusion controls to a parser."""

    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("submissions/task1"),
        help="directory containing extracted submission directories",
    )
    include = parser.add_mutually_exclusive_group()
    if allow_empty_submissions:
        include.add_argument(
            "--submission",
            action="extend",
            nargs="*",
            dest="submission_selectors",
            metavar="SUBMISSION",
            help=(
                "exact submission ID; repeat for a subset, "
                "or pass without values to select no submissions"
            ),
        )
    else:
        include.add_argument(
            "--submission",
            action="append",
            dest="submission_selectors",
            help="exact submission ID; repeat for a subset",
        )
    if allow_exclusions:
        parser.add_argument(
            "--exclude-submission",
            action="append",
            default=[],
            help="exact submission ID to omit",
        )
        parser.add_argument(
            "--exclude-file",
            type=Path,
            help="JSON object with an optional `submissions` ID array",
        )


def resolve_selection(
    arguments: argparse.Namespace,
) -> tuple[tuple[SubmissionDescriptor, ...], tuple[Issue, ...]]:
    """Discover and select participants, applying configured exclusions if present."""

    if arguments.submission_selectors == []:
        return (), ()
    discovered, discovery_issues = discover_submissions(arguments.submissions_dir)
    exclusions = merge_exclusions(
        load_exclusions(getattr(arguments, "exclude_file", None)),
        submissions=getattr(arguments, "exclude_submission", ()),
    )
    selected = select_submissions(
        discovered,
        submission_selectors=arguments.submission_selectors or (),
        exclusions=exclusions,
    )
    return selected, discovery_issues


def resolve_baseline_selection(
    arguments: argparse.Namespace,
) -> tuple[tuple[SubmissionDescriptor, ...], tuple[Issue, ...]]:
    """Discover all, some, or explicitly no organizer baselines."""

    if arguments.baseline_selectors == []:
        return (), ()
    discovered, discovery_issues = discover_baselines(arguments.baselines_dir)
    selected = select_baselines(
        discovered,
        baseline_selectors=arguments.baseline_selectors,
    )
    return selected, discovery_issues
