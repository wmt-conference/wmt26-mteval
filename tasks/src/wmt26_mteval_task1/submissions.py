"""Submission-directory discovery, filtering, and exclusion handling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from wmt26_mteval_task1.models import Issue, SubmissionDescriptor


SUBMISSION_DIRECTORY_PATTERN = re.compile(r"^(?P<submission_id>\d+)$")


def baseline_family_for_root(root: Path) -> str:
    """Return the known snapshot family represented by a baseline root.

    Unknown caller-provided roots retain their directory basenames as their
    identities, preserving compatibility for external or temporary cohorts.
    """

    return {
        "task1": "normal",
        "task1_deduplicated": "deduplicated",
        "task1_prededup_and_enhy_fix": "pre_dedup_pre_enhy_fix",
    }.get(root.name, "")


@dataclass(frozen=True)
class Exclusions:
    """Exact submission IDs omitted from a run."""

    submissions: frozenset[str] = frozenset()


def discover_submissions(
    root: Path,
) -> tuple[tuple[SubmissionDescriptor, ...], tuple[Issue, ...]]:
    """Discover parseable submission directories without checking their files."""

    from wmt26_mteval_task1.parquet_source import current_source

    source = current_source()
    if source is not None:
        entries = source.descriptors(root, baseline=False)
        if entries is not None:
            return entries, ()
    if not root.is_dir():
        raise ValueError(f"{root}: submissions directory is missing or not a directory")
    submissions: list[SubmissionDescriptor] = []
    seen_ids: set[str] = set()
    issues: list[Issue] = []
    for path in sorted(root.iterdir(), key=lambda candidate: candidate.name):
        if not path.is_dir():
            continue
        match = SUBMISSION_DIRECTORY_PATTERN.fullmatch(path.name)
        if match is None:
            issues.append(
                Issue(
                    "discovery",
                    f"{path}: directory name does not match the submission format",
                    path=str(path),
                )
            )
            continue
        submission_id = match.group("submission_id")
        if submission_id in seen_ids:
            raise ValueError(f"{root}: duplicate submission ID {submission_id}")
        seen_ids.add(submission_id)
        submissions.append(
            SubmissionDescriptor(
                path=path,
                submission_id=submission_id,
                username="",
                phase_id="",
                suffix="",
            )
        )
    submissions.sort(key=lambda submission: int(submission.submission_id))
    return tuple(submissions), tuple(issues)


def discover_baselines(
    root: Path,
) -> tuple[tuple[SubmissionDescriptor, ...], tuple[Issue, ...]]:
    """Discover baseline directories, whose names are their stable identities."""

    from wmt26_mteval_task1.parquet_source import current_source

    source = current_source()
    if source is not None:
        entries = source.descriptors(root, baseline=True)
        if entries is not None:
            return entries, ()
    if not root.is_dir():
        raise ValueError(f"{root}: baselines directory is missing or not a directory")
    family = baseline_family_for_root(root)
    baselines = tuple(
        SubmissionDescriptor(
            path=path,
            submission_id="",
            username="baseline",
            phase_id="",
            suffix="",
            is_baseline=True,
            baseline_family=family,
        )
        for path in sorted(
            (candidate for candidate in root.iterdir() if candidate.is_dir()),
            key=lambda candidate: (candidate.name.casefold(), candidate.name),
        )
    )
    return baselines, ()


def select_baselines(
    baselines: Iterable[SubmissionDescriptor],
    *,
    baseline_selectors: Iterable[str] | None = None,
) -> tuple[SubmissionDescriptor, ...]:
    """Select all, some, or explicitly no baselines by physical or source name."""

    descriptors = tuple(baselines)
    if baseline_selectors is None:
        return descriptors
    selected_names = frozenset(baseline_selectors)
    if not selected_names:
        return ()
    selected = tuple(
        baseline
        for baseline in descriptors
        if selected_names
        & {
            baseline.path.name,
            baseline.baseline_name,
            baseline.source_baseline_name,
        }
    )
    if not selected:
        raise ValueError("No baselines matched the requested selection")
    return selected


def load_exclusions(path: Path | None) -> Exclusions:
    """Load optional JSON exclusions with a `submissions` ID array."""

    if path is None:
        return Exclusions()
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"{path}: exclusion file is missing") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: exclusion file must be a JSON object")
    unexpected = set(value) - {"submissions"}
    if unexpected:
        raise ValueError(
            f"{path}: unexpected exclusion keys: {', '.join(sorted(unexpected))}"
        )

    def string_set(key: str) -> frozenset[str]:
        """Validate one exclusion-file field and return its string values."""

        entries = value.get(key, [])
        if not isinstance(entries, list) or any(
            not isinstance(entry, str) or not entry for entry in entries
        ):
            raise ValueError(f"{path}: {key} must be a list of non-empty strings")
        return frozenset(entries)

    return Exclusions(string_set("submissions"))


def merge_exclusions(
    file_exclusions: Exclusions,
    *,
    submissions: Iterable[str] = (),
) -> Exclusions:
    """Combine file-based and command-line exclusion selectors."""

    return Exclusions(
        file_exclusions.submissions | frozenset(submissions),
    )


def select_submissions(
    submissions: Iterable[SubmissionDescriptor],
    *,
    submission_selectors: Iterable[str] = (),
    exclusions: Exclusions | None = None,
) -> tuple[SubmissionDescriptor, ...]:
    """Select one/subset/all submissions by unique ID, then exclude."""

    descriptors = tuple(submissions)
    selected_ids = frozenset(str(value) for value in submission_selectors)
    exclusions = exclusions or Exclusions()

    selected: list[SubmissionDescriptor] = []
    for submission in descriptors:
        if selected_ids and submission.submission_id not in selected_ids:
            continue
        if submission.submission_id in exclusions.submissions:
            continue
        selected.append(submission)

    if not selected:
        raise ValueError("No submissions matched the requested selection")
    return tuple(sorted(selected, key=lambda submission: int(submission.submission_id)))
