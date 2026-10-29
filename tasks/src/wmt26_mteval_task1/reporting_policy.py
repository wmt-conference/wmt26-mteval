"""Shared Task 1 policies for effective report metadata."""

from __future__ import annotations

import re

from wmt26_mteval_task1.models import SubmissionDescriptor


_XCOMET_BASELINE_PATTERN = re.compile(r"^xcomet(?:[_-]|$)", re.IGNORECASE)


def is_xcomet_baseline(descriptor: SubmissionDescriptor) -> bool:
    """Return whether an entry is an organizer-provided xCOMET baseline."""

    return descriptor.is_baseline and bool(
        _XCOMET_BASELINE_PATTERN.match(descriptor.baseline_name)
    )


def reported_fine_tuned(
    descriptor: SubmissionDescriptor, declared_value: bool | None
) -> bool | None:
    """Return the fine-tuning status used in generated Task 1 reports.

    Organizer xCOMET baselines are known fine-tuned systems, so reports treat
    them as fine-tuned even when their model cards omit or contradict that
    fact. All other entries retain the normalized model-card value.
    """

    if is_xcomet_baseline(descriptor):
        return True
    return declared_value
