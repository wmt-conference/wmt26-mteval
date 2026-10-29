"""Convert Task 1 error spans into Task 2 holistic scores."""

from collections.abc import Mapping


def task1_to_task2_score(assessment: Mapping, translation_length: int) -> float:
    """Convert raw Task 1 errors exactly as convert_task1_to_task2.py does.

    In particular, preserve its inclusive whole-span check (end == length - 1),
    empty-target exception, omission counts, and counting of zero-length spans.
    Task 1 span normalization must not change the input to this calculation.
    """

    major = minor = 0
    whole = False
    for error in assessment["errors"]:
        severity = error["severity"]
        if severity == "major":
            major += 1
            whole |= error["start"] == 0 and error["end"] == translation_length - 1
        elif severity == "minor":
            minor += 1
        else:
            raise ValueError(f"Unrecognized error severity {severity!r}")
    omission = assessment.get("omission")
    if omission:
        if omission == "major":
            major += 1
        elif omission == "minor":
            minor += 1
        else:
            raise ValueError(f"Unrecognized omission severity {omission!r}")
    empty = translation_length == 0 and major == 1 and minor == 0
    return 0.0 if whole or empty else max(0.0, 100.0 - 10 * major - 2.5 * minor)
