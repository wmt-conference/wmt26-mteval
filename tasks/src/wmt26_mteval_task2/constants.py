"""Shared constants for WMT26 Task 2 preprocessing and scoring."""

from __future__ import annotations


ITEM_ID_SEPARATOR = "_###_"
OFFICIAL_SET_ID = "official"

# These are the directions whose human evaluation was complete in the
# 2026-08-19 Pearmut snapshot.  The converter accepts explicit overrides so a
# later snapshot can be processed without changing code.
COMPLETE_HUMAN_EVALUATION_DIRECTIONS = frozenset(
    {
        "eng_Latn-ind_Latn",
        "eng_Latn-tha_Thai",
        "zho_Hans-jpn_Jpan",
    }
)

# Each value is an official candidate-system label whose translation is also
# the designated reference for every item in that direction. These candidates
# were present in the blindset shown to Task 2 evaluation systems. The optional
# exclusion mode validates their submitted scores normally, then removes the
# candidates from both human evidence and meta-metric computation.
DESIGNATED_REFERENCE_CANDIDATE_SYSTEMS = {
    "eng_Latn-ind_Latn": "PseudoRef",
    "eng_Latn-tha_Thai": "PseudoRef",
    "zho_Hans-jpn_Jpan": "Human (postediting)",
}

GROUP_BY_SCREEN_PDP = "group-by-screen-pdp"
GROUP_BY_SEGMENT_PDP = "group-by-segment-pdp"
SCREEN_BASED_SPA = "screen-based-spa"
SEGMENT_BASED_SPA = "segment-based-spa"

MEASURE_NAMES = (
    GROUP_BY_SCREEN_PDP,
    GROUP_BY_SEGMENT_PDP,
    SCREEN_BASED_SPA,
    SEGMENT_BASED_SPA,
)

# SPA itself uses an inner paired permutation test for every system pair.
# This is separate from the optional between-metric significance test (k in
# mt-metrics-eval), which WMT26 Task 2 deliberately does not run yet.
DEFAULT_SPA_PERMUTATIONS = 1_000
DEFAULT_SPA_SEED = 4
