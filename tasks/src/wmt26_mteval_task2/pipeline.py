"""Shared input preparation for direct and automatically converted rankings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .conversion_labels import CONVERSION_DIRECTIONS, CONVERSION_ENTRIES
from .conversion_inputs import discover_conversion_inputs, load_hypothesis_lengths
from .domain_coverage import file_sha256, load_domain_index
from .io import discover_submissions
from .metadata import DIRECT_ENTRIES, EntryMetadata
from .models import HumanEvidence, SubmissionDescriptor
from .paper_labels import (
    PER_DIRECTION_DISPLAY_EXCLUSIONS, PLACEHOLDER_ENTRY_IDS,
    EXCLUDED_DIRECT_ENTRY_IDS, PRIMARY_DIRECTIONS,
)
from .predictions import _load_human_supported_predictions
from .preprocessing import aggregate_human_evidence, load_screens
from .sources import validate_sources


@dataclass
class Inputs:
    """Validated evidence, eligible predictions, and provenance for one track."""

    directions: tuple[str, ...]
    evidence: dict[str, HumanEvidence]
    predictions: dict[str, dict[str, dict]]
    metadata: dict[str, EntryMetadata]
    descriptors: dict[str, SubmissionDescriptor]
    eligibility: list[dict]
    exclusions: dict[str, str]
    source_lookup: dict[tuple[str, ...], int]
    hashes: dict[str, str]


def direct_entries(roots: list[Path], *, include_placeholder: bool = False):
    """Return direct inputs listed for the paper ranking.

    A diagnostic switch retains the constant-score file used only in the
    published screen/source validation table, not in the official ranking.
    """
    selected = []
    for entry in discover_submissions(roots):
        identity = entry.submission_id
        if identity in EXCLUDED_DIRECT_ENTRY_IDS:
            continue
        if not include_placeholder and identity in PLACEHOLDER_ENTRY_IDS:
            continue
        if identity not in DIRECT_ENTRIES:
            raise ValueError(f"Direct evaluator is not in the paper catalog: {identity}")
        selected.append(entry)
    if not selected:
        raise ValueError("No canonical direct Task 2 entries found")
    return selected


def load_inputs(test_set: Path, ground_truth: Path, roots: list[Path], *,
                track: str = "direct", scope: str = "all",
                task1_submissions=Path("submissions/task1"),
                task1_dedup=Path("baselines/task1_deduplicated"),
                task1_original=Path("baselines/task1_prededup_and_enhy_fix")) -> Inputs:
    """Load official human support and validate each evaluator independently.

    Direct local eligibility needs complete predictions only in that direction.
    Conversion rankings use the fixed list of original Task 1 inputs recorded
    in the evaluator catalog.
    The blindset determines exclusions by exact reference text, not system names.
    """
    if track not in {"direct", "conversion"} or scope not in {"all", "primary", "nonprimary"}:
        raise ValueError("Unknown track or direction scope")
    screens = load_screens(ground_truth)
    available = {screen.direction for screen in screens}
    if track == "conversion":
        directions = CONVERSION_DIRECTIONS
        entries = discover_conversion_inputs(task1_submissions, task1_dedup, task1_original)
        metadata = CONVERSION_ENTRIES
    else:
        directions = tuple(sorted(available)) if scope == "all" else (
            PRIMARY_DIRECTIONS if scope == "primary" else tuple(sorted(available - set(PRIMARY_DIRECTIONS))))
        entries = direct_entries(roots)
        metadata = DIRECT_ENTRIES
    if not directions or not set(directions) <= available:
        raise ValueError("Selected directions do not all have retained human evidence")
    index = load_domain_index(test_set, directions)
    evidence = {
        direction: aggregate_human_evidence(
            index.test_sets[direction], screens, direction,
            excluded_systems=(index.designated_references[direction],)
            if direction in index.designated_references else (),
        ) for direction in directions
    }
    requirements = {d: {item: set(values) for item, values in ev.segment_scores.items()}
                    for d, ev in evidence.items()}
    predictions = {d: {} for d in directions}
    eligibility = []
    hashes = {"test_set": file_sha256(test_set), "ground_truth": file_sha256(ground_truth)}
    lengths = load_hypothesis_lengths(test_set) if track == "conversion" else None
    for number, entry in enumerate(entries, 1):
        identity = entry.submission_id
        hashes[f"predictions/{identity}"] = file_sha256(entry.scores_path)
        outcomes = _load_human_supported_predictions(entry.scores_path, requirements, hypothesis_lengths=lengths)
        for direction, outcome in outcomes.items():
            excluded = track == "direct" and identity in PER_DIRECTION_DISPLAY_EXCLUSIONS.get(direction, ())
            status = "no_reference_configuration" if excluded else outcome.status
            eligibility.append(dict(direction=direction, submission_id=identity, status=status,
                                    required_scores=outcome.required_scores, message=outcome.message))
            if status == "valid":
                predictions[direction][identity] = outcome.predictions
            elif track == "conversion":
                raise ValueError(f"Incomplete conversion: {identity}/{direction}: {outcome.message}")
        print(f"Loaded {number}/{len(entries)}: {metadata[identity].submission_name} ({identity})", flush=True)
    # Primary runs share structural source occurrences across target languages.
    # Other direction runs construct their own index when preparing statistics.
    primary_items = {item for d, ev in evidence.items() if d in PRIMARY_DIRECTIONS
                     for item in ev.segment_scores}
    lookup, _ = validate_sources(test_set, primary_items) if primary_items else ({}, {})
    return Inputs(tuple(directions), evidence, predictions,
                  {e.submission_id: metadata[e.submission_id] for e in entries},
                  {e.submission_id: e for e in entries}, eligibility,
                  index.designated_references, lookup, hashes)


def aggregate_ids(inputs: Inputs) -> list[str]:
    """Return entries complete on all eight directions, never a shorter mean."""
    if not set(PRIMARY_DIRECTIONS) <= set(inputs.directions):
        return []
    return sorted(set.intersection(*(set(inputs.predictions[d]) for d in PRIMARY_DIRECTIONS)))
