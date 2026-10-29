"""Matched reference-input sensitivity analysis for WMT26 Task 2."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wmt26_mteval_task2.constants import (
    DEFAULT_SPA_PERMUTATIONS,
    DEFAULT_SPA_SEED,
    GROUP_BY_SCREEN_PDP,
    SEGMENT_BASED_SPA,
)
from wmt26_mteval_task2.metrics import prepare_spa_reference, score_measure
from wmt26_mteval_task2.models import HumanEvidence, Screen, SubmissionDescriptor
from wmt26_mteval_task2.preprocessing import aggregate_human_evidence
from wmt26_mteval_task2.metadata import EntryMetadata


MAIN_VS_PSEUDO = "main_vs_pseudo"
FROM_SCRATCH_VS_POSTEDIT = "from_scratch_vs_postedit"
COMPARISONS = (MAIN_VS_PSEUDO, FROM_SCRATCH_VS_POSTEDIT)
REFERENCE_SYSTEM_BY_TYPE = {
    "human": "Human (from scratch)",
    "postedit": "Human (postediting)",
    "pseudo": "PseudoRef",
}
REFERENCE_CANDIDATE_SYSTEMS = frozenset(REFERENCE_SYSTEM_BY_TYPE.values())
ITEM_SEPARATOR = "_###_"


@dataclass(frozen=True)
class ReferenceMirror:
    """One official item and an otherwise identical alternate-reference row."""

    direction: str
    official_item_id: str
    alternate_item_id: str
    left_reference_type: str
    right_reference_type: str
    left_reference_system: str
    right_reference_system: str
    official_systems: tuple[str, ...]
    alternate_systems: tuple[str, ...]
    common_systems: tuple[str, ...]
    omitted_systems: tuple[str, ...]
    reference_text_changed: bool


@dataclass(frozen=True)
class ReferenceComparisonDirection:
    """A complete direction for one paired reference-input comparison."""

    comparison: str
    direction: str
    left_label: str
    right_label: str
    mirrors: tuple[ReferenceMirror, ...]

    @property
    def item_count(self) -> int:
        """Return the number of matched source items."""

        return len(self.mirrors)

    @property
    def changed_reference_count(self) -> int:
        """Return the number of items whose reference strings differ."""

        return sum(mirror.reference_text_changed for mirror in self.mirrors)


@dataclass(frozen=True)
class ReferenceQualityIndex:
    """Validated RefQuality mirrors grouped by comparison and direction."""

    path: Path
    directions: Mapping[str, Mapping[str, ReferenceComparisonDirection]]
    alternate_row_count: int


@dataclass(frozen=True)
class PredictionInventory:
    """Validated relevant Task 2 predictions from one evaluator file."""

    values: Mapping[str, Mapping[str, float]]
    errors: Mapping[str, str]


@dataclass(frozen=True)
class Eligibility:
    """One evaluator's availability for one reference comparison."""

    submission_id: str
    comparison: str
    status: str
    complete_directions: tuple[str, ...]
    partial_directions: tuple[str, ...]
    absent_directions: tuple[str, ...]
    message: str = ""


@dataclass(frozen=True)
class DirectionScore:
    """Paired Task 2 component scores for one evaluator and direction."""

    comparison: str
    direction: str
    submission_id: str
    submission_name: str
    entry_type: str
    left_label: str
    right_label: str
    item_count: int
    system_count: int
    left_pdp: float
    right_pdp: float
    delta_pdp: float
    left_spa: float
    right_spa: float
    delta_spa: float
    left_task2: float
    right_task2: float
    delta_task2: float


@dataclass(frozen=True)
class MacroScore:
    """Equal-direction macro Task 2 scores for one evaluator."""

    comparison: str
    submission_id: str
    submission_name: str
    entry_type: str
    directions: tuple[str, ...]
    left_pdp: float
    right_pdp: float
    delta_pdp: float
    left_spa: float
    right_spa: float
    delta_spa: float
    left_task2: float
    right_task2: float
    delta_task2: float


@dataclass(frozen=True)
class ReferenceQualityResult:
    """Complete Task 2 reference-input analysis and validation audit."""

    index: ReferenceQualityIndex
    directions: Mapping[tuple[str, str], ReferenceComparisonDirection]
    evidence: Mapping[tuple[str, str], HumanEvidence]
    common_directions: Mapping[str, tuple[str, ...]]
    eligibility: tuple[Eligibility, ...]
    direction_scores: tuple[DirectionScore, ...]
    macro_scores: tuple[MacroScore, ...]
    issues: tuple[str, ...]
    spa_permutations: int
    spa_seed: int


def _parse_item_id(value: Any, context: str) -> tuple[str, str, str, str, str, str]:
    """Validate and split one six-field WMT item identifier."""

    if not isinstance(value, str):
        raise ValueError(f"{context}: item_id must be a string")
    fields = tuple(value.split(ITEM_SEPARATOR))
    if len(fields) != 6 or any(not field for field in fields):
        raise ValueError(f"{context}: invalid six-field item_id {value!r}")
    return fields  # type: ignore[return-value]


def _read_jsonl(path: Path):
    """Yield line numbers and validated JSON objects from a JSONL file."""

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each line must be an object")
            yield line_number, value


def _variant(document_id: str) -> tuple[str, str] | None:
    """Return alternate condition and stripped official document ID."""

    remaining = document_id
    tokens: list[str] = []
    while remaining.startswith("RefQuality-PE_") or remaining.startswith("RefQuality-PD_"):
        token, remaining = remaining.split("_", 1)
        tokens.append(token)
    if not tokens:
        return None
    if tokens == ["RefQuality-PE"]:
        condition = "postedit"
    elif tokens in (["RefQuality-PD"], ["RefQuality-PD", "RefQuality-PE"]):
        condition = "pseudo"
    else:
        raise ValueError(f"Unsupported RefQuality prefix sequence {tokens!r}")
    return condition, remaining


def _reference(record: Mapping[str, Any], context: str) -> tuple[str, str]:
    """Return a validated reference type and text."""

    value = record.get("ref")
    if not isinstance(value, dict):
        raise ValueError(f"{context}: ref must be an object")
    kind, text = value.get("type"), value.get("text")
    if kind not in REFERENCE_SYSTEM_BY_TYPE or not isinstance(text, str):
        raise ValueError(f"{context}: invalid reference type or text")
    return kind, text


def _hypotheses(record: Mapping[str, Any], context: str) -> dict[str, str]:
    """Return a strictly validated hypothesis mapping."""

    values = record.get("hyps")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{context}: hyps must be a non-empty object")
    if any(not isinstance(system, str) or not system for system in values):
        raise ValueError(f"{context}: hypothesis names must be non-empty strings")
    if any(not isinstance(text, str) for text in values.values()):
        raise ValueError(f"{context}: hypothesis texts must be strings")
    return dict(values)


def load_reference_quality_index(path: Path) -> ReferenceQualityIndex:
    """Load, match, and validate every complete RefQuality direction."""

    alternate: list[tuple[int, dict[str, Any], str, str, str]] = []
    needed_official_ids: set[str] = set()
    represented_directions: set[str] = set()
    for line_number, record in _read_jsonl(path):
        fields = _parse_item_id(record.get("item_id"), f"{path}:{line_number}")
        if fields[0] != "challenge":
            continue
        parsed = _variant(fields[4])
        if parsed is None:
            continue
        condition, document = parsed
        official_id = ITEM_SEPARATOR.join(("official", fields[1], fields[2], fields[3], document, fields[5]))
        direction = f"{fields[1]}-{fields[2]}"
        alternate.append((line_number, record, condition, official_id, direction))
        needed_official_ids.add(official_id)
        represented_directions.add(direction)
    if not alternate:
        raise ValueError(f"{path}: no RefQuality rows")

    official: dict[str, tuple[int, dict[str, Any]]] = {}
    official_ids_by_direction: dict[str, set[str]] = defaultdict(set)
    for line_number, record in _read_jsonl(path):
        fields = _parse_item_id(record.get("item_id"), f"{path}:{line_number}")
        if fields[0] != "official":
            continue
        direction = f"{fields[1]}-{fields[2]}"
        if direction in represented_directions:
            official_ids_by_direction[direction].add(record["item_id"])
        if record["item_id"] in needed_official_ids:
            official[record["item_id"]] = (line_number, record)

    grouped: dict[str, dict[str, dict[str, ReferenceMirror]]] = {
        comparison: defaultdict(dict) for comparison in COMPARISONS
    }
    for line_number, challenge, condition, official_id, direction in alternate:
        context = f"{path}:{line_number}"
        if official_id not in official:
            raise ValueError(f"{context}: missing official mirror {official_id!r}")
        official_line, source = official[official_id]
        source_context = f"{path}:{official_line}"
        if challenge.get("src") != source.get("src") or not isinstance(source.get("src"), str):
            raise ValueError(f"{context}: source differs from official mirror")
        left_type, left_text = _reference(source, source_context)
        right_type, right_text = _reference(challenge, context)
        if right_type != condition:
            raise ValueError(f"{context}: RefQuality prefix and ref.type disagree")
        if condition == "postedit" and left_type != "human":
            raise ValueError(f"{context}: postedit comparison requires a from-scratch official reference")
        if condition == "pseudo" and left_type not in {"human", "postedit"}:
            raise ValueError(f"{context}: pseudo comparison requires a human official reference")
        source_hypotheses = _hypotheses(source, source_context)
        challenge_hypotheses = _hypotheses(challenge, context)
        challenge_only = sorted(set(challenge_hypotheses) - set(source_hypotheses))
        if challenge_only:
            raise ValueError(f"{context}: challenge-only hypotheses {challenge_only[:3]}")
        mismatched = sorted(system for system,text in challenge_hypotheses.items() if source_hypotheses[system] != text)
        if mismatched:
            raise ValueError(f"{context}: hypotheses differ from official mirror: {mismatched[:3]}")
        left_system = REFERENCE_SYSTEM_BY_TYPE[left_type]
        right_system = REFERENCE_SYSTEM_BY_TYPE[right_type]
        if source_hypotheses.get(left_system) != left_text:
            raise ValueError(f"{source_context}: {left_system!r} does not equal ref.text")
        if source_hypotheses.get(right_system) != right_text:
            raise ValueError(f"{context}: official {right_system!r} candidate does not equal alternate ref.text")
        comparison = MAIN_VS_PSEUDO if condition == "pseudo" else FROM_SCRATCH_VS_POSTEDIT
        common = tuple(sorted(set(challenge_hypotheses) - REFERENCE_CANDIDATE_SYSTEMS))
        if len(common) < 2:
            raise ValueError(f"{context}: fewer than two common non-reference systems")
        by_item = grouped[comparison][direction]
        if official_id in by_item:
            raise ValueError(f"{context}: duplicate {comparison} mirror")
        by_item[official_id] = ReferenceMirror(
            direction, official_id, str(challenge["item_id"]), left_type, right_type,
            left_system, right_system,
            tuple(sorted(source_hypotheses)), tuple(sorted(challenge_hypotheses)), common,
            tuple(sorted(set(source_hypotheses) - set(challenge_hypotheses))),
            left_text != right_text,
        )

    directions: dict[str, dict[str, ReferenceComparisonDirection]] = {comparison: {} for comparison in COMPARISONS}
    for comparison, by_direction in grouped.items():
        for direction, by_item in by_direction.items():
            expected = official_ids_by_direction[direction]
            if set(by_item) != expected:
                raise ValueError(f"{path}: {comparison}/{direction} is not a complete official mirror: missing={len(expected-set(by_item))}, extra={len(set(by_item)-expected)}")
            mirrors = tuple(by_item[item] for item in sorted(by_item))
            left_types = {mirror.left_reference_type for mirror in mirrors}
            right_types = {mirror.right_reference_type for mirror in mirrors}
            if len(left_types) != 1 or len(right_types) != 1:
                raise ValueError(f"{path}: reference types vary within {comparison}/{direction}")
            left_type, right_type = next(iter(left_types)), next(iter(right_types))
            directions[comparison][direction] = ReferenceComparisonDirection(
                comparison, direction,
                REFERENCE_SYSTEM_BY_TYPE[left_type], REFERENCE_SYSTEM_BY_TYPE[right_type],
                mirrors,
            )
    return ReferenceQualityIndex(path, {key: dict(value) for key,value in directions.items()}, len(alternate))


def build_evidence(direction: ReferenceComparisonDirection, screens: Iterable[Screen]) -> HumanEvidence:
    """Build matched official human evidence after removing both references."""

    omitted = {
        system
        for mirror in direction.mirrors
        for system in mirror.omitted_systems
    }
    present_references = {
        system
        for mirror in direction.mirrors
        for system in mirror.official_systems
        if system in REFERENCE_CANDIDATE_SYSTEMS
    }
    excluded = omitted | present_references
    test_set = {
        mirror.official_item_id: {
            system: ""
            for system in mirror.official_systems
        }
        for mirror in direction.mirrors
    }
    evidence = aggregate_human_evidence(
        test_set,
        screens,
        direction.direction,
        excluded_systems=tuple(sorted(excluded)),
    )
    forbidden = REFERENCE_CANDIDATE_SYSTEMS
    if forbidden & set(evidence.systems):
        raise AssertionError("Compared reference candidates survived evidence filtering")
    return evidence


def load_relevant_predictions(path: Path, index: ReferenceQualityIndex) -> PredictionInventory:
    """Scan one evaluator once and validate every relevant RefQuality row."""

    expected: dict[str, tuple[str, ...]] = {}
    for directions in index.directions.values():
        for direction in directions.values():
            for mirror in direction.mirrors:
                expected[mirror.official_item_id] = mirror.official_systems
                expected[mirror.alternate_item_id] = mirror.alternate_systems
    values: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}
    for line_number, record in _read_jsonl(path):
        item_id = record.get("item_id")
        if item_id not in expected:
            continue
        context = f"{path}:{line_number}"
        if item_id in values or item_id in errors:
            errors[str(item_id)] = f"{context}: duplicate relevant item_id"
            values.pop(str(item_id), None)
            continue
        raw = record.get("task2_pred")
        if not isinstance(raw, dict) or not raw:
            errors[str(item_id)] = f"{context}: task2_pred must be a non-empty object"
            continue
        wanted = set(expected[str(item_id)])
        if set(raw) != wanted:
            errors[str(item_id)] = f"{context}: task2_pred systems differ from the mirrored test row (missing={len(wanted-set(raw))}, extra={len(set(raw)-wanted)})"
            continue
        parsed: dict[str, float] = {}
        invalid = None
        for system,value in raw.items():
            if type(value) not in (int,float) or not math.isfinite(value):
                invalid = f"{context}, system {system!r}: score must be finite"
                break
            parsed[system] = float(value)
        if invalid:
            errors[str(item_id)] = invalid
        else:
            values[str(item_id)] = parsed
    return PredictionInventory(values, errors)


def _direction_status(direction: ReferenceComparisonDirection, inventory: PredictionInventory) -> tuple[str,str]:
    """Classify paired prediction coverage as complete, partial, or absent."""

    alternate_ids = {mirror.alternate_item_id for mirror in direction.mirrors}
    seen_alternate = alternate_ids & (set(inventory.values) | set(inventory.errors))
    if not seen_alternate:
        return "absent", "no alternate-reference predictions"
    required = {item for mirror in direction.mirrors for item in (mirror.official_item_id,mirror.alternate_item_id)}
    missing = required - set(inventory.values)
    errored = required & set(inventory.errors)
    if missing or errored:
        return "partial", f"missing={len(missing)}, invalid={len(errored)}"
    return "complete", "complete and finite"


def _condition_predictions(direction: ReferenceComparisonDirection, inventory: PredictionInventory, *, right: bool) -> dict[str,dict[str,float]]:
    """Return matched predictions keyed by official item IDs."""

    return {
        mirror.official_item_id: {
            system: inventory.values[mirror.alternate_item_id if right else mirror.official_item_id][system]
            for system in mirror.common_systems
        }
        for mirror in direction.mirrors
    }


def analyze_reference_quality(
    submissions: Iterable[SubmissionDescriptor],
    metadata: Mapping[str, EntryMetadata],
    index: ReferenceQualityIndex,
    screens: Iterable[Screen],
    *,
    spa_permutations: int = DEFAULT_SPA_PERMUTATIONS,
    spa_seed: int = DEFAULT_SPA_SEED,
) -> ReferenceQualityResult:
    """Validate the evaluator list and compare paired Task 2 scores by reference."""

    screen_values = tuple(screens)
    available: dict[tuple[str,str],ReferenceComparisonDirection] = {}
    evidence: dict[tuple[str,str],HumanEvidence] = {}
    issues: list[str] = []
    for comparison,directions in index.directions.items():
        for direction,definition in directions.items():
            try:
                evidence[(comparison,direction)] = build_evidence(definition,screen_values)
                available[(comparison,direction)] = definition
            except ValueError as exc:
                issues.append(f"Skipped {comparison}/{direction}: {exc}")

    inventories: dict[str,PredictionInventory] = {}
    eligibility: list[Eligibility] = []
    candidates: dict[str,list[str]] = {comparison:[] for comparison in COMPARISONS}
    complete_by_entry: dict[tuple[str,str],tuple[str,...]] = {}
    selected = tuple(submissions)
    for submission in selected:
        entry = metadata[submission.submission_id]
        if not entry.uses_references:
            for comparison in COMPARISONS:
                eligibility.append(Eligibility(entry.submission_id,comparison,"reference_free",(),(),()))
            continue
        inventory = load_relevant_predictions(submission.scores_path,index)
        inventories[entry.submission_id] = inventory
        for comparison in COMPARISONS:
            statuses = {
                direction:_direction_status(definition,inventory)
                for (kind,direction),definition in available.items() if kind==comparison
            }
            complete=tuple(sorted(direction for direction,(status,_) in statuses.items() if status=="complete"))
            partial=tuple(sorted(direction for direction,(status,_) in statuses.items() if status=="partial"))
            absent=tuple(sorted(direction for direction,(status,_) in statuses.items() if status=="absent"))
            status="partial" if partial else ("candidate" if complete else "no_predictions")
            message="; ".join(f"{direction}: {detail}" for direction,(value,detail) in statuses.items() if value=="partial")
            eligibility.append(Eligibility(entry.submission_id,comparison,status,complete,partial,absent,message))
            complete_by_entry[(entry.submission_id,comparison)]=complete
            if status=="candidate": candidates[comparison].append(entry.submission_id)

    common: dict[str,tuple[str,...]] = {}
    for comparison in COMPARISONS:
        ids=candidates[comparison]
        if not ids:
            common[comparison]=()
            continue
        values=set(complete_by_entry[(ids[0],comparison)])
        for entry_id in ids[1:]: values.intersection_update(complete_by_entry[(entry_id,comparison)])
        common[comparison]=tuple(sorted(values))
    eligibility=[Eligibility(row.submission_id,row.comparison,"eligible" if row.status=="candidate" and common[row.comparison] else row.status,row.complete_directions,row.partial_directions,row.absent_directions,row.message) for row in eligibility]

    descriptor_by_id={entry.submission_id:entry for entry in selected}
    direction_scores: list[DirectionScore] = []
    references={key:prepare_spa_reference(value,spa_permutations,spa_seed) for key,value in evidence.items() if key[1] in common[key[0]]}
    eligible_ids={(row.submission_id,row.comparison) for row in eligibility if row.status=="eligible"}
    for entry_id,comparison in sorted(eligible_ids):
        entry=metadata[entry_id]; inventory=inventories[entry_id]
        for direction in common[comparison]:
            definition=available[(comparison,direction)]; human=evidence[(comparison,direction)]
            left=_condition_predictions(definition,inventory,right=False)
            right=_condition_predictions(definition,inventory,right=True)
            left_pdp,_=score_measure(human,left,GROUP_BY_SCREEN_PDP)
            right_pdp,_=score_measure(human,right,GROUP_BY_SCREEN_PDP)
            left_spa,_=score_measure(human,left,SEGMENT_BASED_SPA,spa_permutations=spa_permutations,spa_seed=spa_seed,spa_reference=references[(comparison,direction)])
            right_spa,_=score_measure(human,right,SEGMENT_BASED_SPA,spa_permutations=spa_permutations,spa_seed=spa_seed,spa_reference=references[(comparison,direction)])
            left_task2=(left_pdp+left_spa)/2; right_task2=(right_pdp+right_spa)/2
            direction_scores.append(DirectionScore(comparison,direction,entry_id,entry.submission_name,entry.entry_type,definition.left_label,definition.right_label,definition.item_count,len(human.systems),left_pdp,right_pdp,right_pdp-left_pdp,left_spa,right_spa,right_spa-left_spa,left_task2,right_task2,right_task2-left_task2))

    macros: list[MacroScore] = []
    for entry_id,comparison in sorted(eligible_ids):
        rows=[row for row in direction_scores if row.submission_id==entry_id and row.comparison==comparison]
        if not rows: continue
        entry=metadata[entry_id]
        means={field:statistics.fmean(getattr(row,field) for row in rows) for field in ("left_pdp","right_pdp","left_spa","right_spa","left_task2","right_task2")}
        macros.append(MacroScore(comparison,entry_id,entry.submission_name,entry.entry_type,tuple(row.direction for row in rows),means["left_pdp"],means["right_pdp"],means["right_pdp"]-means["left_pdp"],means["left_spa"],means["right_spa"],means["right_spa"]-means["left_spa"],means["left_task2"],means["right_task2"],means["right_task2"]-means["left_task2"]))
    direction_scores.sort(key=lambda row:(row.comparison,row.direction,row.entry_type,row.submission_name.casefold()))
    macros.sort(key=lambda row:(row.comparison,row.entry_type,row.submission_name.casefold()))
    return ReferenceQualityResult(index,available,evidence,common,tuple(eligibility),tuple(direction_scores),tuple(macros),tuple(issues),spa_permutations,spa_seed)
