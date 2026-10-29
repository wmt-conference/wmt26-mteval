"""Convert Pearmut annotation dumps into Task 1 ground-truth JSONL."""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

ITEM_ID_SEPARATOR = "_###_"
SEPARATOR = ITEM_ID_SEPARATOR
OFFICIAL_SET_ID = "official"
ITEM_SET_IDS = {"official", "challenge"}
SEVERITIES = {"minor", "major"}


@dataclass(frozen=True)
class Screen:
    """One retained human screen with ordered systems, scores, and Task 1 spans."""

    screen_id: str
    item_id: str
    direction: str
    annotator_id: str
    systems: tuple[str, ...]
    scores: Mapping[str, float]
    task1_predictions: Mapping[str, Mapping[str, Any]] | None = None


@dataclass
class DirectionConversionStats:
    """Mutable filtering and history counts for one direction."""

    raw_official_screens: int = 0
    raw_score_observations: int = 0
    screens_with_non_displayed_annotations: int = 0
    non_displayed_annotations_ignored: int = 0
    reset_events: int = 0
    reset_markers: int = 0
    reset_screens_removed: int = 0
    reset_scores_removed: int = 0
    exact_replicas_collapsed: int = 0
    equivalent_snapshots_collapsed: int = 0
    revisions_applied: int = 0
    excluded_dump_items: set[str] = field(default_factory=set)


@dataclass
class GroundTruthConversionStats:
    """Mutable per-direction recap for one shared-reference conversion."""

    by_direction: dict[str, DirectionConversionStats] = field(default_factory=dict)
    selected_directions: frozenset[str] = field(default_factory=frozenset)

    def direction(self, value: str) -> DirectionConversionStats:
        """Return the counters for a direction, creating them when absent."""

        return self.by_direction.setdefault(value, DirectionConversionStats())


def parse_test_item_id(item_id: str) -> tuple[str, str]:
    """Validate a six-field test-set ID and return its set and direction."""

    fields = item_id.split(SEPARATOR)
    if len(fields) != 6 or any(not field for field in fields):
        raise ValueError(f"Invalid test-set item_id: {item_id!r}")
    set_id = fields[0]
    if set_id not in ITEM_SET_IDS:
        raise ValueError(
            f"Invalid test-set item_id set_id {set_id!r}: expected one of "
            f"{sorted(ITEM_SET_IDS)}"
        )
    return set_id, f"{fields[1]}-{fields[2]}"

def item_pair(item_id: str) -> str:
    """Validate a full test-set item ID and return its language direction."""

    _, direction = parse_test_item_id(item_id)
    return direction

def load_test_set(
    path: Path,
    directions: set[str] | frozenset[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Load and validate the authoritative test-set translations from JSONL.

    Blank lines are ignored, while every other line must be valid JSON with a
    structurally valid six-field ``item_id``. Challenge rows and, when
    ``directions`` is supplied, unselected directions are excluded before
    their translation payload is validated. Official selected IDs must be
    unique and have a ``hyps`` object with non-empty system names and string
    translations, preserving exact Unicode text without normalization. An
    empty selected test set is rejected so conversion cannot silently filter
    out the entire dump.
    """

    test_set: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            context = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{context}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{context}: each line must be a JSON object")
            item_id = record.get("item_id")
            if not isinstance(item_id, str):
                raise ValueError(f"{context}: item_id must be a string")
            set_id, direction = parse_test_item_id(item_id)
            if set_id != OFFICIAL_SET_ID:
                continue
            if directions is not None and direction not in directions:
                continue
            if item_id in test_set:
                raise ValueError(f"{context}: duplicate item_id {item_id!r}")
            hypotheses = record.get("hyps")
            if not isinstance(hypotheses, dict):
                raise ValueError(f"{context}: hyps must be an object")
            for system, translation in hypotheses.items():
                if not isinstance(system, str) or not system:
                    raise ValueError(
                        f"{context}: system names must be non-empty strings"
                    )
                if not isinstance(translation, str):
                    raise ValueError(
                        f"{context}: translation for {system!r} must be a string"
                    )
            test_set[item_id] = hypotheses
    if not test_set:
        raise ValueError(f"{path}: official test set is empty")
    return test_set

def normalize_severity(value: Any, context: str) -> str:
    """Return the canonical lowercase Task 1 severity."""

    if not isinstance(value, str):
        raise ValueError(f"{context}: severity must be a string")
    severity = value.strip().lower()
    if severity not in SEVERITIES:
        raise ValueError(f"{context}: severity must be one of {sorted(SEVERITIES)}")
    return severity

def _utf16_boundaries(translation: str, context: str) -> tuple[int, dict[int, int]]:
    """Return UTF-16 length and boundary-to-code-point positions."""

    utf16_position = 0
    boundaries = {0: 0}
    for codepoint_position, character in enumerate(translation, 1):
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError(
                f"{context}: translation contains an unpaired Unicode surrogate"
            )
        utf16_position += 2 if codepoint > 0xFFFF else 1
        boundaries[utf16_position] = codepoint_position
    return utf16_position, boundaries

def _map_utf16_range_to_codepoints(
    start: int,
    inclusive_end: int,
    boundaries: Mapping[int, int],
) -> tuple[int, int]:
    """Map one in-bounds inclusive UTF-16 range to Python boundaries."""

    start_boundary = start if start in boundaries else start - 1
    exclusive_end = inclusive_end + 1
    end_boundary = exclusive_end if exclusive_end in boundaries else exclusive_end + 1
    return boundaries[start_boundary], boundaries[end_boundary]

def convert_spans(
    raw_spans: Any,
    translation: str,
    context: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Apply the Task 1 converter's canonical span and sentinel semantics."""

    if not isinstance(raw_spans, list):
        raise ValueError(f"{context}: error_spans must be a list")

    utf16_length, boundaries = _utf16_boundaries(translation, context)
    errors: list[dict[str, Any]] = []
    omission: str | None = None
    for index, raw_span in enumerate(raw_spans):
        span_context = f"{context}, error_spans[{index}]"
        if not isinstance(raw_span, dict):
            raise ValueError(f"{span_context}: span must be an object")
        start = raw_span.get("start_i")
        inclusive_end = raw_span.get("end_i")
        if type(start) is not int or type(inclusive_end) is not int:
            raise ValueError(f"{span_context}: start_i and end_i must be integers")
        if start < 0 or inclusive_end < start:
            raise ValueError(f"{span_context}: require 0 <= start_i <= end_i")

        if start == inclusive_end == utf16_length:
            severity = normalize_severity(raw_span.get("severity"), span_context)
            if omission is not None:
                raise ValueError(f"{context}: multiple omissions are not allowed")
            omission = severity
            continue

        if start == inclusive_end == utf16_length + 1:
            # Instruction-fault markers are not retained Task 1 gold content;
            # the canonical instruction_fault field remains None.
            continue

        if inclusive_end >= utf16_length:
            raise ValueError(
                f"{span_context}: range [{start}, {inclusive_end}] is outside "
                f"the UTF-16 target length {utf16_length}; only singleton "
                f"omission {utf16_length} and instruction-fault "
                f"{utf16_length + 1} sentinels are allowed"
            )
        mapped_start, mapped_end = _map_utf16_range_to_codepoints(
            start, inclusive_end, boundaries
        )
        severity = normalize_severity(raw_span.get("severity"), span_context)
        errors.append(
            {
                "start": mapped_start,
                "end": mapped_end,
                "severity": severity,
                "category": raw_span.get("category"),
            }
        )
    return errors, omission

def task1_assessment_content(
    raw_spans: Any,
    translation: str,
    annotator_id: str,
    context_models: list[str],
    context: str,
) -> dict[str, Any]:
    """Return exactly the Task 1 assessment fields relevant to equality."""

    errors, omission = convert_spans(raw_spans, translation, context)
    return {
        "errors": errors,
        "omission": omission,
        "instruction_fault": None,
        "annotator_id": annotator_id,
        "context_models": context_models,
    }

@dataclass(frozen=True)
class _ScreenCandidate:
    """Internal screen snapshot retained until RESET and replica handling."""

    direction: str
    source_language: str
    target_language: str
    item_id: str
    annotator_id: str
    systems: tuple[str, ...]
    scores: tuple[float, ...]
    task1_contents: tuple[Mapping[str, Any], ...]
    context: str
    submitted_at: float
    ordinal: int

    @property
    def identity(self) -> tuple[str, str, str, tuple[str, ...]]:
        """Return the shared logical-screen identity."""

        return self.direction, self.item_id, self.annotator_id, self.systems

    @property
    def content(self) -> tuple[tuple[str, float, Mapping[str, Any]], ...]:
        """Return Task 1-equivalent content independent of dump replication."""

        return tuple(zip(self.systems, self.scores, self.task1_contents))

def _parse_pair(value: Any) -> tuple[str, str]:
    """Parse ``SOURCE---TARGET vN`` and return its two language codes."""

    if not isinstance(value, str):
        raise ValueError("Annotation-dump language-pair keys must be strings")
    pair, marker, version = value.rpartition(" v")
    languages = pair.split("---")
    if not marker or not version.isdigit() or len(languages) != 2:
        raise ValueError(f"Invalid annotation-dump language-pair key: {value!r}")
    if not all(languages):
        raise ValueError(f"Invalid annotation-dump language-pair key: {value!r}")
    return languages[0], languages[1]

def _submission_time(record: Mapping[str, Any], context: str) -> float:
    """Return the latest finite timestamp among successful submit actions."""

    actions = record.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError(f"{context}: actions must be a list")
    times = [
        action.get("time")
        for action in actions
        if isinstance(action, dict) and action.get("action") == "submit"
    ]
    if not times:
        raise ValueError(f"{context}: real annotation has no valid submit action")
    for value in times:
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{context}: submit action times must be finite numbers")
    return float(max(times))

def _screen_load_time(record: Mapping[str, Any], context: str) -> float:
    """Return the sole finite load timestamp for one retained screen.

    The timestamp validates that the record represents one real Pearmut page
    visit. Logical duplicate identity deliberately does not use its value.
    """

    actions = record.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError(f"{context}: actions must be a list")
    times = [
        action.get("time")
        for action in actions
        if isinstance(action, dict) and action.get("action") == "load"
    ]
    if len(times) != 1:
        raise ValueError(
            f"{context}: real annotation must have exactly one load action"
        )
    value = times[0]
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{context}: load action time must be a finite number")
    return float(value)

def _screen_index(value: Any, context: str) -> str | int:
    """Validate a Pearmut ordinary screen identifier."""

    if type(value) not in (str, int):
        raise ValueError(f"{context}: item_i must be a string or integer")
    return value

def _validate_reset_block(
    pair_name: str, active_users: set[str], reset_users: set[str]
) -> None:
    """Require each reset user to have history in its current epoch."""

    unexpected = sorted(reset_users - active_users)
    if unexpected:
        raise ValueError(
            f"{pair_name!r}: invalid annotator reset block (users without "
            f"preceding annotations: {', '.join(unexpected)})"
        )

def _finite_score(value: Any, context: str) -> float:
    """Validate and return one exact numeric, finite human quality score."""

    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{context}: score must be a finite JSON number")
    score = float(value)
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"{context}: human cESA score must be between 0 and 100")
    return score

def _process_section(
    pair_name: str,
    source_language: str,
    target_language: str,
    sessions: Iterable[Any],
    test_set: Mapping[str, Mapping[str, str]],
    stats: GroundTruthConversionStats,
    starting_ordinal: int,
) -> tuple[list[_ScreenCandidate], int]:
    """Convert one selected direction while applying its ordered RESET history."""

    direction = f"{source_language}-{target_language}"
    direction_stats = stats.direction(direction)
    candidates: list[_ScreenCandidate] = []
    active_users: set[str] = set()
    reset_users: set[str] | None = None
    ordinal = starting_ordinal

    for session_index, session in enumerate(sessions):
        context = f"{pair_name!r}, session[{session_index}]"
        if not isinstance(session, dict):
            raise ValueError(f"{context}: session must be an object")
        user_id = session.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"{context}: user_id must be a non-empty string")
        annotations = session.get("annotation")

        if annotations == "__RESET__":
            if "item_i" in session:
                _screen_index(session["item_i"], context)
            if reset_users is None:
                reset_users = set()
                direction_stats.reset_events += 1
            reset_users.add(user_id)
            removed = [
                candidate
                for candidate in candidates
                if candidate.annotator_id == user_id
            ]
            if removed:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.annotator_id != user_id
                ]
                direction_stats.reset_screens_removed += len(removed)
                direction_stats.reset_scores_removed += sum(
                    len(candidate.scores) for candidate in removed
                )
            direction_stats.reset_markers += 1
            continue

        if reset_users is not None:
            _validate_reset_block(pair_name, active_users, reset_users)
            active_users.difference_update(reset_users)
            reset_users = None

        _screen_index(session.get("item_i"), context)
        active_users.add(user_id)
        items = session.get("item")
        if not isinstance(annotations, list) or not isinstance(items, list):
            raise ValueError(f"{context}: annotation and item must be lists")
        if len(annotations) != len(items):
            raise ValueError(f"{context}: annotation and item lengths differ")
        submitted_at: float | None = None
        load_validated = False

        for item_index, (annotation, item) in enumerate(zip(annotations, items)):
            item_context = f"{context}, item[{item_index}]"
            if not isinstance(annotation, dict) or not isinstance(item, dict):
                raise ValueError(f"{item_context}: annotation and item must be objects")
            short_id = item.get("item_id")
            if not isinstance(short_id, str):
                if short_id is None and not annotation:
                    continue
                raise ValueError(f"{item_context}: item_id must be a string")
            short_fields = short_id.split(ITEM_ID_SEPARATOR)
            if len(short_fields) != 3:
                # Tutorials, attention checks, and completion screens are not
                # official human evidence.
                continue

            full_id = ITEM_ID_SEPARATOR.join(
                [OFFICIAL_SET_ID, source_language, target_language, *short_fields]
            )
            test_hypotheses = test_set.get(full_id)
            if test_hypotheses is None:
                direction_stats.excluded_dump_items.add(full_id)
                continue
            targets = item.get("tgt")
            if not isinstance(targets, dict):
                raise ValueError(f"{item_context}: tgt must be an object")
            systems = tuple(targets)
            if len(systems) != 3:
                raise ValueError(
                    f"{item_context}: retained official screens must contain "
                    f"exactly three systems, found {len(systems)}"
                )
            if any(not isinstance(system, str) or not system for system in systems):
                raise ValueError(
                    f"{item_context}: model names must be non-empty strings"
                )
            unknown = [system for system in systems if system not in test_hypotheses]
            if unknown:
                raise ValueError(
                    f"{item_context}: retained screen contains systems absent "
                    f"from the blindset: {unknown}"
                )
            missing_annotations = [
                system for system in systems if system not in annotation
            ]
            if missing_annotations:
                raise ValueError(
                    f"{item_context}: retained screen lacks annotations for "
                    f"displayed systems: {missing_annotations}"
                )
            non_displayed = [
                system for system in annotation if system not in targets
            ]
            if non_displayed:
                direction_stats.screens_with_non_displayed_annotations += 1
                direction_stats.non_displayed_annotations_ignored += len(
                    non_displayed
                )

            scores: list[float] = []
            task1_contents: list[Mapping[str, Any]] = []
            for system in systems:
                model_context = f"{item_context}, model {system!r}"
                payload = annotation[system]
                if not isinstance(payload, dict):
                    raise ValueError(f"{model_context}: annotation must be an object")
                scores.append(_finite_score(payload.get("score"), model_context))
                dump_translation = targets.get(system)
                if not isinstance(dump_translation, str):
                    raise ValueError(
                        f"{model_context}: target translation must be a string"
                    )
                if dump_translation != test_hypotheses[system]:
                    raise ValueError(
                        f"{model_context}: dump translation differs from the "
                        "authoritative blindset translation"
                    )
                task1_contents.append(
                    task1_assessment_content(
                        payload.get("error_spans"),
                        test_hypotheses[system],
                        user_id,
                        [other for other in systems if other != system],
                        model_context,
                    )
                )

            if submitted_at is None:
                submitted_at = _submission_time(session, item_context)
            if not load_validated:
                _screen_load_time(session, item_context)
                load_validated = True
            direction_stats.raw_official_screens += 1
            direction_stats.raw_score_observations += len(scores)
            candidates.append(
                _ScreenCandidate(
                    direction,
                    source_language,
                    target_language,
                    full_id,
                    user_id,
                    systems,
                    tuple(scores),
                    tuple(task1_contents),
                    item_context,
                    submitted_at,
                    ordinal,
                )
            )
            ordinal += 1

    if reset_users is not None:
        _validate_reset_block(pair_name, active_users, reset_users)

    # A repeated logical screen is a dump snapshot or edit, not independent
    # evidence. Identical content is safe to collapse. Changed content from a
    # later dump occurrence replaces the earlier value with an explicit warning.
    unique: dict[tuple[str, str, str, tuple[str, ...]], _ScreenCandidate] = {}
    for candidate in candidates:
        previous = unique.get(candidate.identity)
        if previous is None:
            unique[candidate.identity] = candidate
            continue
        if candidate.content != previous.content:
            warnings.warn(
                "WARNING: later annotation replaces the previous value for "
                f"logical screen item {candidate.item_id!r}, annotator "
                f"{candidate.annotator_id!r}, systems {candidate.systems!r} "
                f"from {previous.context} (submitted {previous.submitted_at}) "
                f"with {candidate.context} (submitted {candidate.submitted_at})",
                RuntimeWarning,
                stacklevel=2,
            )
            unique[candidate.identity] = candidate
            direction_stats.revisions_applied += 1
            continue
        if candidate.submitted_at == previous.submitted_at:
            direction_stats.exact_replicas_collapsed += 1
        else:
            direction_stats.equivalent_snapshots_collapsed += 1

    return sorted(unique.values(), key=lambda value: value.ordinal), ordinal

def _finalize_candidates(candidates: Iterable[_ScreenCandidate]) -> list[Screen]:
    """Assign stable per-direction counters and create public screen objects."""

    counters: Counter[str] = Counter()
    screens: list[Screen] = []
    for candidate in sorted(candidates, key=lambda value: value.ordinal):
        index = counters[candidate.direction]
        counters[candidate.direction] += 1
        screen_id = ITEM_ID_SEPARATOR.join(
            [candidate.source_language, candidate.target_language, str(index)]
        )
        screens.append(
            Screen(
                screen_id,
                candidate.item_id,
                candidate.direction,
                candidate.annotator_id,
                candidate.systems,
                dict(zip(candidate.systems, candidate.scores)),
                dict(zip(candidate.systems, candidate.task1_contents)),
            )
        )
    return screens

def convert_annotation_dump(
    dump: Any,
    test_set: Mapping[str, Mapping[str, str]],
    directions: Iterable[str],
    stats: GroundTruthConversionStats | None = None,
) -> list[Screen]:
    """Convert an in-memory Pearmut dump to unified human screens.

    This entry point is convenient for tests and small dumps. The command-line
    workflow uses :func:`convert_annotation_path`, which applies the same
    section processor while streaming the large official dump.
    """

    if not isinstance(dump, dict):
        raise ValueError("Annotation dump must be a JSON object")
    selected = frozenset(directions)
    if not selected:
        raise ValueError("At least one direction must be selected")
    stats = (
        GroundTruthConversionStats(selected_directions=selected) if stats is None else stats
    )
    stats.selected_directions = selected
    seen: dict[str, str] = {}
    found: set[str] = set()
    all_candidates: list[_ScreenCandidate] = []
    ordinal = 0
    for pair_name, sessions in dump.items():
        source, target = _parse_pair(pair_name)
        direction = f"{source}-{target}"
        if direction in seen:
            raise ValueError(
                f"Multiple dump sections describe {direction!r}: "
                f"{seen[direction]!r} and {pair_name!r}"
            )
        seen[direction] = pair_name
        if direction not in selected:
            continue
        found.add(direction)
        if not isinstance(sessions, list):
            raise ValueError(f"{pair_name!r}: expected a list of sessions")
        section_candidates, ordinal = _process_section(
            pair_name, source, target, sessions, test_set, stats, ordinal
        )
        all_candidates.extend(section_candidates)
    missing = sorted(selected - found)
    if missing:
        raise ValueError(f"Selected directions missing from dump: {', '.join(missing)}")
    return _finalize_candidates(all_candidates)

class _JsonStream:
    """Small incremental JSON reader specialized for a top-level object."""

    def __init__(self, stream: TextIO, path: Path, chunk_size: int = 1 << 20):
        self.stream = stream
        self.path = path
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _fill(self) -> bool:
        """Compact consumed text and append one bounded input chunk."""

        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        chunk = self.stream.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def _skip_space(self) -> None:
        """Advance over JSON whitespace, filling the buffer as needed."""

        while True:
            while (
                self.position < len(self.buffer)
                and self.buffer[self.position] in " \t\r\n"
            ):
                self.position += 1
            if self.position < len(self.buffer) or not self._fill():
                return

    def peek(self) -> str:
        """Return the next non-whitespace character or fail at EOF."""

        self._skip_space()
        if self.position >= len(self.buffer):
            raise ValueError(f"{self.path}: unexpected end of JSON")
        return self.buffer[self.position]

    def consume(self, expected: str) -> None:
        """Consume one required non-whitespace structural character."""

        actual = self.peek()
        if actual != expected:
            raise ValueError(f"{self.path}: expected {expected!r}, found {actual!r}")
        self.position += 1

    def value(self) -> Any:
        """Decode one complete JSON value without reading the full file."""

        self._skip_space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as exc:
                if self.eof or not self._fill():
                    raise ValueError(
                        f"{self.path}: invalid JSON near character {exc.pos}: {exc.msg}"
                    ) from exc
                continue
            self.position = end
            return value

    def skip_compound(self) -> None:
        """Skip one array/object lexically without materializing its values."""

        opening = self.peek()
        if opening not in "[{":
            self.value()
            return
        closing = {"[": "]", "{": "}"}
        stack = [closing[opening]]
        self.position += 1
        in_string = False
        escaped = False
        while stack:
            if self.position >= len(self.buffer):
                if not self._fill():
                    raise ValueError(f"{self.path}: unterminated JSON value")
                continue
            character = self.buffer[self.position]
            self.position += 1
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in "[{":
                stack.append(closing[character])
            elif character in "]}":
                expected = stack.pop()
                if character != expected:
                    raise ValueError(
                        f"{self.path}: mismatched JSON delimiter {character!r}"
                    )

    def array_values(self) -> Iterator[Any]:
        """Yield values from the next JSON array one at a time."""

        self.consume("[")
        if self.peek() == "]":
            self.position += 1
            return
        while True:
            yield self.value()
            delimiter = self.peek()
            if delimiter == "]":
                self.position += 1
                return
            if delimiter != ",":
                raise ValueError(f"{self.path}: expected ',' or ']' in section array")
            self.position += 1

def _consume_dump_path(
    path: Path,
    selected: frozenset[str],
    consumer: Callable[[str, str, str, Iterable[Any]], None],
) -> frozenset[str]:
    """Stream selected sections and return every validated dump direction."""

    seen: dict[str, str] = {}
    found: set[str] = set()
    with path.open(encoding="utf-8") as source:
        reader = _JsonStream(source, path)
        reader.consume("{")
        if reader.peek() == "}":
            reader.position += 1
        else:
            while True:
                pair_name = reader.value()
                source_language, target_language = _parse_pair(pair_name)
                direction = f"{source_language}-{target_language}"
                if direction in seen:
                    raise ValueError(
                        f"Multiple dump sections describe {direction!r}: "
                        f"{seen[direction]!r} and {pair_name!r}"
                    )
                seen[direction] = pair_name
                reader.consume(":")
                if direction in selected:
                    found.add(direction)
                    consumer(
                        pair_name,
                        source_language,
                        target_language,
                        reader.array_values(),
                    )
                else:
                    if reader.peek() != "[":
                        raise ValueError(f"{pair_name!r}: expected a session array")
                    reader.skip_compound()
                delimiter = reader.peek()
                if delimiter == "}":
                    reader.position += 1
                    break
                if delimiter != ",":
                    raise ValueError(f"{path}: expected ',' or '}}' after dump section")
                reader.position += 1
        reader._skip_space()
        if reader.position < len(reader.buffer) or reader._fill():
            reader._skip_space()
            if reader.position < len(reader.buffer):
                raise ValueError(f"{path}: trailing content after top-level object")
    missing = sorted(selected - found)
    if missing:
        raise ValueError(f"Selected directions missing from dump: {', '.join(missing)}")

    return frozenset(seen)

def load_dump_directions(path: Path) -> frozenset[str]:
    """Return dump directions without materializing any section array.

    The complete top-level structure is still validated, including duplicate
    normalized directions and trailing content.
    """

    return _consume_dump_path(path, frozenset(), lambda *_: None)

def convert_annotation_path(
    path: Path,
    test_set: Mapping[str, Mapping[str, str]],
    directions: Iterable[str],
    stats: GroundTruthConversionStats | None = None,
) -> list[Screen]:
    """Stream the dump and convert the selected official directions."""

    selected = frozenset(directions)
    if not selected:
        raise ValueError("At least one direction must be selected")
    stats = (
        GroundTruthConversionStats(selected_directions=selected) if stats is None else stats
    )
    stats.selected_directions = selected
    all_candidates: list[_ScreenCandidate] = []
    ordinal = 0

    def consume(
        pair_name: str,
        source_language: str,
        target_language: str,
        sessions: Iterable[Any],
    ) -> None:
        """Process one streamed selected section before parsing resumes."""

        nonlocal ordinal
        section_candidates, ordinal = _process_section(
            pair_name,
            source_language,
            target_language,
            sessions,
            test_set,
            stats,
            ordinal,
        )
        all_candidates.extend(section_candidates)

    _consume_dump_path(path, selected, consume)
    return _finalize_candidates(all_candidates)

def records_from_screens(screens: Iterable[Screen]) -> list[dict[str, Any]]:
    """Return deterministic item records containing unified nested screens."""

    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for screen in screens:
        assessments = screen.task1_predictions
        if assessments is None:
            raise ValueError(
                f"Screen {screen.screen_id!r} has no Task 1 assessment content"
            )
        if (
            tuple(assessments) != screen.systems
            or tuple(screen.scores) != screen.systems
        ):
            raise ValueError(
                f"Screen {screen.screen_id!r} has inconsistent ordered systems"
            )
        task1_predictions: dict[str, dict[str, Any]] = {}
        for system in screen.systems:
            assessment = assessments[system]
            if not isinstance(assessment, Mapping):
                raise ValueError(
                    f"Screen {screen.screen_id!r}, system {system!r}: "
                    "Task 1 assessment must be an object"
                )
            task1_predictions[system] = {
                "errors": assessment.get("errors"),
                "omission": assessment.get("omission"),
                "instruction_fault": assessment.get("instruction_fault"),
            }
        by_item[screen.item_id].append(
            {
                "screen_id": screen.screen_id,
                "annotator_id": screen.annotator_id,
                "task1_pred": task1_predictions,
                "task2_pred": dict(screen.scores),
            }
        )
    return [
        {"item_id": item_id, "screens": by_item[item_id]} for item_id in sorted(by_item)
    ]

def write_ground_truth(screens: Iterable[Screen], path: Path) -> None:
    """Write unified item-based ground-truth annotation JSONL after validation."""

    records = records_from_screens(screens)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")




def print_recap(
    test_set: Mapping[str, Mapping[str, str]],
    screens: Iterable[Screen],
    stats: GroundTruthConversionStats,
    stream: TextIO | None = None,
) -> None:
    """Print retained-screen and official-item coverage for each direction."""

    if stream is None:
        stream = sys.stdout
    retained = Counter(screen.direction for screen in screens)
    official = Counter(item_pair(item_id) for item_id in test_set)
    print("Task 1 reference conversion recap", file=stream)
    print("direction\tofficial_items\tretained_screens", file=stream)
    for direction in sorted(stats.selected_directions):
        print(
            f"{direction}\t{official[direction]}\t{retained[direction]}",
            file=stream,
        )
