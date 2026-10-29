"""Schema-aware readers shared by analysis and scoring."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from wmt26_mteval_task1.constants import (
    API_PROVIDER_QUESTION,
    CHALLENGE_SET_ID,
    CLOSED_SOURCE_MODEL_QUESTION,
    EVALUATOR_CHARACTERISTIC_QUESTIONS,
    ITEM_ID_SEPARATOR,
    ITEM_SET_IDS,
    MODEL_SIZE_QUESTION,
    OFFICIAL_SET_ID,
    ORGANIZATION_NAME_QUESTION,
    REFERENCE_DESCRIPTION_QUESTION,
    SEVERITY_WEIGHTS,
    SUBMISSION_DESCRIPTION_QUESTION,
    SUBMISSION_NAME_QUESTION,
    SYSTEM_LINK_QUESTION,
)
from wmt26_mteval_task1.models import (
    Assessment,
    Issue,
    ItemId,
    ModelCardMetadata,
    ModelCardTemplate,
    GroundTruthSet,
    Span,
    SubmissionDescriptor,
    TestSetIndex,
)


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield non-blank JSON objects with one-based line numbers."""

    from wmt26_mteval_task1.parquet_source import current_source

    source = current_source()
    if source is not None and source.contains(path):
        yield from source.iter_records(path)
        return
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: each line must be a JSON object"
                )
            yield line_number, record


def parse_item_id(value: Any, context: str) -> ItemId:
    """Validate the exact six-field WMT26 item-ID format."""

    if not isinstance(value, str):
        raise ValueError(f"{context}: item_id must be a string")
    fields = value.split(ITEM_ID_SEPARATOR)
    if len(fields) != 6 or any(not field for field in fields):
        raise ValueError(
            f"{context}: item_id must contain set_id, src_lang, tgt_lang, "
            "domain, doc_id, and seg_id separated by '_###_'"
        )
    if fields[0] not in ITEM_SET_IDS:
        raise ValueError(
            f"{context}: item_id set_id must be one of {sorted(ITEM_SET_IDS)}"
        )
    return ItemId(value, *fields)


def _load_json(path: Path, description: str) -> Any:
    """Load one JSON document and normalize file errors to validation errors."""

    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"{path}: required {description} is missing") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc


def _index_model_card_entries(
    value: Any, path: Path, description: str
) -> dict[str, dict[str, Any]]:
    """Validate and index model-card entries by their question text."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: {description} must be a non-empty JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(value, 1):
        context = f"{path}: {description} entry {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{context} must be an object")
        question = entry.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{context} must contain a non-empty question")
        if question in indexed:
            raise ValueError(
                f"{path}: {description} contains duplicate question {question!r}"
            )
        indexed[question] = entry
    return indexed


def load_model_card_template(path: Path) -> ModelCardTemplate:
    """Load a publication-safe template without private identity fields."""

    entries = _index_model_card_entries(
        _load_json(path, "model card template"), path, "model card template"
    )
    private = sorted(question for question in entries if _is_private_card_question(question))
    if private:
        raise ValueError(f"{path}: model card template contains private question(s)")
    for index, entry in enumerate(entries.values(), 1):
        if type(entry.get("required")) is not bool:
            raise ValueError(
                f"{path}: model card template entry {index} must contain a "
                "boolean required field"
            )
    reference_use_question = EVALUATOR_CHARACTERISTIC_QUESTIONS["reference_use"]
    reference_use_entry = entries.get(reference_use_question)
    if reference_use_entry is None or reference_use_entry.get("required") is not True:
        raise ValueError(
            f"{path}: model card template must require question "
            f"{reference_use_question!r}"
        )
    return ModelCardTemplate(path, entries)


def _is_private_card_question(question: str) -> bool:
    """Identify identity and consent questions excluded from public cards."""

    return question in {"Participant name", "Email"} or question.startswith(
        "Data Collection Consent:"
    )


def normalize_boolean_answer(value: Any) -> bool | None:
    """Normalize genuine booleans and case-insensitive true/false strings."""

    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def normalize_reference_use_answer(value: Any) -> bool | None:
    """Normalize reference-use answers, including the documented yes/no legacy."""

    normalized = normalize_boolean_answer(value)
    if normalized is not None:
        return normalized
    if isinstance(value, str):
        text = value.strip().casefold()
        if text == "yes":
            return True
        if text == "no":
            return False
    return None


def _metadata_from_entries(
    entries: Mapping[str, Mapping[str, Any]], *, valid: bool
) -> ModelCardMetadata:
    """Extract the reportable metadata from indexed model-card entries."""

    def text_answer(question: str) -> str | None:
        """Return one stripped, non-empty textual answer when available."""

        answer = entries.get(question, {}).get("answer")
        if not isinstance(answer, str):
            return None
        answer = answer.strip()
        return answer or None

    characteristics = {
        key: (
            normalize_reference_use_answer(entries.get(question, {}).get("answer"))
            if key == "reference_use"
            else normalize_boolean_answer(entries.get(question, {}).get("answer"))
        )
        for key, question in EVALUATOR_CHARACTERISTIC_QUESTIONS.items()
    }
    return ModelCardMetadata(
        organization_name=text_answer(ORGANIZATION_NAME_QUESTION),
        submission_name=text_answer(SUBMISSION_NAME_QUESTION),
        submission_description=text_answer(SUBMISSION_DESCRIPTION_QUESTION),
        reference_description=text_answer(REFERENCE_DESCRIPTION_QUESTION),
        model_size=text_answer(MODEL_SIZE_QUESTION),
        closed_source_model_name=text_answer(CLOSED_SOURCE_MODEL_QUESTION),
        api_provider=text_answer(API_PROVIDER_QUESTION),
        system_link=text_answer(SYSTEM_LINK_QUESTION),
        evaluator_characteristics=characteristics,
        valid=valid,
    )


def _validate_submitted_entries(
    entries: Mapping[str, Mapping[str, Any]],
    path: Path,
    template: ModelCardTemplate,
) -> None:
    """Validate submitted model-card questions and required answers."""

    if any(_is_private_card_question(question) for question in entries):
        raise ValueError(f"{path}: model card contains private question(s)")
    expected = set(template.entries_by_question)
    submitted = set(entries)
    missing = sorted(expected - submitted)
    if missing:
        examples = ", ".join(repr(question) for question in missing[:3])
        raise ValueError(
            f"{path}: model card is missing {len(missing)} template "
            f"question(s): {examples}"
        )
    unexpected = sorted(submitted - expected)
    if unexpected:
        examples = ", ".join(repr(question) for question in unexpected[:3])
        raise ValueError(
            f"{path}: model card contains {len(unexpected)} unexpected "
            f"question(s): {examples}"
        )

    for question, template_entry in template.entries_by_question.items():
        entry = entries[question]
        if template_entry["required"] is not True:
            continue
        if "answer" not in entry:
            raise ValueError(
                f"{path}: model card question {question!r} is missing its answer"
            )
        answer = entry["answer"]
        if answer is None:
            raise ValueError(
                f"{path}: model card question {question!r} has a null answer"
            )
        if isinstance(answer, str) and not answer.strip():
            raise ValueError(
                f"{path}: model card question {question!r} has an empty answer"
            )


def validate_model_card(
    path: Path, template: ModelCardTemplate | None
) -> ModelCardMetadata:
    """Read report metadata, optionally applying the legacy card template."""

    entries = _index_model_card_entries(
        _load_json(path, "model card"), path, "model card"
    )
    if template is not None:
        _validate_submitted_entries(entries, path, template)
    return _metadata_from_entries(entries, valid=True)


def read_model_card_metadata(
    submission: SubmissionDescriptor,
    template: ModelCardTemplate,
) -> tuple[ModelCardMetadata, list[Issue]]:
    """Read model-card report fields without aborting a batch analysis."""

    path = submission.model_card_path
    try:
        entries = _index_model_card_entries(
            _load_json(path, "model card"), path, "model card"
        )
    except (OSError, ValueError) as exc:
        issue = Issue(
            "model_card",
            str(exc),
            submission.submission_id,
            submission.username,
            str(path),
        )
        return ModelCardMetadata(
            evaluator_characteristics={
                key: None for key in EVALUATOR_CHARACTERISTIC_QUESTIONS
            },
        ), [issue]

    issues: list[Issue] = []
    try:
        _validate_submitted_entries(entries, path, template)
    except ValueError as exc:
        issues.append(
            Issue(
                "model_card",
                str(exc),
                submission.submission_id,
                submission.username,
                str(path),
            )
        )

    metadata = _metadata_from_entries(entries, valid=not issues)
    if issues and metadata.valid:
        metadata = replace(
            metadata,
            valid=not any(issue.severity == "error" for issue in issues),
        )
    return metadata, issues


def get_task1_mapping(record: Mapping[str, Any], context: str) -> dict[str, Any]:
    """Validate the common outer task-1 system mapping."""

    assessments = record.get("task1_pred")
    if not isinstance(assessments, dict):
        raise ValueError(f"{context}: task1_pred must be an object")
    if not assessments:
        raise ValueError(f"{context}: task1_pred must be a non-empty object")
    for system in assessments:
        if not isinstance(system, str) or not system:
            raise ValueError(f"{context}: system names must be non-empty strings")
    return assessments


def get_prediction_assessments(
    record: Mapping[str, Any], context: str
) -> dict[str, Assessment]:
    """Return validated Task 1 assessment objects keyed by evaluator system."""

    assessments = get_task1_mapping(record, context)
    for system, assessment in assessments.items():
        if not isinstance(assessment, dict):
            raise ValueError(f"{context}: assessment for {system!r} must be an object")
    return assessments


def _get_ground_truth_assessments(
    record: Mapping[str, Any], context: str
) -> dict[str, list[Assessment]]:
    """Return validated ground-truth assessments while preserving every annotation.

    Legacy single-assessment objects are wrapped in a list. Explicit lists
    require a non-empty annotator identity and valid screen context for each
    assessment, but repeated annotator identities are allowed because one
    annotator may submit multiple distinct annotations for the same pair.
    """

    assessments = get_task1_mapping(record, context)
    normalized: dict[str, list[Assessment]] = {}
    for system, value in assessments.items():
        if isinstance(value, dict):
            normalized[system] = [value]
            continue
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"{context}: ground truth for {system!r} must be an assessment "
                "object or a non-empty list"
            )
        for index, assessment in enumerate(value):
            gold_context = f"{context}, system {system!r}, assessment[{index}]"
            if not isinstance(assessment, dict):
                raise ValueError(f"{gold_context}: assessment must be an object")
            annotator = assessment.get("annotator_id")
            if not isinstance(annotator, str) or not annotator:
                raise ValueError(
                    f"{gold_context}: annotator_id must be a non-empty string"
                )
            context_models = assessment.get("context_models")
            if not isinstance(context_models, list) or any(
                not isinstance(model, str) or not model for model in context_models
            ):
                raise ValueError(
                    f"{gold_context}: context_models must be a list of non-empty strings"
                )
            if len(context_models) != len(set(context_models)):
                raise ValueError(f"{gold_context}: context_models contains duplicates")
            if system in context_models:
                raise ValueError(
                    f"{gold_context}: context_models must exclude the scored system"
                )
        normalized[system] = value
    return normalized


def _get_unified_ground_truth_assessments(
    record: Mapping[str, Any],
    context: str,
    seen_screen_ids: set[str],
) -> dict[str, list[Assessment]]:
    """Flatten validated unified screens into Task 1 ground-truth assessments.

    The enclosing screen supplies annotator identity and ordered model context;
    copies are injected into each normalized assessment for the existing Task 1
    loading pipeline without mutating the parsed JSON object.
    """

    screens = record.get("screens")
    if not isinstance(screens, list) or not screens:
        raise ValueError(f"{context}: screens must be a non-empty list")
    flattened: dict[str, list[Assessment]] = {}
    for screen_index, screen in enumerate(screens):
        screen_context = f"{context}, screens[{screen_index}]"
        if not isinstance(screen, dict):
            raise ValueError(f"{screen_context}: screen must be an object")
        screen_id = screen.get("screen_id")
        annotator = screen.get("annotator_id")
        task1_mapping = screen.get("task1_pred")
        task2_mapping = screen.get("task2_pred")
        if not isinstance(screen_id, str) or not screen_id:
            raise ValueError(f"{screen_context}: screen_id must be a non-empty string")
        if screen_id in seen_screen_ids:
            raise ValueError(f"{screen_context}: duplicate screen_id {screen_id!r}")
        seen_screen_ids.add(screen_id)
        if not isinstance(annotator, str) or not annotator:
            raise ValueError(
                f"{screen_context}: annotator_id must be a non-empty string"
            )
        if not isinstance(task1_mapping, dict) or len(task1_mapping) != 3:
            raise ValueError(
                f"{screen_context}: task1_pred must contain exactly three systems"
            )
        if not isinstance(task2_mapping, dict) or tuple(task2_mapping) != tuple(
            task1_mapping
        ):
            raise ValueError(
                f"{screen_context}: task1_pred and task2_pred must have the same "
                "three systems in the same order"
            )
        systems = tuple(task1_mapping)
        for system, score in task2_mapping.items():
            if not isinstance(system, str) or not system:
                raise ValueError(
                    f"{screen_context}: system names must be non-empty strings"
                )
            try:
                valid_score = type(score) in (int, float) and math.isfinite(score)
            except OverflowError:
                valid_score = False
            if not valid_score or not 0 <= score <= 100:
                raise ValueError(
                    f"{screen_context}, system {system!r}: score must be a "
                    "finite number between 0 and 100"
                )
            assessment = task1_mapping[system]
            if not isinstance(assessment, dict):
                raise ValueError(
                    f"{screen_context}, system {system!r}: assessment must be an object"
                )
            if "annotator_id" in assessment or "context_models" in assessment:
                raise ValueError(
                    f"{screen_context}, system {system!r}: annotator and context "
                    "belong to the enclosing screen"
                )
            enriched = dict(assessment)
            enriched["annotator_id"] = annotator
            enriched["context_models"] = [other for other in systems if other != system]
            flattened.setdefault(system, []).append(enriched)
    return flattened


def normalize_assessment(
    assessment: Mapping[str, Any],
    translation_length: int,
    context: str,
) -> tuple[list[Span], int]:
    """Validate one assessment and return metric spans plus ignored empties."""

    errors = assessment.get("errors")
    if not isinstance(errors, list):
        raise ValueError(f"{context}: errors must be a list")
    spans: list[Span] = []
    ignored = 0
    for index, error in enumerate(errors):
        error_context = f"{context}, errors[{index}]"
        if not isinstance(error, dict):
            raise ValueError(f"{error_context}: error span must be an object")
        start = error.get("start")
        end = error.get("end")
        severity = error.get("severity")
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"{error_context}: start and end must be integers")
        if start < 0 or end < 0:
            raise ValueError(f"{error_context}: start and end must be non-negative")
        if start > end:
            raise ValueError(f"{error_context}: start must not exceed end")
        if end > translation_length:
            raise ValueError(
                f"{error_context}: end {end} exceeds translation length "
                f"{translation_length}"
            )
        if not isinstance(severity, str) or severity not in SEVERITY_WEIGHTS:
            raise ValueError(
                f"{error_context}: severity must be one of {sorted(SEVERITY_WEIGHTS)}"
            )
        if start == end:
            ignored += 1
            continue
        spans.append(
            {"start": start, "end": end, "side": "target", "severity": severity}
        )

    omission = assessment.get("omission")
    if omission is not None:
        if not isinstance(omission, str) or omission not in SEVERITY_WEIGHTS:
            raise ValueError(
                f"{context}: omission severity must be one of "
                f"{sorted(SEVERITY_WEIGHTS)}"
            )
        spans.append(
            {
                "start": translation_length,
                "end": translation_length + 1,
                "side": "target",
                "severity": omission,
            }
        )
    return spans, ignored


def load_test_set(path: Path) -> TestSetIndex:
    """Load official authority and derive direction-level reference policy."""

    lengths: dict[str, dict[str, int]] = {}
    directions: dict[str, str] = {}
    domains: dict[str, str] = {}
    expected: dict[str, set[tuple[str, str]]] = {}
    source_lengths: dict[str, int] = {}
    reference_rows: dict[str, list[tuple[bool, set[str]]]] = {}
    challenge_rows = 0
    official_rows = 0
    for line_number, record in read_jsonl(path):
        context = f"{path}:{line_number}"
        item = parse_item_id(record.get("item_id"), context)
        if item.set_id == CHALLENGE_SET_ID:
            challenge_rows += 1
            continue
        official_rows += 1
        if item.raw in lengths:
            raise ValueError(f"{context}: duplicate item_id {item.raw!r}")
        hypotheses = record.get("hyps")
        if not isinstance(hypotheses, dict):
            raise ValueError(f"{context}: hyps must be an object")
        for system, translation in hypotheses.items():
            if not isinstance(system, str) or not system:
                raise ValueError(f"{context}: system names must be non-empty strings")
            if not isinstance(translation, str):
                raise ValueError(
                    f"{context}: translation for system {system!r} must be a string"
                )
        reference = record.get("ref")
        if reference is not None and not isinstance(reference, dict):
            raise ValueError(f"{context}: ref must be an object or null")
        reference_text = reference.get("text") if isinstance(reference, dict) else None
        has_reference = reference_text is not None
        if has_reference:
            if not isinstance(reference_text, str):
                raise ValueError(f"{context}: ref.text must be a string or null")
            matching_systems = {
                system
                for system, translation in hypotheses.items()
                if translation == reference_text
            }
        else:
            matching_systems = set()
        reference_rows.setdefault(item.language_pair, []).append(
            (has_reference, matching_systems)
        )
        lengths[item.raw] = {
            system: len(translation) for system, translation in hypotheses.items()
        }
        source = record.get("src")
        if isinstance(source, str):
            source_lengths[item.raw] = len(source)
        directions[item.raw] = item.language_pair
        domains[item.raw] = item.domain
        expected.setdefault(item.language_pair, set()).update(
            (item.raw, system) for system in hypotheses
        )
    if not lengths:
        raise ValueError(f"{path}: official test set is empty")
    reference_available = frozenset(
        direction
        for direction, rows in reference_rows.items()
        if rows and all(has_reference for has_reference, _ in rows)
    )
    reference_systems: dict[str, str] = {}
    for direction in sorted(reference_available):
        rows = reference_rows[direction]
        consistent = set.intersection(*(matches for _, matches in rows))
        if len(consistent) > 1:
            systems = ", ".join(repr(system) for system in sorted(consistent))
            raise ValueError(
                f"{path}: direction {direction!r} has multiple systems whose "
                f"hypotheses consistently equal ref.text: {systems}"
            )
        if consistent:
            reference_systems[direction] = next(iter(consistent))
    return TestSetIndex(
        path,
        lengths,
        directions,
        {direction: frozenset(pairs) for direction, pairs in expected.items()},
        challenge_rows,
        official_rows,
        source_lengths,
        reference_available,
        reference_systems,
        domains,
    )


def load_ground_truth(
    path: Path,
    test_set: TestSetIndex,
    *,
    require_complete: bool = False,
) -> GroundTruthSet:
    """Load and normalize human ground-truth annotations against official items."""

    raw: dict[str, dict[str, list[Assessment]]] = {}
    layout: str | None = None
    seen_screen_ids: set[str] = set()
    for line_number, record in read_jsonl(path):
        context = f"{path}:{line_number}"
        item = parse_item_id(record.get("item_id"), context)
        if item.set_id != OFFICIAL_SET_ID:
            continue
        record_layout = "unified" if "screens" in record else "legacy"
        if layout is None:
            layout = record_layout
        elif layout != record_layout:
            raise ValueError(f"{context}: ground-truth layouts must not be mixed")
        assessments = (
            _get_unified_ground_truth_assessments(record, context, seen_screen_ids)
            if record_layout == "unified"
            else _get_ground_truth_assessments(record, context)
        )
        if item.raw in raw:
            raise ValueError(f"{context}: duplicate item_id {item.raw!r}")
        raw[item.raw] = assessments
    if not raw:
        raise ValueError(f"{path}: no official ground-truth annotations were available")

    normalized: dict[str, dict[str, tuple[list[Span], ...]]] = {}
    normalized_annotators: dict[str, dict[str, tuple[str | None, ...]]] = {}
    zero_length_count = 0
    pair_counts_by_direction: dict[str, int] = {}
    for item_id, item_assessments in raw.items():
        supported_systems = test_set.translation_lengths.get(item_id)
        if supported_systems is None:
            continue
        direction = test_set.item_directions[item_id]
        excluded_system = test_set.reference_systems_by_direction.get(direction)
        retained: dict[str, tuple[list[Span], ...]] = {}
        retained_annotators: dict[str, tuple[str | None, ...]] = {}
        for system, assessments in item_assessments.items():
            if system not in supported_systems:
                continue
            if system == excluded_system:
                continue
            normalized_assessments: list[list[Span]] = []
            annotator_ids: list[str | None] = []
            for index, assessment in enumerate(assessments):
                spans, ignored = normalize_assessment(
                    assessment,
                    supported_systems[system],
                    f"ground-truth item {item_id!r}, system {system!r}, "
                    f"assessment[{index}]",
                )
                zero_length_count += ignored
                normalized_assessments.append(spans)
                annotator = assessment.get("annotator_id")
                annotator_ids.append(
                    annotator if isinstance(annotator, str) and annotator else None
                )
            retained[system] = tuple(normalized_assessments)
            retained_annotators[system] = tuple(annotator_ids)
            pair_counts_by_direction[direction] = pair_counts_by_direction.get(
                direction, 0
            ) + len(normalized_assessments)
        if retained:
            normalized[item_id] = retained
            normalized_annotators[item_id] = retained_annotators

    if require_complete:
        missing: dict[str, set[str]] = {}
        for pairs in test_set.scorable_pairs_by_direction.values():
            for item_id, system in pairs:
                if system not in normalized.get(item_id, {}):
                    missing.setdefault(item_id, set()).add(system)
        if missing:
            item_id = sorted(missing)[0]
            systems = ", ".join(sorted(missing[item_id]))
            raise ValueError(
                "Ground-truth annotations do not cover the complete official test set; "
                f"missing item {item_id!r}, systems: {systems}."
            )
    if not normalized:
        raise ValueError("No official test-set-supported ground-truth annotations were available")
    return GroundTruthSet(
        path=path,
        assessments=normalized,
        annotator_ids=normalized_annotators,
        pair_counts_by_direction=pair_counts_by_direction,
        zero_length_span_count=zero_length_count,
    )
