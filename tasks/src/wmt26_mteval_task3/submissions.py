"""Validate and evaluate selected direct Task 3 submission archives."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import (
    DataError,
    find_submission_archive,
    iter_jsonl_text,
    parse_json,
    parse_item_id,
    read_archive_member,
    sha256_bytes,
    sha256_file,
)
from .metrics import PairObservations
from .model_cards import load_answers, required_text, resolved_base_model
from .models import (
    HumanReference,
    OfficialInventory,
    SubmissionMetadata,
    SubmissionSpec,
)


def _validate_prediction_map(
    value: Any, source: str
) -> dict[str, bool]:
    """Validate a nonempty Boolean system map and return it unchanged."""
    if not isinstance(value, dict) or not value:
        raise DataError(f"{source}: task3_pred must be a non-empty object")
    for system, prediction in value.items():
        if not isinstance(system, str) or not system:
            raise DataError(f"{source}: system names must be non-empty strings")
        if type(prediction) is not bool:
            raise DataError(
                f"{source}: prediction for {system!r} must be a JSON Boolean"
            )
    return value


def validate_model_card(
    data: bytes, template: list[dict[str, Any]], source: str
) -> dict[str, str]:
    """Validate an original unsanitized submission card against its original template."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataError(f"{source}: model card is not UTF-8") from error
    card = parse_json(text, source)
    if not isinstance(card, list) or len(card) != len(template):
        raise DataError(
            f"{source}: expected {len(template)} model-card questions"
        )
    fields = {"question", "required", "answer"}
    answers = {}
    for index, (expected, actual) in enumerate(zip(template, card), start=1):
        if not isinstance(expected, dict) or set(expected) != fields:
            raise DataError(f"Model-card template question {index} is invalid")
        if not isinstance(actual, dict) or set(actual) != fields:
            raise DataError(f"{source}: question {index} has invalid fields")
        if actual["question"] != expected["question"]:
            raise DataError(f"{source}: question {index} text was altered")
        if actual["required"] is not expected["required"]:
            raise DataError(f"{source}: question {index} required flag was altered")
        answer = actual["answer"]
        if not isinstance(answer, str):
            raise DataError(f"{source}: answer {index} must be a string")
        if expected["required"] and not answer.strip():
            raise DataError(f"{source}: required question {index} is unanswered")
        answers[actual["question"]] = answer.strip()
    if card[-1]["answer"].strip() != "0":
        raise DataError(f"{source}: data collection consent is required")
    return answers


def evaluate_submission(
    spec: SubmissionSpec,
    submissions_dir: Path,
    inventory: OfficialInventory,
    human_reference: HumanReference,
    model_card_template: list[dict[str, Any]] | None = None,
    base_model_overrides: dict[int, str] | None = None,
) -> tuple[SubmissionMetadata, list[dict[str, Any]]]:
    # The release distributes numeric-ID directories, not private ZIP wrappers.
    """Validate a released prediction payload and return metadata and per-pair metrics."""
    archive_path = submissions_dir / str(spec.submission_id)
    archive_sha256 = ""
    prediction_data = (archive_path / "prediction.jsonl").read_bytes()
    prediction_sha256 = sha256_bytes(prediction_data)
    if prediction_sha256 != spec.prediction_sha256:
        raise DataError(
            f"{archive_path.name}: prediction SHA-256 changed; expected "
            f"{spec.prediction_sha256}, found {prediction_sha256}"
        )
    model_card_data = (archive_path / "model_card.json").read_bytes()
    model_card_answers = load_answers(archive_path / "model_card.json")
    for question in ("Organization name", "Submission name"):
        required_text(model_card_answers, question, str(archive_path))
    try:
        prediction_text = prediction_data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataError(f"{archive_path.name}: prediction.jsonl is not UTF-8") from error

    submitted_ids: set[str] = set()
    submitted_official_by_pair: defaultdict[str, set[str]] = defaultdict(set)
    submitted_challenge_by_pair: defaultdict[str, set[str]] = defaultdict(set)
    observations: defaultdict[str, PairObservations] = defaultdict(PairObservations)

    for line_number, record in iter_jsonl_text(
        prediction_text, f"{archive_path.name}:prediction.jsonl"
    ):
        source = f"{archive_path.name}:prediction.jsonl:{line_number}"
        if set(record) != {"item_id", "task3_pred"}:
            raise DataError(f"{source}: expected only item_id and task3_pred")
        item_id = record["item_id"]
        parsed_id = parse_item_id(item_id, source)
        predictions = _validate_prediction_map(record["task3_pred"], source)
        if item_id in submitted_ids:
            raise DataError(f"{source}: duplicate item_id {item_id!r}")
        submitted_ids.add(item_id)

        if parsed_id.set_id == "challenge":
            test_item = inventory.challenge_items.get(item_id)
            if test_item is None:
                raise DataError(f"{source}: unknown challenge item_id {item_id!r}")
            if set(predictions) != set(test_item.hypotheses):
                raise DataError(f"{source}: challenge system coverage mismatch")
            submitted_challenge_by_pair[parsed_id.language_pair].add(item_id)
            continue
        if parsed_id.set_id != "official":
            raise DataError(f"{source}: unknown set {parsed_id.set_id!r}")
        test_item = inventory.items.get(item_id)
        if test_item is None:
            raise DataError(f"{source}: unknown official item_id {item_id!r}")
        if set(predictions) != set(test_item.hypotheses):
            missing = sorted(set(test_item.hypotheses) - set(predictions))
            extra = sorted(set(predictions) - set(test_item.hypotheses))
            raise DataError(
                f"{source}: system coverage mismatch; missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )

        language_pair = parsed_id.language_pair
        submitted_official_by_pair[language_pair].add(item_id)
        for system, prediction in predictions.items():
            key = (item_id, system)
            for human in human_reference.judgments_for(key):
                observations[language_pair].add(
                    item_id, human.error_free, prediction
                )

    if not submitted_official_by_pair:
        raise DataError(f"{archive_path.name}: no official predictions")
    for language_pair, actual_ids in submitted_official_by_pair.items():
        expected_ids = inventory.item_ids_by_pair.get(language_pair)
        if expected_ids is None:
            raise DataError(
                f"{archive_path.name}: unknown language pair {language_pair}"
            )
        if actual_ids != expected_ids:
            raise DataError(
                f"{archive_path.name}: incomplete {language_pair} coverage; "
                f"expected {len(expected_ids)} items, found {len(actual_ids)}"
            )
    for language_pair, actual_ids in submitted_challenge_by_pair.items():
        expected_ids = inventory.challenge_item_ids_by_pair.get(language_pair)
        if expected_ids is None or actual_ids != expected_ids:
            expected_count = 0 if expected_ids is None else len(expected_ids)
            raise DataError(
                f"{archive_path.name}: incomplete challenge {language_pair} "
                f"coverage; expected {expected_count} items, found {len(actual_ids)}"
            )

    attempted_pairs = tuple(sorted(submitted_official_by_pair))
    metadata = SubmissionMetadata(
        spec=spec,
        archive_name=archive_path.name,
        archive_sha256=archive_sha256,
        prediction_sha256=prediction_sha256,
        model_card_sha256=sha256_bytes(model_card_data),
        submission_name=model_card_answers["Submission name"],
        organization_name=model_card_answers["Organization name"],
        attempted_pairs=attempted_pairs,
        base_model=resolved_base_model(
            model_card_answers, spec.submission_id, base_model_overrides
        ),
    )
    rows = []
    for language_pair in attempted_pairs:
        pair_observations = observations.get(language_pair)
        if pair_observations is None:
            raise DataError(
                f"{archive_path.name}: no human labels for {language_pair}"
            )
        rows.append(
            {
                "submission_id": spec.submission_id,
                "submission_key": spec.key,
                "language_pair": language_pair,
                **pair_observations.metrics(),
            }
        )
    return metadata, rows
