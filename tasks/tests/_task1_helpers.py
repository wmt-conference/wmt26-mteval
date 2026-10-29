from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from wmt26_mteval_task1.constants import (
    EVALUATOR_CHARACTERISTIC_QUESTIONS,
    PRIMARY_LANGUAGE_PAIRS,
)
from wmt26_mteval_task1.models import SubmissionDescriptor


def model_card_template() -> list[dict[str, Any]]:
    """Build the minimal required model-card question template for tests."""
    questions = [
        ("Organization name", True),
        (EVALUATOR_CHARACTERISTIC_QUESTIONS["reference_use"], True),
        ("Submission name", True),
    ]
    return [
        {"question": question, "required": required, "answer": ""}
        for question, required in questions
    ]


def model_card(
    *,
    reference_use: str = "false",
    organization: str = "Test Site",
    submission: str = "Test System",
) -> list[dict[str, Any]]:
    """Answer the test model-card template with one evaluator's metadata."""
    answers = {
        "Organization name": organization,
        "Submission name": submission,
        EVALUATOR_CHARACTERISTIC_QUESTIONS["reference_use"]: reference_use,
    }
    return [
        {"question": entry["question"], "answer": answers[entry["question"]]}
        for entry in model_card_template()
    ]


def write_json(path: Path, value: Any) -> None:
    """Write one JSON test fixture, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    """Write ordered JSONL test records, creating their parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def item_id(
    target: str = "deu_Latn",
    *,
    set_id: str = "official",
    document: str = "doc",
    segment: str = "0",
) -> str:
    """Build one WMT-style item ID for a test direction and segment."""
    return f"{set_id}_###_eng_Latn_###_{target}_###_news_###_{document}_###_{segment}"


def assessment(
    errors: list[dict[str, Any]] | None = None,
    *,
    omission: str | None = None,
) -> dict[str, Any]:
    """Build one Task 1 assessment with text errors and an optional omission."""
    return {"errors": errors or [], "omission": omission}


def major_span(start: int, end: int) -> dict[str, Any]:
    """Build one major text-error span for a test prediction."""
    return {"start": start, "end": end, "severity": "major"}


def baseline_fixture(tmp_path: Path) -> tuple:
    """Create complete priority-direction test, gold, and baseline fixtures.

    Returns the descriptor, three input paths, prediction rows, and test rows.
    The first target is empty to exercise termination coverage at that boundary.
    """

    test_rows, predictions, gold = [], [], []
    for index, direction in enumerate(sorted(PRIMARY_LANGUAGE_PAIRS)):
        source, target = direction.split("-")
        key = f"official_###_{source}_###_{target}_###_news_###_doc_###_0"
        empty = index == 0
        test_rows.append({"item_id": key, "hyps": {"system": "" if empty else "abcd"}})
        prediction = (
            assessment(omission="major")
            if empty else assessment([major_span(0, 2)])
        )
        predictions.append({"item_id": key, "task1_pred": {"system": prediction}})
        gold.append({"item_id": key, "task1_pred": {"system": prediction}})
    test_path, reference_path, template_path = (
        tmp_path / name for name in ("test.jsonl", "ground-truth.jsonl", "template.json")
    )
    write_jsonl(test_path, test_rows)
    write_jsonl(reference_path, gold)
    write_json(template_path, model_card_template())
    path = tmp_path / "baselines" / "test_task1"
    write_json(path / "model_card.json", model_card())
    write_jsonl(path / "predictions.jsonl", predictions)
    descriptor = SubmissionDescriptor(path, "", "baseline", "", "", True)
    return descriptor, test_path, reference_path, template_path, predictions, test_rows
