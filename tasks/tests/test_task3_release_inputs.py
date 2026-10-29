"""Validate temporary metadata repair and the transplanted upstream logic."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from wmt26_mteval_task3 import preparation
from wmt26_mteval_task3.io import DataError


PACKAGE = Path(preparation.__file__).parent
LOGIC = json.loads((PACKAGE / "upstream_logic.json").read_text())["definitions"]


def normalized_ast(node: ast.AST) -> str:
    """Canonicalize one definition while ignoring documentation-only changes."""
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(child) is not None:
                child.body = child.body[1:]
    return ast.dump(node, include_attributes=False)


@pytest.mark.parametrize("definition", LOGIC, ids=lambda row: row["file"] + ":" + row["definition"])
def test_original_definitions_remain_unchanged(definition: dict) -> None:
    """Keep original computational/reporting bodies intact, not just similar outputs."""
    module = ast.parse((PACKAGE / definition["file"]).read_text())
    node = next(n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name == definition["definition"])
    assert hashlib.sha256(normalized_ast(node).encode()).hexdigest() == definition["ast_sha256"]


def release_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Provide an independent synthetic release and a private original metadata card."""
    card = [{"question": "Organization name", "required": True, "answer": "Example"}]
    original = [*card, {"question": "Permission", "required": True, "answer": False},
                {"question": "Threshold", "required": False, "answer": ""},
                {"question": "Email", "required": True, "answer": "private@example.test"}]
    metadata = tmp_path / "local-cards"
    metadata.mkdir()
    original_path = metadata / "1_model_card.json"
    original_path.write_text(json.dumps(original))
    prediction = b'{"item_id":"synthetic"}\n'
    reference = b'{}\n'
    dump = tmp_path / "dump.zip"
    with zipfile.ZipFile(dump, "w") as archive:
        archive.writestr("human_annotations/ground-truth.jsonl", reference)
        archive.writestr("submissions/task2/1/predictions.jsonl", prediction)
        archive.writestr("submissions/task2/1/model_card.json", json.dumps(card))
    targets = {
        "dump": {"sha256": hashlib.sha256(dump.read_bytes()).hexdigest()},
        "shared_inputs": [{"member": "human_annotations/ground-truth.jsonl",
                           "sha256": hashlib.sha256(reference).hexdigest()}],
        "families": [{"name": "task2_converted", "required_card_questions": ["Permission", "Threshold"],
                      "entries": [{"entry_id": "1", "dump_candidates": ["submissions/task2/1"],
                                   "prediction_filename": "predictions.jsonl",
                                   "historical_prediction_sha256": hashlib.sha256(prediction).hexdigest(),
                                   "historical_model_card_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest()}]}],
    }
    monkeypatch.setattr(preparation, "load_targets", lambda: targets)
    return dump, metadata


def test_repair_is_explicit_and_never_copies_personal_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore only absent evaluation fields and record the authorized source identity."""
    dump, metadata = release_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "prepared"
    manifest = preparation.prepare_inputs(dump, destination, task2_cards=metadata)
    card = json.loads((destination / "submissions/task2/1/model_card.json").read_text())
    answers = {row["question"]: row["answer"] for row in card}
    assert answers == {"Organization name": "Example", "Permission": False, "Threshold": ""}
    assert len(manifest["metadata_repairs"]) == 1
    assert "private@example.test" not in json.dumps(manifest)
    assert preparation.validate_inputs(destination)["schema_version"] == 1
    # Prepared data remains independently replayable after the local originals disappear.
    (metadata / "1_model_card.json").unlink()
    assert preparation.validate_inputs(destination)["schema_version"] == 1


def test_missing_metadata_fails_atomically_without_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never silently opt a system in or publish a partially prepared input tree."""
    dump, _ = release_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "prepared"
    with pytest.raises(DataError, match="Missing conversion metadata"):
        preparation.prepare_inputs(dump, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".prepared-*"))


def test_repair_rejects_wrong_local_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not restore fields from an unverified local model-card revision."""
    dump, metadata = release_fixture(tmp_path, monkeypatch)
    (metadata / "1_model_card.json").write_text("[]")
    with pytest.raises(DataError, match="frozen original card hash"):
        preparation.prepare_inputs(dump, tmp_path / "prepared", task2_cards=metadata)


def test_prepared_payload_cannot_be_changed_by_editing_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check selected prediction hashes against the independent frozen catalog."""
    dump, metadata = release_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "prepared"
    preparation.prepare_inputs(dump, destination, task2_cards=metadata)
    with pytest.raises(DataError, match="existing input directory"):
        preparation.prepare_inputs(dump, destination, task2_cards=metadata)
    relative = "submissions/task2/1/predictions.jsonl"
    path = destination / relative
    path.write_text("changed\n")
    manifest_path = destination / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative].update(bytes=path.stat().st_size,
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DataError, match="frozen target"):
        preparation.validate_inputs(destination)


def test_existing_blank_or_false_answers_are_not_overwritten() -> None:
    """A missing field differs from an explicit opt-out or blank threshold answer."""
    card = [{"question": "Permission", "answer": False}, {"question": "Threshold", "answer": ""}]
    repaired, changes = preparation._repair_card(card, ["Permission", "Threshold"], None, "unused")
    assert repaired == card
    assert changes == []
