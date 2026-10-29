"""Test results-first release checks using synthetic inputs, not private dumps."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from wmt26_mteval_task3.reproduction import (
    ReproductionError,
    audit_dump,
    csv_projection,
    load_targets,
    main,
    verify_results,
)


def synthetic_dump(tmp_path: Path, *, card: list[dict] | None = None) -> tuple[Path, dict]:
    """Return a minimal ZIP and independent targets with deliberately missing metadata."""
    path = tmp_path / "data.zip"
    reference = b'{}\n'
    prediction = b'{"item_id": "synthetic"}\n'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("human_annotations/ground-truth.jsonl", reference)
        archive.writestr("submissions/task2/1/predictions.jsonl", prediction)
        archive.writestr("submissions/task2/1/model_card.json", json.dumps(card or []))
    targets = {
        "dump": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "shared_inputs": [{"member": "human_annotations/ground-truth.jsonl",
                           "sha256": hashlib.sha256(reference).hexdigest()}],
        "families": [{"name": "task2_converted", "required_card_questions": ["Permission", "Threshold"],
                      "entries": [{"entry_id": "1", "dump_candidates": ["submissions/task2/1", "baselines/task2/1"],
                                   "prediction_filename": "predictions.jsonl",
                                   "historical_prediction_sha256": hashlib.sha256(prediction).hexdigest()}]}],
        "remaining_gaps": [],
    }
    return path, targets


def test_packaged_targets_pin_all_result_families() -> None:
    """Keep the inspected result populations, not stale prose counts, as targets."""
    targets = load_targets()
    assert [len(family["entries"]) for family in targets["families"]] == [13, 15, 27, 9, 9]
    artifacts = {target["path"]: target for target in targets["artifacts"]}
    assert artifacts["latex/table1_official_ranking.csv"]["rows"] == 18
    assert artifacts["latex/table3_priority_all_tracks.csv"]["rows"] == 34
    assert artifacts["latex/table4_all_pairs.csv"]["rows"] == 25
    assert artifacts["threshold/threshold_summary.csv"]["rows"] == 21
    assert artifacts["threshold/per_language_threshold_summary.csv"]["rows"] == 168
    assert targets["human_reference"]["selected_labels"] == 171774
    for family in targets["families"][3:]:
        assert all("task1_prededup_and_enhy_fix" in entry["dump_candidates"][0]
                   for entry in family["entries"])
        assert all(entry["model_card_member"].startswith("baselines/task2/")
                   for entry in family["entries"])


def test_csv_projection_ignores_row_order_and_unselected_metadata(tmp_path: Path) -> None:
    """Permit source layout changes but retain exact numerical precision."""
    left, right = tmp_path / "a.csv", tmp_path / "b.csv"
    left.write_text("id,mcc,name\n2,0.20,old\n1,0.10,old\n")
    right.write_text("id,mcc,name\n1,0.10,new\n2,0.20,new\n")
    assert csv_projection(left, ["id", "mcc"], ["id"]) == csv_projection(right, ["id", "mcc"], ["id"])
    right.write_text("id,mcc,name\n1,0.11,new\n2,0.20,new\n")
    assert csv_projection(left, ["id", "mcc"], ["id"]) != csv_projection(right, ["id", "mcc"], ["id"])


def test_csv_projection_rejects_duplicate_keys_and_missing_fields(tmp_path: Path) -> None:
    """Reject ambiguous rows or schemas instead of hashing incomplete results."""
    path = tmp_path / "a.csv"
    path.write_text("id,mcc\n1,0.2\n1,0.3\n")
    with pytest.raises(ReproductionError, match="duplicate"):
        csv_projection(path, ["id", "mcc"], ["id"])
    with pytest.raises(ReproductionError, match="missing"):
        csv_projection(path, ["id", "recall"], ["id"])


def test_verify_reports_missing_and_changed_results(tmp_path: Path) -> None:
    """Require every frozen artifact and detect numeric drift."""
    path = tmp_path / "result.csv"
    path.write_text("id,mcc\n1,0.2\n")
    target = {"path": "result.csv", "columns": ["id", "mcc"], "keys": ["id"],
              **csv_projection(path, ["id", "mcc"], ["id"])}
    assert verify_results(tmp_path, {"artifacts": [target]})["matches"]
    path.write_text("id,mcc\n1,0.3\n")
    assert not verify_results(tmp_path, {"artifacts": [target]})["matches"]
    path.unlink()
    assert "error" in verify_results(tmp_path, {"artifacts": [target]})["checks"][0]


def test_audit_reports_missing_metadata_without_inventing_answers(tmp_path: Path) -> None:
    """Input hashes can match while sanitized metadata still blocks reproduction."""
    path, targets = synthetic_dump(tmp_path)
    report = audit_dump(path, targets)
    assert report["archive"]["matches_inspected_snapshot"]
    assert all(check["matches"] for check in report["checks"])
    assert report["cards"][0]["missing_questions"] == ["Permission", "Threshold"]
    assert not report["ready_for_unmodified_historical_runners"]
    assert report["blockers"][0]["affected_cards"] == 1


def test_audit_distinguishes_blank_and_missing_answers(tmp_path: Path) -> None:
    """A historically blank threshold is not a field removed by sanitization."""
    path, targets = synthetic_dump(tmp_path, card=[{"question": "Threshold", "answer": ""},
                                                   {"question": "Permission", "answer": False}])
    report = audit_dump(path, targets)
    assert report["cards"][0]["missing_questions"] == []
    assert report["cards"][0]["blank_questions"] == ["Threshold"]
    assert report["blockers"] == []


def test_audit_uses_explicit_metadata_member(tmp_path: Path) -> None:
    """Support historical prediction snapshots with separately located released cards."""
    path, targets = synthetic_dump(tmp_path)
    member = "baselines/task2/example/model_card.json"
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(member, json.dumps([{"question": "Threshold", "answer": 85},
                                             {"question": "Permission", "answer": True}]))
    targets["dump"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    targets["families"][0]["entries"][0]["model_card_member"] = member
    report = audit_dump(path, targets)
    assert report["cards"][0]["missing_questions"] == []
    assert report["cards"][0]["member"] == member
    assert report["blockers"] == []


def test_audit_reports_changed_reference(tmp_path: Path) -> None:
    """Reject a shared-reference mismatch even if selected prediction files match."""
    path, targets = synthetic_dump(tmp_path)
    targets["shared_inputs"][0]["sha256"] = "0" * 64
    report = audit_dump(path, targets)
    assert not report["checks"][0]["matches"]
    assert any(blocker["reason"] == "missing or changed frozen payloads"
               for blocker in report["blockers"])


def test_audit_rejects_duplicate_card_questions(tmp_path: Path) -> None:
    """Do not let last-value-wins JSON cards conceal conflicting conversion rules."""
    path, targets = synthetic_dump(tmp_path, card=[{"question": "Threshold", "answer": 1}] * 2)
    with pytest.raises(ReproductionError, match="duplicate"):
        audit_dump(path, targets)


def test_audit_never_falls_back_to_another_data_root(tmp_path: Path) -> None:
    """Report a missing selected payload even if similarly named files exist nearby."""
    path, targets = synthetic_dump(tmp_path)
    targets["families"][0]["entries"][0]["dump_candidates"] = ["submissions/task2/2"]
    (tmp_path / "model_card.json").write_text("[]")
    report = audit_dump(path, targets)
    assert report["blockers"][0]["reason"] == "expected one selected payload directory"


def test_cli_reports_absent_inputs(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Return a controlled failure rather than require an old private checkout."""
    assert main(["audit", str(tmp_path / "absent.zip")]) == 2
    assert "No such file" in capsys.readouterr().err
