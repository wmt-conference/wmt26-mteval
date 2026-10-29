"""Regression tests for Task 1's shared Pearmut reference converter."""

from __future__ import annotations

import copy
import json
import warnings

from scripts.convert_annotation_dump import main
from wmt26_mteval_task1.ground_truth_conversion import (
    GroundTruthConversionStats,
    convert_annotation_dump,
    convert_spans,
    records_from_screens,
)


DIRECTION = "eng_Latn-ind_Latn"
ITEM_ID = "official_###_eng_Latn_###_ind_Latn_###_news_###_doc_###_0"
TRANSLATIONS = {"A": "a", "B": "b", "C": "c"}
TEST_SET = {ITEM_ID: TRANSLATIONS}


def session(user: str, scores: tuple[int, int, int], timestamp: float) -> dict:
    """Build one valid three-system Pearmut screen for the official item."""

    return {
        "user_id": user,
        "item_i": 9,
        "item": [{"item_id": "news_###_doc_###_0", "tgt": TRANSLATIONS}],
        "annotation": [{
            system: {"score": score, "error_spans": []}
            for system, score in zip(TRANSLATIONS, scores)
        }],
        "actions": [
            {"action": "load", "time": timestamp - 0.5},
            {"action": "submit", "time": timestamp},
        ],
    }


def test_span_conversion_preserves_task1_unicode_and_sentinels() -> None:
    """Map UTF-16 ranges to code points and extract omission severity."""

    errors, omission = convert_spans(
        [
            {"start_i": 1, "end_i": 1, "severity": "Major"},
            {"start_i": 4, "end_i": 4, "severity": "minor"},
            {"start_i": 5, "end_i": 5, "severity": None},
        ],
        "A😀B",
        "fixture",
    )
    assert errors == [
        {"start": 1, "end": 2, "severity": "major", "category": None}
    ]
    assert omission == "minor"


def test_replica_and_revision_keep_latest_screen_content() -> None:
    """Collapse exact copies and keep a later changed logical screen."""

    original = session("u1", (10, 20, 30), 1)
    replica = copy.deepcopy(original)
    replica["actions"].append({"action": "mousemove", "time": 2})
    revision = session("u1", (11, 20, 30), 3)
    stats = GroundTruthConversionStats()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        screens = convert_annotation_dump(
            {"eng_Latn---ind_Latn v3": [original, replica, revision]},
            TEST_SET,
            {DIRECTION},
            stats,
        )
    assert len(screens) == 1
    assert screens[0].scores["A"] == 11
    assert stats.direction(DIRECTION).exact_replicas_collapsed == 1
    assert stats.direction(DIRECTION).revisions_applied == 1
    assert any("later annotation replaces" in str(w.message) for w in caught)


def test_reset_removes_only_that_annotators_earlier_screens() -> None:
    """Retain other annotators and later screens after a reset marker."""

    screens = convert_annotation_dump(
        {"eng_Latn---ind_Latn v3": [
            session("u1", (10, 20, 30), 1),
            session("u2", (40, 50, 60), 2),
            {"user_id": "u1", "annotation": "__RESET__", "item_i": 999},
            session("u1", (70, 80, 90), 3),
        ]},
        TEST_SET,
        {DIRECTION},
    )
    assert [(screen.annotator_id, screen.scores["A"]) for screen in screens] == [
        ("u2", 40), ("u1", 70)
    ]


def test_nested_records_preserve_task1_assessments_and_shared_scores() -> None:
    """Publish the Task 1 span content with aligned ordered score keys."""

    raw = session("u1", (10, 20, 30), 1)
    raw["annotation"][0]["A"]["error_spans"] = [
        {"start_i": 0, "end_i": 0, "severity": "major"}
    ]
    screens = convert_annotation_dump(
        {"eng_Latn---ind_Latn v3": [raw]}, TEST_SET, {DIRECTION}
    )
    records = records_from_screens(screens)
    assert records[0]["item_id"] == ITEM_ID
    screen = records[0]["screens"][0]
    assert tuple(screen["task1_pred"]) == tuple(screen["task2_pred"]) == ("A", "B", "C")
    assert screen["task1_pred"]["A"]["errors"] == [
        {"start": 0, "end": 1, "severity": "major", "category": None}
    ]


def test_cli_streams_a_selected_official_direction(tmp_path, capsys) -> None:
    """Convert a small dump and write one item-based shared JSONL record."""

    dump = tmp_path / "dump.json"
    test_set = tmp_path / "test.jsonl"
    destination = tmp_path / "ground-truth.jsonl"
    dump.write_text(
        json.dumps({"eng_Latn---ind_Latn v3": [session("u1", (10, 20, 30), 1)]}),
        encoding="utf-8",
    )
    test_set.write_text(json.dumps({"item_id": ITEM_ID, "hyps": TRANSLATIONS}) + "\n")
    assert main([str(dump), str(test_set), str(destination), "--direction", DIRECTION]) == 0
    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["item_id"] == ITEM_ID
    assert "Task 1 reference conversion recap" in capsys.readouterr().out
