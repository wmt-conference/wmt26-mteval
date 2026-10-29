from __future__ import annotations

import argparse
from wmt26_mteval_task1.cli import resolve_baseline_selection, resolve_selection
from wmt26_mteval_task1.coverage import build_coverage_report
from wmt26_mteval_task1.models import TestSetIndex as Index
from wmt26_mteval_task1.submissions import (
    Exclusions,
    discover_baselines,
    discover_submissions,
    select_baselines,
    select_submissions,
)

from _task1_helpers import (
    write_json,
    write_jsonl,
)


def make_submission(tmp_path, name, *, card=None, predictions=()):
    path = tmp_path / "submissions" / name
    path.mkdir(parents=True)
    if card is not None:
        write_json(path / "model_card.json", card)
    write_jsonl(path / "predictions.jsonl", predictions)
    return path


def test_discovery_selects_unique_numeric_submission_ids(tmp_path):
    """Discover anonymized ID-only directories in numeric ID order."""

    make_submission(tmp_path, "2")
    make_submission(tmp_path, "1")
    submissions, issues = discover_submissions(tmp_path / "submissions")

    selected = select_submissions(submissions)

    assert not issues
    assert [entry.submission_id for entry in selected] == ["1", "2"]
    assert all(entry.username == "" for entry in selected)


def test_baseline_discovery_uses_name_without_synthetic_submission_id(tmp_path):
    (tmp_path / "baselines" / "Gemini").mkdir(parents=True)
    (tmp_path / "baselines" / "Qwen_ref").mkdir()

    baselines, issues = discover_baselines(tmp_path / "baselines")

    assert not issues
    assert [entry.path.name for entry in baselines] == ["Gemini", "Qwen_ref"]
    assert all(entry.is_baseline for entry in baselines)
    assert all(entry.submission_id == "" for entry in baselines)
    assert [entry.report_label for entry in baselines] == [
        "Baseline: Gemini",
        "Baseline: Qwen_ref",
    ]


def test_baseline_selection_supports_subsets_and_an_explicit_empty_set(tmp_path):
    (tmp_path / "baselines" / "Gemini").mkdir(parents=True)
    (tmp_path / "baselines" / "Qwen_ref").mkdir()
    baselines, _ = discover_baselines(tmp_path / "baselines")

    selected = select_baselines(baselines, baseline_selectors=("Qwen_ref",))

    assert [entry.path.name for entry in selected] == ["Qwen_ref"]
    assert select_baselines(baselines, baseline_selectors=()) == ()


def test_snapshot_baseline_families_keep_common_report_identities(tmp_path):
    """Keep physical families distinct while reports use the shared source name."""

    roots = {
        "task1": ("gemini_task1__normal", "gemini_task1"),
        "task1_deduplicated": ("gemini_task1__deduplicated", "gemini_task1"),
        "task1_prededup_and_enhy_fix": (
            "gemini_task1__pre_dedup_pre_enhy_fix",
            "gemini_task1",
        ),
    }
    for root_name, (directory, source_name) in roots.items():
        root = tmp_path / root_name
        (root / directory).mkdir(parents=True)
        (baseline,) = discover_baselines(root)[0]
        assert baseline.baseline_name == source_name
        assert baseline.source_baseline_name == source_name
        assert baseline.cache_name == directory
        assert baseline.report_label == f"Baseline: {source_name}"
        assert select_baselines((baseline,), baseline_selectors=(source_name,)) == (
            baseline,
        )


def test_empty_category_selections_do_not_require_input_directories(tmp_path):
    arguments = argparse.Namespace(
        submissions_dir=tmp_path / "missing-submissions",
        submission_selectors=[],
        exclude_file=None,
        exclude_submission=[],
        baselines_dir=tmp_path / "missing-baselines",
        baseline_selectors=[],
    )

    assert resolve_selection(arguments) == ((), ())
    assert resolve_baseline_selection(arguments) == ((), ())


def test_selection_supports_exact_exclusions(tmp_path):
    """Exclude only the exact numeric submission ID requested."""

    for submission_id in ("1", "2"):
        make_submission(tmp_path, submission_id)
    submissions, _ = discover_submissions(tmp_path / "submissions")

    selected = select_submissions(
        submissions, exclusions=Exclusions(submissions=frozenset({"2"}))
    )

    assert [entry.submission_id for entry in selected] == ["1"]


def test_primary_only_requires_all_primary_and_no_other_predictions(tmp_path):
    from wmt26_mteval_task1.constants import PRIMARY_LANGUAGE_PAIRS

    expected = {
        direction: frozenset({(direction, "system")})
        for direction in PRIMARY_LANGUAGE_PAIRS | {"extra-direction"}
    }
    index = Index(tmp_path / "test", {}, {}, expected, 0, 0)
    primary_pairs = {
        direction: {(direction, "system")} for direction in PRIMARY_LANGUAGE_PAIRS
    }

    primary = build_coverage_report(
        index, primary_pairs, official_rows=8, challenge_rows=0, total_rows=8
    )
    with_extra = build_coverage_report(
        index,
        {**primary_pairs, "extra-direction": {("extra-direction", "system")}},
        official_rows=9,
        challenge_rows=0,
        total_rows=9,
    )

    assert primary.all_primary_complete and primary.primary_only
    assert with_extra.all_primary_complete and not with_extra.primary_only
