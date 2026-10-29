"""Tests for matched RefQuality indexing, eligibility, scoring, and reports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from wmt26_mteval_task1.io import (
    load_model_card_template,
    load_ground_truth,
    load_test_set,
)
from wmt26_mteval_task1.reference_quality import (
    NATIVE_VS_POSTEDIT,
    NATIVE_VS_PSEUDO,
    OFFICIAL_VS_PSEUDO,
    POSTEDIT_VS_PSEUDO,
    REFERENCE_CANDIDATE_SYSTEMS,
    analyze_reference_quality,
    load_reference_quality_index,
)
from wmt26_mteval_task1.reference_quality_reporting import (
    _unified_non_xcomet_rows,
    write_paper_reference_quality_report,
)
from wmt26_mteval_task1.submissions import discover_baselines, discover_submissions

from _task1_helpers import (
    assessment,
    item_id,
    major_span,
    model_card,
    model_card_template,
    write_json,
    write_jsonl,
)


def _challenge_id(prefix: str, *, document: str = "doc", segment: str = "0") -> str:
    """Return one fixture RefQuality item identifier."""

    return item_id(
        set_id="challenge",
        document=f"{prefix}_{document}",
        segment=segment,
    )


def _analysis_context(tmp_path: Path, *, partial_pseudo: bool = False):
    """Create one complete native/postedit/pseudo comparison fixture."""

    official = item_id()
    postedit = _challenge_id("RefQuality-PE")
    pseudo = _challenge_id("RefQuality-PD")
    test_path = tmp_path / "test.jsonl"
    reference_path = tmp_path / "ground-truth.jsonl"
    template_path = tmp_path / "template.json"
    hypotheses = {
        "native": "native reference",
        "Human (from scratch)": "reserved native candidate",
        "Human (postediting)": "post-edited reference",
        "PseudoRef": "pseudo reference",
        "pseudo-system": "pseudo candidate",
        "mt": "abcd",
    }
    write_jsonl(
        test_path,
        [
            {
                "item_id": official,
                "src": "source",
                "ref": {"type": "human", "text": "native reference"},
                "hyps": hypotheses,
            },
            {
                "item_id": postedit,
                "src": "source",
                "ref": {"type": "postedit", "text": "post-edited reference"},
                "hyps": {
                    "native": "native reference",
                    "Human (from scratch)": "reserved native candidate",
                    "Human (postediting)": "post-edited reference",
                    "PseudoRef": "pseudo reference",
                    "mt": "abcd",
                },
            },
            {
                "item_id": pseudo,
                "src": "source",
                "ref": {"type": "pseudo", "text": "pseudo reference"},
                "hyps": {
                    "native": "native reference",
                    "Human (from scratch)": "reserved native candidate",
                    "Human (postediting)": "post-edited reference",
                    "PseudoRef": "pseudo reference",
                    "mt": "abcd",
                },
            },
        ],
    )
    write_jsonl(
        reference_path,
        [
            {
                "item_id": official,
                "task1_pred": {
                    "mt": assessment([major_span(0, 4)]),
                    "pseudo-system": assessment([major_span(0, 4)]),
                    "native": assessment([]),
                    "Human (from scratch)": assessment([major_span(0, 4)]),
                    "Human (postediting)": assessment([major_span(0, 4)]),
                    "PseudoRef": assessment([major_span(0, 4)]),
                },
            }
        ],
    )
    write_json(template_path, model_card_template())
    submission_path = (
        tmp_path / "submissions" / "1"
    )
    submission_path.mkdir(parents=True)
    write_json(
        submission_path / "model_card.json",
        model_card(reference_use="true", submission="Reference System"),
    )
    pseudo_predictions = {
        "native": assessment([]),
        "Human (from scratch)": assessment([]),
        "Human (postediting)": assessment([]),
        "PseudoRef": assessment([]),
    }
    if not partial_pseudo:
        pseudo_predictions["mt"] = assessment([])
    write_jsonl(
        submission_path / "predictions.jsonl",
        [
            {
                "item_id": official,
                "task1_pred": {
                    "native": assessment([]),
                    "Human (from scratch)": assessment([]),
                    "Human (postediting)": assessment([]),
                    "PseudoRef": assessment([]),
                    "pseudo-system": assessment([major_span(0, 4)]),
                    "mt": assessment([major_span(0, 2)]),
                },
            },
            {
                "item_id": postedit,
                "task1_pred": {
                    "native": assessment([]),
                    "Human (from scratch)": assessment([]),
                    "Human (postediting)": assessment([]),
                    "PseudoRef": assessment([]),
                    "mt": assessment([major_span(0, 4)]),
                },
            },
            {"item_id": pseudo, "task1_pred": pseudo_predictions},
        ],
    )
    test_set = load_test_set(test_path)
    return (
        discover_submissions(tmp_path / "submissions")[0],
        test_set,
        load_ground_truth(reference_path, test_set),
        load_model_card_template(template_path),
        load_reference_quality_index(test_path, test_set),
    )


def _two_pseudo_direction_context(tmp_path: Path):
    """Extend the base fixture with one postedit-vs-pseudo direction."""

    _analysis_context(tmp_path)
    test_path = tmp_path / "test.jsonl"
    reference_path = tmp_path / "ground-truth.jsonl"
    template_path = tmp_path / "template.json"
    predictions_path = next((tmp_path / "submissions").iterdir()) / "predictions.jsonl"
    official = item_id(target="fra_Latn", document="other")
    pseudo = _challenge_id(
        "RefQuality-PD", document="other"
    ).replace("deu_Latn", "fra_Latn")

    test_records = [json.loads(line) for line in test_path.read_text().splitlines()]
    test_records.extend(
        [
            {
                "item_id": official,
                "src": "other source",
                "ref": {"type": "postedit", "text": "post-edited other"},
                "hyps": {
                    "Human (from scratch)": "reserved native other",
                    "Human (postediting)": "post-edited other",
                    "PseudoRef": "pseudo other",
                    "mt": "wxyz",
                },
            },
            {
                "item_id": pseudo,
                "src": "other source",
                "ref": {"type": "pseudo", "text": "pseudo other"},
                "hyps": {
                    "Human (from scratch)": "reserved native other",
                    "Human (postediting)": "post-edited other",
                    "PseudoRef": "pseudo other",
                    "mt": "wxyz",
                },
            },
        ]
    )
    write_jsonl(test_path, test_records)

    reference_records = [
        json.loads(line) for line in reference_path.read_text().splitlines()
    ]
    reference_records.append(
        {
            "item_id": official,
            "task1_pred": {"mt": assessment([major_span(0, 4)])},
        }
    )
    write_jsonl(reference_path, reference_records)

    prediction_records = [
        json.loads(line) for line in predictions_path.read_text().splitlines()
    ]
    prediction_records.extend(
        [
            {"item_id": official, "task1_pred": {"mt": assessment([])}},
            {"item_id": pseudo, "task1_pred": {"mt": assessment([])}},
        ]
    )
    write_jsonl(predictions_path, prediction_records)

    test_set = load_test_set(test_path)
    return (
        discover_submissions(tmp_path / "submissions")[0][0],
        test_set,
        load_ground_truth(reference_path, test_set),
        load_model_card_template(template_path),
        load_reference_quality_index(test_path, test_set),
        pseudo,
    )


def test_index_builds_both_complete_comparisons(tmp_path):
    """Index prefix semantics, reference types, and identical-text disclosure."""

    _, _, _, _, index = _analysis_context(tmp_path)

    assert index.refquality_row_count == 2
    assert index.pseudo_row_count == 1
    assert index.postedit_row_count == 1
    assert index.prefix_counts == {"RefQuality-PD": 1, "RefQuality-PE": 1}
    pseudo = index.directions_by_comparison[OFFICIAL_VS_PSEUDO][
        "eng_Latn-deu_Latn"
    ]
    postedit = index.directions_by_comparison[NATIVE_VS_POSTEDIT][
        "eng_Latn-deu_Latn"
    ]
    assert pseudo.left_reference_type == "native (human)"
    assert pseudo.right_reference_type == "pseudo"
    assert postedit.right_reference_type == "postedit"
    assert pseudo.changed_reference_count == 1
    assert pseudo.shared_test_pair_count == 2
    assert set(pseudo.mirrors[0].common_systems).isdisjoint(
        REFERENCE_CANDIDATE_SYSTEMS
    )
    assert set(index.directions_by_comparison[NATIVE_VS_PSEUDO]) == {
        "eng_Latn-deu_Latn"
    }
    assert set(index.directions_by_comparison[POSTEDIT_VS_PSEUDO]) == set()


def test_analysis_scores_matched_pairs_with_official_mpp_rules(tmp_path):
    """Score the same MT hypothesis and gold across all comparison views."""

    submissions, test_set, references, template, index = _analysis_context(tmp_path)

    result = analyze_reference_quality(
        submissions, test_set, references, template, index
    )

    assert not result.failed
    assert result.common_directions_by_comparison == {
        OFFICIAL_VS_PSEUDO: ("eng_Latn-deu_Latn",),
        NATIVE_VS_POSTEDIT: ("eng_Latn-deu_Latn",),
        NATIVE_VS_PSEUDO: ("eng_Latn-deu_Latn",),
        POSTEDIT_VS_PSEUDO: (),
    }
    official_pseudo = result.scores_by_comparison[OFFICIAL_VS_PSEUDO][0]
    native_postedit = result.scores_by_comparison[NATIVE_VS_POSTEDIT][0]
    native_pseudo = result.scores_by_comparison[NATIVE_VS_PSEUDO][0]
    assert official_pseudo.macro_left.f1 == pytest.approx(2 / 3)
    assert official_pseudo.macro_right.f1 == pytest.approx(0.0)
    assert native_postedit.macro_left.f1 == pytest.approx(2 / 3)
    assert native_postedit.macro_right.f1 == pytest.approx(1.0)
    assert native_pseudo.macro_left.f1 == pytest.approx(2 / 3)
    assert native_pseudo.macro_right.f1 == pytest.approx(0.0)
    assert official_pseudo.directions[0].scored_pair_count == 1
    assert official_pseudo.directions[0].left.aligned_assessments == 1
    status_by_comparison = {
        row.comparison: row.status for row in result.eligibility
    }
    assert status_by_comparison == {
        OFFICIAL_VS_PSEUDO: "eligible",
        NATIVE_VS_POSTEDIT: "eligible",
        NATIVE_VS_PSEUDO: "eligible",
        POSTEDIT_VS_PSEUDO: "no_predictions",
    }


def test_partial_challenge_direction_fails_only_that_comparison(tmp_path):
    """Reject partial pseudo coverage while preserving the postedit comparison."""

    submissions, test_set, references, template, index = _analysis_context(
        tmp_path, partial_pseudo=True
    )

    result = analyze_reference_quality(
        submissions, test_set, references, template, index
    )

    assert result.failed
    status = {
        row.comparison: row.status for row in result.eligibility
    }
    assert status == {
        OFFICIAL_VS_PSEUDO: "partial",
        NATIVE_VS_POSTEDIT: "eligible",
        NATIVE_VS_PSEUDO: "partial",
        POSTEDIT_VS_PSEUDO: "no_predictions",
    }
    assert result.scores_by_comparison[OFFICIAL_VS_PSEUDO] == ()
    assert len(result.scores_by_comparison[NATIVE_VS_POSTEDIT]) == 1
    assert "partial official_vs_pseudo coverage" in result.issues[0].message


def test_incomplete_entry_is_excluded_without_shrinking_directions(tmp_path):
    """Keep every valid direction and exclude an entry missing one of them."""

    complete, test_set, references, template, index, omitted_item = (
        _two_pseudo_direction_context(tmp_path)
    )
    incomplete_path = (
        tmp_path / "submissions" / "2"
    )
    incomplete_path.mkdir()
    shutil.copy2(complete.model_card_path, incomplete_path / "model_card.json")
    complete_records = [
        json.loads(line) for line in complete.predictions_path.read_text().splitlines()
    ]
    write_jsonl(
        incomplete_path / "predictions.jsonl",
        (row for row in complete_records if row["item_id"] != omitted_item),
    )
    entries = discover_submissions(tmp_path / "submissions")[0]

    result = analyze_reference_quality(
        entries, test_set, references, template, index
    )

    assert result.common_directions_by_comparison[OFFICIAL_VS_PSEUDO] == (
        "eng_Latn-deu_Latn",
        "eng_Latn-fra_Latn",
    )
    assert [
        score.descriptor.submission_id
        for score in result.scores_by_comparison[OFFICIAL_VS_PSEUDO]
    ] == ["1"]
    statuses = {
        (row.descriptor.submission_id, row.comparison): row.status
        for row in result.eligibility
    }
    assert statuses[("2", OFFICIAL_VS_PSEUDO)] == "incomplete_directions"
    assert statuses[("2", NATIVE_VS_PSEUDO)] == "eligible"


def test_invalid_reference_use_is_rejected_like_official_scoring(tmp_path):
    """Reject model cards whose required reference-use answer is not boolean."""

    submissions, test_set, references, template, index = _analysis_context(tmp_path)
    write_json(
        submissions[0].model_card_path,
        model_card(reference_use="maybe", submission="Reference System"),
    )

    result = analyze_reference_quality(
        submissions, test_set, references, template, index
    )

    assert result.failed
    assert {row.status for row in result.eligibility} == {"invalid_model_card"}
    assert result.scores_by_comparison == {
        OFFICIAL_VS_PSEUDO: (),
        NATIVE_VS_POSTEDIT: (),
        NATIVE_VS_PSEUDO: (),
        POSTEDIT_VS_PSEUDO: (),
    }
    assert len(result.issues) == 1
    assert result.issues[0].stage == "model_card"
    assert "valid required reference-use answer" in result.issues[0].message


def test_cohorts_keep_per_entry_scores_and_exclude_xcomet(tmp_path):
    """Keep individual rows while excluding xCOMET only from the second cohort."""

    submissions, test_set, references, template, index = _analysis_context(tmp_path)
    submission = submissions[0]
    baseline_path = tmp_path / "baselines" / "xcomet_fixture"
    baseline_path.mkdir(parents=True)
    shutil.copy2(submission.model_card_path, baseline_path / "model_card.json")
    shutil.copy2(submission.predictions_path, baseline_path / "predictions.jsonl")
    baselines, issues = discover_baselines(tmp_path / "baselines")
    assert not issues

    result = analyze_reference_quality(
        (submission, *baselines), test_set, references, template, index
    )

    all_scores = result.scores_by_cohort_and_comparison["all_reference_based"]
    without_xcomet = result.scores_by_cohort_and_comparison["without_xcomet"]
    assert {
        score.descriptor.report_label
        for score in all_scores[OFFICIAL_VS_PSEUDO]
    } == {submission.report_label, "Baseline: xcomet_fixture"}
    assert {
        score.descriptor.report_label
        for score in without_xcomet[OFFICIAL_VS_PSEUDO]
    } == {submission.report_label}
    assert result.cohort_membership["all_reference_based"] == (
        "1",
        "Baseline: xcomet_fixture",
    )
    assert result.cohort_membership["without_xcomet"] == ("1",)
    assert result.common_directions_by_cohort_and_comparison[
        "all_reference_based"
    ] == result.common_directions_by_cohort_and_comparison["without_xcomet"]
    summaries = {
        (summary.cohort, summary.comparison): summary
        for summary in result.cohort_summaries
    }
    all_summary = summaries[("all_reference_based", OFFICIAL_VS_PSEUDO)]
    llm_summary = summaries[("without_xcomet", OFFICIAL_VS_PSEUDO)]
    assert all_summary.entry_count == 2
    assert llm_summary.entry_count == 1
    assert all_summary.mean_delta_f1 == pytest.approx(llm_summary.mean_delta_f1)


def test_index_rejects_partially_mirrored_direction(tmp_path):
    """Require a represented RefQuality condition to mirror its full direction."""

    first = item_id(document="one")
    second = item_id(document="two")
    challenge = _challenge_id("RefQuality-PD", document="one")
    path = tmp_path / "test.jsonl"
    records = [
        {
            "item_id": first,
            "src": "one",
            "ref": {"type": "human", "text": "reference"},
            "hyps": {"reference": "reference", "mt": "abcd"},
        },
        {
            "item_id": second,
            "src": "two",
            "ref": {"type": "human", "text": "reference"},
            "hyps": {"reference": "reference", "mt": "abcd"},
        },
        {
            "item_id": challenge,
            "src": "one",
            "ref": {"type": "pseudo", "text": "pseudo"},
            "hyps": {"reference": "reference", "mt": "abcd"},
        },
    ]
    write_jsonl(path, records)
    test_set = load_test_set(path)

    with pytest.raises(ValueError, match="not a complete official mirror"):
        load_reference_quality_index(path, test_set)






def test_unified_non_xcomet_plot_rows_include_cesa_and_no_xcomet_only():
    """Select supported typed comparisons without synthesizing missing bars."""

    rows = _unified_non_xcomet_rows(
        (
            {
                "cohort": "without_xcomet",
                "comparison": NATIVE_VS_POSTEDIT,
                "language_pair": "ces_Latn-deu_Latn",
                "delta_f1": "1.25",
            },
            {
                "cohort": "without_xcomet",
                "comparison": NATIVE_VS_PSEUDO,
                "language_pair": "ces_Latn-deu_Latn",
                "delta_f1": "-0.75",
            },
            {
                "cohort": "all_reference_based",
                "comparison": POSTEDIT_VS_PSEUDO,
                "language_pair": "ces_Latn-deu_Latn",
                "delta_f1": "9.99",
            },
        )
    )

    assert [(row["language_pair"], row["comparison"]) for row in rows] == [
        ("ces_Latn-deu_Latn", NATIVE_VS_POSTEDIT),
        ("ces_Latn-deu_Latn", NATIVE_VS_PSEUDO),
    ]
    assert [row["delta_f1"] for row in rows] == [1.25, -0.75]
    assert rows[0]["cesa_label"] == "cESA: Native 80.7 vs Post-edited 79.8"
    assert rows[1]["cesa_label"] == "cESA: Native 80.7 vs Pseudo 78.4"


def test_real_refquality_mapping_snapshot():
    """Keep the reviewed real-data direction and row mapping stable."""

    from wmt26_mteval_task1.parquet_source import Task1ParquetSource

    parquet = Path(__file__).resolve().parents[1] / "wmt26-grand.parquet"
    if not parquet.is_file():
        pytest.skip("integrated WMT26 parquet file is unavailable")
    source = Task1ParquetSource(parquet, parquet.parent / "model_cards/task1")
    with source.activate():
        index = load_reference_quality_index(source.TEST_SET, load_test_set(source.TEST_SET))

    assert index.refquality_row_count == 2804
    assert index.pseudo_row_count == 1948
    assert index.postedit_row_count == 856
    assert set(index.directions_by_comparison[OFFICIAL_VS_PSEUDO]) == {
        "ces_Latn-deu_Latn",
        "eng_Latn-deu_Latn",
        "eng_Latn-ekk_Latn",
        "eng_Latn-isl_Latn",
        "zho_Hans-jpn_Jpan",
    }
    assert set(index.directions_by_comparison[NATIVE_VS_POSTEDIT]) == {
        "ces_Latn-deu_Latn",
        "ces_Latn-vie_Latn",
    }
    assert set(index.directions_by_comparison[NATIVE_VS_PSEUDO]) == {
        "ces_Latn-deu_Latn",
        "eng_Latn-ekk_Latn",
        "eng_Latn-isl_Latn",
    }
    assert set(index.directions_by_comparison[POSTEDIT_VS_PSEUDO]) == {
        "eng_Latn-deu_Latn",
        "zho_Hans-jpn_Jpan",
    }
    for directions in index.directions_by_comparison.values():
        for definition in directions.values():
            assert all(
                set(mirror.common_systems).isdisjoint(REFERENCE_CANDIDATE_SYSTEMS)
                for mirror in definition.mirrors
            )
    assert sum(
        direction.identical_reference_count
        for direction in index.directions_by_comparison[
            OFFICIAL_VS_PSEUDO
        ].values()
    ) == 51
    assert sum(
        direction.identical_reference_count
        for direction in index.directions_by_comparison[
            NATIVE_VS_POSTEDIT
        ].values()
    ) == 12


def test_paper_reference_quality_writes_only_selected_pdf(tmp_path):
    """The matched cohort produces a PDF with a numeric Markdown companion."""

    pytest.importorskip("matplotlib")
    submissions, test_set, references, template, index = _analysis_context(tmp_path)
    result = analyze_reference_quality(
        submissions, test_set, references, template, index
    )
    output = tmp_path / "out"
    path = write_paper_reference_quality_report(output, result)
    assert path == output / "plots/task1-reference-quality-effects.pdf"
    assert path.stat().st_size > 0
    assert set(output.rglob("*.*")) == {path, path.with_suffix(".md")}
    assert "Mean ΔF1" in path.with_suffix(".md").read_text()
