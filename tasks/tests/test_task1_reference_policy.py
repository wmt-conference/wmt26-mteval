"""Regression tests for direction-level translation-reference policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmt26_mteval_task1.io import (
    load_model_card_template,
    load_ground_truth,
    load_test_set,
)
from wmt26_mteval_task1.parquet_source import Task1ParquetSource
from wmt26_mteval_task1.score_statistics import analyze_score_statistics
from wmt26_mteval_task1.scoring import score_submission, score_submissions
from wmt26_mteval_task1.models import SubmissionDescriptor

from _task1_helpers import (
    assessment,
    item_id,
    major_span,
    model_card,
    model_card_template,
    write_json,
    write_jsonl,
)


EXPECTED_REAL_REFERENCE_SYSTEMS = {
    "ces_Latn-deu_Latn": "Human (from scratch)",
    "ces_Latn-vie_Latn": "Human (from scratch)",
    "eng_Latn-arz_Arab": "Human (postediting)",
    "eng_Latn-bel_Cyrl": "PseudoRef",
    "eng_Latn-ces_Latn": "Human (postediting)",
    "eng_Latn-deu_Latn": "Human (postediting)",
    "eng_Latn-ekk_Latn": "Human (from scratch)",
    "eng_Latn-hye_Armn": "PseudoRef",
    "eng_Latn-ind_Latn": "PseudoRef",
    "eng_Latn-isl_Latn": "Human (from scratch)",
    "eng_Latn-kaz_Cyrl": "PseudoRef",
    "eng_Latn-lij_Latn": "Human (from scratch)",
    "eng_Latn-lld_Latn": "Human (from scratch)",
    "eng_Latn-rus_Cyrl": "PseudoRef",
    "eng_Latn-sme_Latn": "Human (from scratch)",
    "eng_Latn-tha_Thai": "PseudoRef",
    "eng_Latn-ukr_Cyrl": "PseudoRef",
    "eng_Latn-zho_Hans": "PseudoRef",
    "eng_Latn-zho_Hant_TW": "PseudoRef",
    "zho_Hans-jpn_Jpan": "Human (postediting)",
}


def _ces_ukr_item_id() -> str:
    """Return one canonical official ID for the no-reference direction."""

    return "official_###_ces_Latn_###_ukr_Cyrl_###_news_###_document_###_0"


def _descriptor(
    tmp_path: Path,
    submission_id: str,
    *,
    reference_based: bool,
    predictions: list[dict[str, object]],
) -> SubmissionDescriptor:
    """Write one evaluator entry and return its scorer descriptor."""

    path = tmp_path / "submissions" / str(submission_id)
    path.mkdir(parents=True)
    write_json(
        path / "model_card.json",
        model_card(reference_use="true" if reference_based else "false"),
    )
    write_jsonl(path / "predictions.jsonl", predictions)
    return SubmissionDescriptor(
        path,
        submission_id,
        "user",
        "29413",
        "date_system",
    )


def _policy_context(tmp_path: Path):
    """Build a reference-backed direction plus ces_Latn-ukr_Cyrl fixtures."""

    with_reference = item_id(document="with-reference")
    without_reference = _ces_ukr_item_id()
    test_path = tmp_path / "test.jsonl"
    ground_truth_path = tmp_path / "ground-truth.jsonl"
    template_path = tmp_path / "template.json"
    write_jsonl(
        test_path,
        [
            {
                "item_id": with_reference,
                "ref": {"text": "reference", "type": "human"},
                "hyps": {
                    "reference-system": "reference",
                    "system": "abcd",
                },
            },
            {
                "item_id": without_reference,
                "ref": {"text": None, "type": None},
                "hyps": {"system": "wxyz"},
            },
        ],
    )
    write_jsonl(
        ground_truth_path,
        [
            {
                "item_id": with_reference,
                "task1_pred": {
                    "reference-system": assessment([major_span(0, 9)]),
                    "system": assessment([major_span(0, 4)]),
                },
            },
            {
                "item_id": without_reference,
                "task1_pred": {
                    "system": assessment([major_span(0, 4)]),
                },
            },
        ],
    )
    write_json(template_path, model_card_template())
    test_set = load_test_set(test_path)
    references = load_ground_truth(
        ground_truth_path,
        test_set,
        require_complete=True,
    )
    template = load_model_card_template(template_path)
    return with_reference, without_reference, test_set, references, template


def test_real_test_set_reference_policy_matches_reviewed_mapping():
    """Keep the reviewed real-data mapping and no-reference direction stable."""

    repository = Path(__file__).resolve().parents[1]
    parquet = repository / "wmt26-grand.parquet"
    if not parquet.is_file():
        pytest.skip("integrated WMT26 parquet file is unavailable")
    source = Task1ParquetSource(parquet, repository / "model_cards/task1")
    with source.activate():
        index = load_test_set(source.TEST_SET)

    assert index.reference_systems_by_direction == EXPECTED_REAL_REFERENCE_SYSTEMS
    assert index.official_directions - index.reference_available_directions == {
        "ces_Latn-ukr_Cyrl"
    }


def test_reference_system_requires_exact_match_on_every_direction_row(tmp_path):
    """Ignore incidental row matches and reject multiple consistent systems."""

    first = item_id(document="first")
    second = item_id(document="second")
    path = tmp_path / "test.jsonl"
    write_jsonl(
        path,
        [
            {
                "item_id": first,
                "ref": {"text": "reference", "type": "human"},
                "hyps": {"reference-system": "reference", "incidental": "reference"},
            },
            {
                "item_id": second,
                "ref": {"text": "other reference", "type": "human"},
                "hyps": {
                    "reference-system": "other reference",
                    "incidental": "different",
                },
            },
        ],
    )

    assert load_test_set(path).reference_systems_by_direction == {
        "eng_Latn-deu_Latn": "reference-system"
    }

    write_jsonl(
        path,
        [
            {
                "item_id": first,
                "ref": {"text": "reference", "type": "human"},
                "hyps": {"one": "reference", "two": "reference"},
            },
            {
                "item_id": second,
                "ref": {"text": "other reference", "type": "human"},
                "hyps": {"one": "other reference", "two": "other reference"},
            },
        ],
    )
    with pytest.raises(ValueError, match="multiple systems"):
        load_test_set(path)


def test_reference_policy_controls_scoring_coverage_and_warnings(tmp_path):
    """Score evaluator types on their applicable pairs and discard no-ref rows."""

    with_reference, without_reference, test_set, references, template = _policy_context(
        tmp_path
    )
    assert set(references.assessments[with_reference]) == {"system"}
    assert set(references.assessments[without_reference]) == {"system"}

    reference_free = _descriptor(
        tmp_path,
        "1",
        reference_based=False,
        predictions=[
            {
                "item_id": with_reference,
                "task1_pred": {
                    "reference-system": assessment(),
                    "system": assessment([major_span(0, 4)]),
                },
            },
            {
                "item_id": without_reference,
                "task1_pred": {"system": assessment([major_span(0, 4)])},
            },
        ],
    )
    free_score = score_submission(
        reference_free,
        test_set,
        references,
        template,
    )
    assert {row.language_pair for row in free_score.direction_scores} == {
        "eng_Latn-deu_Latn",
        "ces_Latn-ukr_Cyrl",
    }
    assert free_score.coverage.all_official_complete
    assert not free_score.issues

    missing_reference_prediction = _descriptor(
        tmp_path,
        "2",
        reference_based=False,
        predictions=[
            {
                "item_id": with_reference,
                "task1_pred": {
                    "system": assessment([major_span(0, 4)]),
                },
            },
            {
                "item_id": without_reference,
                "task1_pred": {"system": assessment([major_span(0, 4)])},
            },
        ],
    )
    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        score_submission(
            missing_reference_prediction,
            test_set,
            references,
            template,
        )

    reference_based = _descriptor(
        tmp_path,
        "3",
        reference_based=True,
        predictions=[
            {
                "item_id": with_reference,
                "task1_pred": {
                    "system": assessment([major_span(0, 4)]),
                },
            },
            {
                "item_id": without_reference,
                "task1_pred": "discard before annotation validation",
            },
        ],
    )
    based_score = score_submission(
        reference_based,
        test_set,
        references,
        template,
    )
    assert [row.language_pair for row in based_score.direction_scores] == [
        "eng_Latn-deu_Latn"
    ]
    assert based_score.coverage.all_official_complete
    assert len(based_score.issues) == 1
    assert based_score.issues[0].severity == "warning"
    assert "discarded 1 official prediction row" in based_score.issues[0].message
    assert "ces_Latn-ukr_Cyrl" in based_score.issues[0].message

    batch = score_submissions(
        [reference_based],
        test_set,
        references,
        template,
    )
    assert not batch.failed
    assert batch.issues == based_score.issues


def test_score_statistics_use_only_policy_scorable_pairs(tmp_path):
    """Exclude reference translations from every gold and evaluator statistic."""

    with_reference, without_reference, test_set, references, template = _policy_context(
        tmp_path
    )
    descriptor = _descriptor(
        tmp_path,
        "1",
        reference_based=False,
        predictions=[
            {
                "item_id": with_reference,
                "task1_pred": {
                    "reference-system": assessment(),
                    "system": assessment([major_span(0, 4)]),
                },
            },
            {
                "item_id": without_reference,
                "task1_pred": {"system": assessment([major_span(0, 4)])},
            },
        ],
    )

    result = analyze_score_statistics(
        [descriptor],
        test_set,
        references,
        template,
    )

    assert result.ground_truth_redundancy.total_pair_count == 2
    gold_all = next(
        row for row in result.gold_statistics if row.cohort == "all_official"
    )
    assert gold_all.statistics.pair_count == 2
    available = next(
        row for row in result.entry_statistics if row.cohort == "available"
    )
    assert available.prediction_statistics.pair_count == 2
    assert available.gold_statistics.pair_count == 2
    assert result.reference_systems_by_direction == {
        "eng_Latn-deu_Latn": "reference-system"
    }
    assert result.directions_without_references == ("ces_Latn-ukr_Cyrl",)
