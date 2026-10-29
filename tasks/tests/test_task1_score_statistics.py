from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import wmt26_mteval_task1.scoring as scoring_module

from wmt26_mteval_task1.constants import EVALUATOR_CHARACTERISTIC_QUESTIONS
from wmt26_mteval_task1.io import (
    load_model_card_template,
    load_ground_truth,
    load_test_set,
)
from wmt26_mteval_task1.mpp_cache import MppCache
from wmt26_mteval_task1.models import ModelCardMetadata, SubmissionDescriptor
from wmt26_mteval_task1.score_statistics import (
    DirectionStatisticsRow,
    EvaluatorCharacteristics,
    LengthStatisticsRow,
    SpanStatistics,
    TokenLengthIndex,
    _build_correlations,
    _build_length_buckets,
    _build_length_summaries,
    _characteristic_values,
    _compute_record_mpp,
    _length_behavior_values,
    _spearman,
    analyze_score_statistics,
    load_evaluator_characteristics,
    load_token_lengths,
)
from wmt26_mteval_task1.score_statistics_reporting import write_paper_length_effect
from wmt26_mteval_task1.scoring import (
    MppComponents,
    PreparedAssessment,
    ScoredAssessment,
    score_submission,
)
from wmt26_mteval_task1.submissions import discover_submissions

from _task1_helpers import (
    assessment,
    item_id,
    major_span,
    model_card,
    model_card_template,
    write_json,
    write_jsonl,
)


def _template_and_card():
    template = model_card_template()
    card = model_card()
    for key, question in EVALUATOR_CHARACTERISTIC_QUESTIONS.items():
        if key == "reference_use":
            continue
        template.append({"question": question, "required": True, "answer": ""})
        card.append(
            {
                "question": question,
                "answer": "no" if key == "reference_use" else "false",
            }
        )
    return template, card


def _analysis_context(tmp_path, *, repeat_annotator: bool = False):
    official = item_id()
    test_path = tmp_path / "test.jsonl"
    ground_truth_path = tmp_path / "ground-truth.jsonl"
    template_path = tmp_path / "template.json"
    submission_path = tmp_path / "submissions" / "1"
    submission_path.mkdir(parents=True)

    write_jsonl(
        test_path,
        [
            {
                "item_id": official,
                "src": "<p>source</p> ",
                "ref": {"text": "reference", "type": "human"},
                "hyps": {"system": "abcdefghij"},
            }
        ],
    )
    write_jsonl(
        ground_truth_path,
        [
            {
                "item_id": official,
                "task1_pred": {
                    "system": [
                        {
                            **assessment([major_span(0, 1)]),
                            "annotator_id": "a",
                            "context_models": [],
                        },
                        {
                            **assessment(
                                [
                                    {
                                        "start": 0,
                                        "end": 3,
                                        "severity": "minor",
                                    },
                                    major_span(3, 5),
                                ],
                                omission="minor",
                            ),
                            "annotator_id": "a" if repeat_annotator else "b",
                            "context_models": [],
                        },
                    ]
                },
            }
        ],
    )
    template, card = _template_and_card()
    write_json(template_path, template)
    write_json(submission_path / "model_card.json", card)
    write_jsonl(
        submission_path / "predictions.jsonl",
        [
            {
                "item_id": official,
                "task1_pred": {
                    "system": assessment(
                        [
                            major_span(0, 2),
                            {"start": 2, "end": 5, "severity": "minor"},
                        ],
                        omission="major",
                    )
                },
            }
        ],
    )

    test_set = load_test_set(test_path)
    references = load_ground_truth(ground_truth_path, test_set)
    template = load_model_card_template(template_path)
    descriptor = discover_submissions(tmp_path / "submissions")[0][0]
    labels = EvaluatorCharacteristics(
        tmp_path / "characteristics.csv",
        {("submission", "1"): "reasoning"},
    )
    return descriptor, test_set, references, template, labels


def test_intrinsic_prediction_and_human_assessment_counting(tmp_path):
    descriptor, test_set, references, template, labels = _analysis_context(tmp_path)

    result = analyze_score_statistics(
        [descriptor],
        test_set,
        references,
        template,
        evaluator_characteristics=labels,
    )

    available = next(
        row for row in result.entry_statistics if row.cohort == "available"
    )
    prediction = available.prediction_statistics
    gold = available.gold_statistics

    assert prediction.pair_count == 1
    assert prediction.assessment_count == 1
    assert prediction.text_span_count == 2
    assert prediction.mean_text_span_length_characters == pytest.approx(2.5)
    assert prediction.total_error_count == 3
    assert prediction.minor_error_count == 1
    assert prediction.major_error_count == 2
    assert prediction.omission_count == 1
    assert prediction.mean_severity_weight == pytest.approx(0.75)

    assert gold.pair_count == 1
    assert gold.assessment_count == 2
    assert gold.text_span_count == 3
    assert gold.mean_text_span_length_characters == pytest.approx(2.0)
    assert gold.total_error_count == 4
    assert gold.minor_error_count == 2
    assert gold.major_error_count == 2
    assert gold.omission_count == 1

    assert available.aligned_assessments == 2
    assert result.scores[0].direction_scores[0].aligned_assessments == 2
    assert available.characteristics["reference_use"] == "reference_free"
    assert available.characteristics["reasoning"] == "reasoning"
    assert len(result.direction_statistics) == 1
    assert result.direction_statistics[0].prediction_statistics.assessment_count == 1

    redundancy = result.ground_truth_redundancy
    assert redundancy.total_pair_count == 1
    assert redundancy.total_assessment_count == 2
    assert redundancy.extra_assessment_count == 1
    assert redundancy.multiple_annotation_pair_count == 1
    assert redundancy.multiple_annotation_pair_percentage == pytest.approx(100.0)
    assert redundancy.multiple_distinct_annotator_pair_count == 1
    assert redundancy.multiple_distinct_annotator_pair_percentage == pytest.approx(
        100.0
    )
    assert redundancy.same_annotator_repeated_pair_count == 0
    assert redundancy.same_annotator_repeated_pair_percentage == pytest.approx(0.0)
    assert redundancy.maximum_assessments_per_pair == 2


def test_repeated_annotator_assessments_are_scored_and_reported(tmp_path):
    """Keep same-annotator references aligned and expose their redundancy."""

    descriptor, test_set, references, template, labels = _analysis_context(
        tmp_path, repeat_annotator=True
    )

    result = analyze_score_statistics(
        [descriptor],
        test_set,
        references,
        template,
        evaluator_characteristics=labels,
    )

    assert result.scores[0].direction_scores[0].aligned_assessments == 2
    assert result.ground_truth_redundancy.total_assessment_count == 2
    assert result.ground_truth_redundancy.multiple_distinct_annotator_pair_count == 0
    assert result.ground_truth_redundancy.same_annotator_repeated_pair_count == 1


def test_statistics_reuses_cache_created_by_official_scorer(monkeypatch, tmp_path):
    descriptor, test_set, references, template, labels = _analysis_context(tmp_path)
    cache_path = tmp_path / "shared.sqlite3"
    with MppCache(
        cache_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
    ) as cache:
        official = score_submission(
            descriptor, test_set, references, template, mpp_cache=cache
        )
        assert cache.statistics.computed_entries == 1

    def unexpected_compute(*args, **kwargs):
        raise AssertionError("statistics must reuse the official MPP cache")

    monkeypatch.setattr(scoring_module, "compute", unexpected_compute)
    with MppCache(
        cache_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
    ) as cache:
        statistics = analyze_score_statistics(
            [descriptor],
            test_set,
            references,
            template,
            evaluator_characteristics=labels,
            mpp_cache=cache,
        )
        assert cache.statistics.reused_entries == 1
        assert cache.statistics.computed_entries == 0

    assert statistics.scores[0].direction_scores == official.direction_scores


def test_token_loader_counts_exact_markup_without_truncation_or_special_tokens(
    tmp_path,
):
    _, test_set, references, _, _ = _analysis_context(tmp_path)

    class RecordingTokenizer:
        def __init__(self):
            self.calls = []
            self.truncation_disabled = False

        def no_truncation(self):
            """Record that the loader explicitly disabled token truncation."""

            self.truncation_disabled = True

        def encode_batch(self, texts, *, add_special_tokens):
            self.calls.append((list(texts), add_special_tokens))
            return [SimpleNamespace(ids=list(range(len(text)))) for text in texts]

    tokenizer = RecordingTokenizer()
    lengths = load_token_lengths(
        test_set.path, references, tokenizer=tokenizer, batch_size=1
    )

    assert lengths.source_lengths[next(iter(references.assessments))] == len(
        "<p>source</p> "
    )
    assert lengths.target_lengths[
        (next(iter(references.assessments)), "system")
    ] == len("abcdefghij")
    assert any("<p>source</p> " in texts for texts, _ in tokenizer.calls)
    assert all(add_special_tokens is False for _, add_special_tokens in tokenizer.calls)
    assert tokenizer.truncation_disabled


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("entry_type,entry_id\nsubmission,1\n", "header must be exactly"),
        (
            "entry_type,entry_id,reasoning\nsubmission,1,maybe\n",
            "reasoning must be",
        ),
        (
            "entry_type,entry_id,reasoning\n"
            "submission,1,reasoning\nsubmission,1,non_reasoning\n",
            "duplicate",
        ),
    ],
)
def test_reasoning_mapping_validation(tmp_path, content, message):
    path = tmp_path / "characteristics.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_evaluator_characteristics(path)


def test_xcomet_baselines_are_reported_as_supervised(tmp_path):
    """Override an xCOMET baseline's stale model-card supervision answer."""

    descriptor = SubmissionDescriptor(
        path=tmp_path / "xcomet_xl_ref_task1__deduplicated",
        submission_id="baseline",
        username="organizers",
        phase_id="",
        suffix="",
        is_baseline=True,
        baseline_family="deduplicated",
    )
    metadata = ModelCardMetadata(
        evaluator_characteristics={
            key: False for key in EVALUATOR_CHARACTERISTIC_QUESTIONS
        }
    )
    labels = EvaluatorCharacteristics(
        tmp_path / "characteristics.csv",
        {("baseline", "xcomet_xl_ref_task1"): "non_reasoning"},
    )

    values = _characteristic_values(descriptor, metadata, labels, [])

    assert values["supervised"] == "true"




def test_missing_reasoning_mapping_is_warning_not_failure(tmp_path):
    descriptor, test_set, references, template, _ = _analysis_context(tmp_path)

    result = analyze_score_statistics(
        [descriptor],
        test_set,
        references,
        template,
    )

    assert not result.failed
    assert any(
        issue.severity == "warning" and "reasoning metadata is unknown" in issue.message
        for issue in result.issues
    )
    available = next(
        row for row in result.entry_statistics if row.cohort == "available"
    )
    assert available.characteristics["reasoning"] == "unknown"




def test_relative_and_global_token_buckets_use_the_correct_units():
    records = [
        PreparedAssessment("item-one", "system-one", "eng-deu", 10, 5, [], [], 0),
        PreparedAssessment("item-one", "system-two", "eng-deu", 10, 7, [], [], 0),
        PreparedAssessment("item-two", "system-one", "eng-deu", 20, 9, [], [], 0),
    ]
    token_lengths = TokenLengthIndex(
        "fake",
        "revision",
        {"item-one": 10, "item-two": 20},
        {
            ("item-one", "system-one"): 5,
            ("item-one", "system-two"): 7,
            ("item-two", "system-one"): 9,
        },
    )

    definitions = _build_length_buckets(records, token_lengths, {"eng-deu"})

    relative_source = next(
        row
        for row in definitions
        if row.bucket_scope == "within_direction"
        and row.length_basis == "source_tokens"
    )
    relative_target = next(
        row
        for row in definitions
        if row.bucket_scope == "within_direction"
        and row.length_basis == "target_tokens"
    )
    relative_combined = next(
        row
        for row in definitions
        if row.bucket_scope == "within_direction"
        and row.length_basis == "source_target_tokens"
    )
    global_source = next(
        row
        for row in definitions
        if row.bucket_scope == "global_priority" and row.length_basis == "source_tokens"
    )
    assert relative_source.unit_count == 2
    assert relative_target.unit_count == 3
    assert relative_combined.unit_count == 3
    assert relative_combined.minimum == 15
    assert relative_combined.maximum == 29
    assert global_source.unit_count == 2


def test_length_bucket_mpp_weights_each_item_system_pair_once():
    records = [
        PreparedAssessment("item-one", "system", "eng-deu", 10, 5, [], [], 0),
        PreparedAssessment("item-two", "system", "eng-deu", 10, 5, [], [], 0),
        PreparedAssessment("item-two", "system", "eng-deu", 10, 5, [], [], 1),
        PreparedAssessment("item-three", "system", "eng-deu", 10, 5, [], [], 0),
        PreparedAssessment("item-three", "system", "eng-deu", 10, 5, [], [], 1),
        PreparedAssessment("item-three", "system", "eng-deu", 10, 5, [], [], 2),
    ]
    weights = [1.0, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3]
    scored = [
        ScoredAssessment(
            record,
            weight,
            MppComponents(
                1.0 if index == 0 else 0.0,
                1.0,
                1.0 if index == 0 else 0.0,
                1.0,
            ),
        )
        for index, (record, weight) in enumerate(zip(records, weights))
    ]

    result = _compute_record_mpp(scored)

    assert result == pytest.approx((1 / 3, 1, 6))


def test_spearman_handles_ties_and_undefined_inputs():
    coefficient, status = _spearman([1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0])
    assert coefficient == pytest.approx(1.0)
    assert status == "ok"

    coefficient, status = _spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert coefficient is None
    assert status == "constant_value"

    coefficient, status = _spearman([1.0, 2.0], [1.0, 2.0])
    assert coefficient is None
    assert status == "insufficient_observations"


def _ranked_span_statistics(rank):
    """Build internally consistent span statistics with three increasing metrics."""

    text_span_count = 10 * rank
    major_count = (1, 10, 27)[rank - 1]
    return SpanStatistics(
        pair_count=10,
        assessment_count=10,
        text_span_count=text_span_count,
        text_span_character_count=text_span_count * rank,
        omission_count=0,
        minor_error_count=text_span_count - major_count,
        major_error_count=major_count,
        minor_omission_count=0,
        major_omission_count=0,
    )


def test_correlations_are_per_direction_and_summary_is_direction_equal(tmp_path):
    descriptors = [
        SubmissionDescriptor(
            tmp_path / f"entry-{index}",
            str(index),
            f"user-{index}",
            "phase",
            "system",
            is_baseline=index == 3,
        )
        for index in range(1, 4)
    ]
    rows = []
    for direction, f1_values in (
        ("eng_Latn-ind_Latn", (0.1, 0.2, 0.3)),
        ("zho_Hans-jpn_Jpan", (0.3, 0.2, 0.1)),
    ):
        rows.extend(
            DirectionStatisticsRow(
                descriptor,
                direction,
                f1,
                _ranked_span_statistics(rank),
            )
            for rank, (descriptor, f1) in enumerate(
                zip(descriptors, f1_values), start=1
            )
        )

    correlations, summaries = _build_correlations(
        rows, ("eng_Latn-ind_Latn", "zho_Hans-jpn_Jpan")
    )

    assert len(correlations) == 6
    assert all(row.observation_count == 3 for row in correlations)
    assert all(row.status == "ok" for row in correlations)
    positive = [
        row.spearman_correlation
        for row in correlations
        if row.language_pair == "eng_Latn-ind_Latn"
    ]
    negative = [
        row.spearman_correlation
        for row in correlations
        if row.language_pair == "zho_Hans-jpn_Jpan"
    ]
    assert positive == pytest.approx([1.0, 1.0, 1.0])
    assert negative == pytest.approx([-1.0, -1.0, -1.0])
    for summary in summaries:
        assert summary.defined_direction_count == 2
        assert summary.total_direction_count == 2
        assert summary.median_correlation == pytest.approx(0.0)
        assert summary.first_quartile_correlation == pytest.approx(-0.5)
        assert summary.third_quartile_correlation == pytest.approx(0.5)


def test_correlations_report_an_undefined_predictor_without_dropping_observations(tmp_path):
    """Keep undefined observations visible while leaving other predictors usable."""

    descriptor = SubmissionDescriptor(
        tmp_path / "entry-zero", "0", "user-zero", "phase", "system"
    )
    no_errors = SpanStatistics(
        pair_count=1,
        assessment_count=1,
        text_span_count=0,
        text_span_character_count=0,
        omission_count=0,
        minor_error_count=0,
        major_error_count=0,
        minor_omission_count=0,
        major_omission_count=0,
    )
    correlations, summaries = _build_correlations(
        [DirectionStatisticsRow(descriptor, "eng-deu", 0.05, no_errors)],
        ("eng-deu",),
    )
    assert all(row.observation_count == 1 for row in correlations)
    assert correlations[0].status == "undefined_predictor"
    assert correlations[0].spearman_correlation is None
    assert correlations[0].undefined_evaluators == ("0",)
    assert correlations[1].status != "undefined_predictor"
    assert summaries[0].defined_direction_count == 0


def test_correlations_reject_non_finite_f1(tmp_path):
    """Correlation analysis requires finite F1 for every observation."""

    descriptor = SubmissionDescriptor(
        tmp_path / "entry", "1", "user", "phase", "system"
    )
    with pytest.raises(ValueError, match="finite F1"):
        _build_correlations(
            [
                DirectionStatisticsRow(
                    descriptor,
                    "eng-deu",
                    float("nan"),
                    _ranked_span_statistics(1),
                )
            ],
            ("eng-deu",),
        )


def test_length_summary_reports_median_and_iqr_across_evaluators(tmp_path):
    metadata = ModelCardMetadata()
    rows = [
        LengthStatisticsRow(
            SubmissionDescriptor(
                tmp_path / f"entry-{index}",
                str(index),
                f"user-{index}",
                "phase",
                "system",
            ),
            metadata,
            "within_direction",
            "source_tokens",
            "Q1",
            8,
            f1,
            10,
        )
        for index, f1 in enumerate((0.1, 0.2, 0.3, 0.4), start=1)
    ]

    summaries = _build_length_summaries(rows)
    summary = next(
        row
        for row in summaries
        if row.bucket_scope == "within_direction"
        and row.length_basis == "source_tokens"
        and row.length_bucket == "Q1"
    )

    assert summary.observation_count == 4
    assert summary.median_f1 == pytest.approx(0.25)
    assert summary.first_quartile_f1 == pytest.approx(0.175)
    assert summary.third_quartile_f1 == pytest.approx(0.325)


def test_length_behavior_uses_unique_predictions_and_direction_means():
    """Bucket behavior matches effects analysis despite duplicate gold or unequal LP sizes."""
    first = PreparedAssessment(
        "one", "system", "eng-deu", 10, 10,
        [{"start": 0, "end": 4, "severity": "minor"},
         {"start": 10, "end": 11, "severity": "major"}], [], 0,
    )
    second = PreparedAssessment(
        "two", "system", "eng-deu", 10, 10, [], [], 0,
    )
    third = PreparedAssessment(
        "three", "system", "eng-fra", 10, 10,
        [{"start": 0, "end": 8, "severity": "major"}], [], 0,
    )
    rows = [ScoredAssessment(record, 1.0, MppComponents(0, 1, 0, 1))
            for record in (first, replace(first, reference_index=1), second, third)]
    values = _length_behavior_values(rows)
    assert values["mean_total_errors_per_assessment"] == pytest.approx(1.0)
    assert values["mean_text_span_length_characters"] == pytest.approx(6.0)
    assert values["mean_severity_weight"] == pytest.approx(0.8125)
    assert all(value is None for value in _length_behavior_values([]).values())
    empty_values = _length_behavior_values([rows[2]])
    assert empty_values["mean_total_errors_per_assessment"] == 0
    assert empty_values["mean_text_span_length_characters"] is None
    assert empty_values["mean_severity_weight"] is None


def test_paper_length_figure_writes_only_selected_pdf(tmp_path):
    """Quartile medians and IQRs appear in both the PDF and Markdown table."""

    pytest.importorskip("matplotlib")
    rows = tuple(
        SimpleNamespace(
            bucket_scope="within_direction", length_basis="source_tokens",
            length_bucket=bucket, median_f1=0.34 - 0.02 * index,
            first_quartile_f1=0.25 - 0.01 * index,
            third_quartile_f1=0.38 - 0.01 * index,
        )
        for index, bucket in enumerate(("Q1", "Q2", "Q3", "Q4"))
    )
    path = write_paper_length_effect(
        tmp_path, SimpleNamespace(length_summaries=rows)
    )
    assert path.name == "task1-length-effect.pdf"
    assert path.stat().st_size > 0
    assert {item.name for item in tmp_path.iterdir()} == {
        path.name, path.with_suffix(".md").name,
    }
    markdown = path.with_suffix(".md").read_text()
    assert "| Q1 | 34.00 | 25.00 | 38.00 |" in markdown
    assert "shaded figure" not in markdown
    assert "F1 (%)" not in markdown
