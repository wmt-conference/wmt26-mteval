from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest
import wmt26_mteval_task1.scoring as scoring_module

from wmt26_mteval_task1.io import (
    load_model_card_template,
    load_ground_truth,
    load_test_set,
)
from wmt26_mteval_task1.models import (
    BatchScoreResult,
    CoverageReport,
    DirectionScore,
    ModelCardMetadata,
    SubmissionDescriptor,
    SubmissionScore,
)
from wmt26_mteval_task1.mpp_cache import MppCache
from wmt26_mteval_task1.reporting import write_paper_score_reports
from wmt26_mteval_task1.scoring import (
    PreparedAssessment,
    aggregate_mpp_components,
    scored_assessments_from_result,
    score_prepared_assessments,
    score_submission,
    score_submissions,
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


def scoring_context(tmp_path):
    official = item_id()
    test_path = tmp_path / "test.jsonl"
    ref_path = tmp_path / "ground-truth.jsonl"
    template_path = tmp_path / "template.json"
    write_jsonl(test_path, [{"item_id": official, "ref": {"text": "efgh"}, "hyps": {"system": "abcd"}}])
    write_jsonl(
        ref_path,
        [
            {
                "item_id": official,
                "task1_pred": {"system": assessment([major_span(0, 4)])},
            }
        ],
    )
    write_json(template_path, model_card_template())
    test_set = load_test_set(test_path)
    return (
        official,
        test_set,
        load_ground_truth(ref_path, test_set),
        load_model_card_template(template_path),
    )


def make_submission(tmp_path, submission_id, official, *, prediction=True, card=True):
    path = tmp_path / "submissions" / str(submission_id)
    path.mkdir(parents=True)
    if card:
        write_json(path / "model_card.json", model_card())
    if prediction:
        write_jsonl(
            path / "predictions.jsonl",
            [
                {
                    "item_id": official,
                    "task1_pred": {"system": assessment([major_span(0, 2)])},
                }
            ],
        )
    return path


def test_score_submission_uses_direction_micro_mpp(tmp_path):
    official, test_set, references, template = scoring_context(tmp_path)
    make_submission(tmp_path, "1", official)
    submission = discover_submissions(tmp_path / "submissions")[0][0]

    result = score_submission(submission, test_set, references, template)

    assert len(result.direction_scores) == 1
    assert result.direction_scores[0].precision == pytest.approx(1.0)
    assert result.direction_scores[0].recall == pytest.approx(0.5)
    assert result.direction_scores[0].f1 == pytest.approx(2 / 3)
    assert result.macro_f1 == pytest.approx(2 / 3)


def test_post_compute_segment_weighting_aggregates_raw_mpp_components():
    """Apply reciprocal assessment weights after one unweighted metric call."""

    records = (
        PreparedAssessment(
            "item-a",
            "system",
            "lp",
            None,
            8,
            [major_span(0, 4)],
            [major_span(0, 4)],
            0,
        ),
        PreparedAssessment(
            "item-a",
            "system",
            "lp",
            None,
            8,
            [major_span(0, 4)],
            [major_span(4, 8)],
            1,
        ),
        PreparedAssessment(
            "item-b",
            "system",
            "lp",
            None,
            8,
            [],
            [major_span(0, 4)],
            0,
        ),
    )
    result = scoring_module.compute(
        [record.prediction for record in records],
        [record.reference for record in records],
    )

    scores = aggregate_mpp_components(
        scored_assessments_from_result(records, (0.5, 0.5, 1.0), result)
    )

    assert scores == (0.5, 0.25, 1 / 3)


def test_mpp_cache_reuses_and_force_recomputes_components(monkeypatch, tmp_path):
    official, test_set, references, template = scoring_context(tmp_path)
    make_submission(tmp_path, "1", official)
    submission = discover_submissions(tmp_path / "submissions")[0][0]
    cache_path = tmp_path / "mpp.sqlite3"
    real_compute = scoring_module.compute
    calls = 0

    def counting_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(scoring_module, "compute", counting_compute)
    with MppCache(
        cache_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
    ) as cache:
        first = score_submission(
            submission, test_set, references, template, mpp_cache=cache
        )
        assert cache.statistics.computed_entries == 1
        assert cache.statistics.computed_segments == 1
    assert calls == 1

    def unexpected_compute(*args, **kwargs):
        raise AssertionError("MPP should have been loaded from cache")

    monkeypatch.setattr(scoring_module, "compute", unexpected_compute)
    with MppCache(
        cache_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
    ) as cache:
        second = score_submission(
            submission, test_set, references, template, mpp_cache=cache
        )
        assert cache.statistics.reused_entries == 1
        assert cache.statistics.reused_segments == 1
    assert second == first

    calls = 0
    monkeypatch.setattr(scoring_module, "compute", counting_compute)
    with MppCache(
        cache_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
    ) as cache:
        forced = score_submission(
            submission,
            test_set,
            references,
            template,
            mpp_cache=cache,
            recompute_mpp=True,
        )
        assert cache.statistics.reused_entries == 0
        assert cache.statistics.computed_entries == 1
    assert calls == 1
    assert forced == first


def test_mpp_cache_reuses_subsets_and_merges_recomputed_rows(monkeypatch, tmp_path):
    """Reuse a cached subset and update it without evicting other assessments."""

    test_path = tmp_path / "test.jsonl"
    reference_path = tmp_path / "ground-truth.jsonl"
    predictions_path = tmp_path / "entry" / "predictions.jsonl"
    write_jsonl(test_path, [{"item_id": item_id(), "hyps": {"system": "x"}}])
    write_jsonl(reference_path, [{"item_id": item_id(), "task1_pred": {}}])
    write_jsonl(predictions_path, [{"cache": "identity"}])
    descriptor = SubmissionDescriptor(
        predictions_path.parent, "1", "user", "1", "suffix"
    )
    records = tuple(
        PreparedAssessment(
            f"official_###_eng_Latn_###_deu_Latn_###_news_###_doc_###_{index}",
            "system",
            "eng_Latn-deu_Latn",
            1,
            1,
            [],
            [],
            0,
        )
        for index in range(2)
    )
    cache_path = tmp_path / "greedy" / "mpp_cache.sqlite3"
    with MppCache(
        cache_path,
        test_set_path=test_path,
        ground_truth_path=reference_path,
    ) as cache:
        score_prepared_assessments(descriptor, records, mpp_cache=cache)

    def unexpected_compute(*args, **kwargs):
        """Fail if a complete cache hit invokes the metric."""

        raise AssertionError("cached assessments should be reused")

    monkeypatch.setattr(scoring_module, "compute", unexpected_compute)
    with MppCache(
        cache_path,
        test_set_path=test_path,
        ground_truth_path=reference_path,
    ) as cache:
        score_prepared_assessments(descriptor, records[:1], mpp_cache=cache)
        assert cache.statistics.reused_segments == 1

    real_compute = __import__("span_mt_metrics_eval").compute
    monkeypatch.setattr(scoring_module, "compute", real_compute)
    with MppCache(
        cache_path,
        test_set_path=test_path,
        ground_truth_path=reference_path,
    ) as cache:
        score_prepared_assessments(
            descriptor, records[:1], mpp_cache=cache, recompute_mpp=True
        )

    monkeypatch.setattr(scoring_module, "compute", unexpected_compute)
    with MppCache(
        cache_path,
        test_set_path=test_path,
        ground_truth_path=reference_path,
    ) as cache:
        score_prepared_assessments(descriptor, records, mpp_cache=cache)
        assert cache.statistics.reused_segments == 2


def test_mpp_cache_invalidates_changed_predictions(monkeypatch, tmp_path):
    official, test_set, references, template = scoring_context(tmp_path)
    submission_path = make_submission(tmp_path, "1", official)
    submission = discover_submissions(tmp_path / "submissions")[0][0]
    cache_path = tmp_path / "mpp.sqlite3"
    with MppCache(
        cache_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
    ) as cache:
        score_submission(submission, test_set, references, template, mpp_cache=cache)

    write_jsonl(
        submission_path / "predictions.jsonl",
        [
            {
                "item_id": official,
                "task1_pred": {"system": assessment([major_span(0, 4)])},
            }
        ],
    )
    real_compute = scoring_module.compute
    calls = 0

    def counting_compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(scoring_module, "compute", counting_compute)
    with MppCache(
        cache_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
    ) as cache:
        refreshed = score_submission(
            submission, test_set, references, template, mpp_cache=cache
        )
        assert cache.statistics.reused_entries == 0
        assert cache.statistics.computed_entries == 1
    assert calls == 1
    assert refreshed.direction_scores[0].f1 == pytest.approx(1.0)


def test_mpp_cache_invalidates_changed_global_inputs(tmp_path):
    official, test_set, references, template = scoring_context(tmp_path)
    make_submission(tmp_path, "1", official)
    submission = discover_submissions(tmp_path / "submissions")[0][0]
    cache_path = tmp_path / "mpp.sqlite3"
    with MppCache(
        cache_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
    ) as cache:
        score_submission(submission, test_set, references, template, mpp_cache=cache)

    write_jsonl(
        references.path,
        [
            {
                "item_id": official,
                "task1_pred": {"system": assessment([major_span(0, 2)])},
            }
        ],
    )
    with MppCache(
        cache_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
    ) as cache:
        assert cache.statistics.invalidated_entries == 1


def test_mpp_cache_rejects_corrupt_database(tmp_path):
    official, test_set, references, _ = scoring_context(tmp_path)
    cache_path = tmp_path / "mpp.sqlite3"
    cache_path.write_text("not a SQLite database", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot initialize MPP cache"):
        MppCache(
            cache_path,
            test_set_path=test_set.path,
            ground_truth_path=references.path,
        )


def test_default_scoring_logs_hide_direction_progress(caplog, tmp_path):
    official, test_set, references, template = scoring_context(tmp_path)
    make_submission(tmp_path, "1", official)
    submission = discover_submissions(tmp_path / "submissions")[0][0]

    with caplog.at_level(logging.INFO):
        score_submission(submission, test_set, references, template)

    assert not any("scoring eng_Latn-deu_Latn" in row.message for row in caplog.records)


def test_multiple_references_repeat_prediction(monkeypatch, tmp_path):
    official = item_id()
    test_path = tmp_path / "test.jsonl"
    ref_path = tmp_path / "ground-truth.jsonl"
    template_path = tmp_path / "template.json"
    write_jsonl(test_path, [{"item_id": official, "hyps": {"system": "abcd"}}])
    write_jsonl(
        ref_path,
        [
            {
                "item_id": official,
                "task1_pred": {
                    "system": [
                        {
                            **assessment([major_span(0, 4)]),
                            "annotator_id": "a",
                            "context_models": [],
                        },
                        {
                            **assessment([major_span(2, 4)]),
                            "annotator_id": "b",
                            "context_models": [],
                        },
                    ]
                },
            }
        ],
    )
    write_json(template_path, model_card_template())
    make_submission(tmp_path, "1", official)
    captured = []

    def fake_compute(predictions, references, **options):
        captured.append((predictions, references, options))
        components = [
            SimpleNamespace(
                precision_numerator=1.0,
                precision_denominator=1.0,
                recall_numerator=1.0,
                recall_denominator=1.0,
            )
            for _ in predictions
        ]
        return SimpleNamespace(
            precision=1.0,
            recall=1.0,
            f_score=1.0,
            details=SimpleNamespace(segments_components=components),
        )

    monkeypatch.setattr("wmt26_mteval_task1.scoring.compute", fake_compute)
    test_set = load_test_set(test_path)
    score_submission(
        discover_submissions(tmp_path / "submissions")[0][0],
        test_set,
        load_ground_truth(ref_path, test_set),
        load_model_card_template(template_path),
    )

    assert len(captured[0][0]) == 2
    assert captured[0][2] == {
        "measure": "MPP",
        "matching": "one_to_one",
        "matching_algorithm": "greedy",
        "averaging": "micro",
        "severity_weights": {"minor": 0.25, "major": 1.0},
    }


def test_repeated_references_do_not_increase_pair_weight(tmp_path):
    first = item_id(document="first")
    second = item_id(document="second")
    test_path = tmp_path / "test.jsonl"
    ref_path = tmp_path / "ground-truth.jsonl"
    template_path = tmp_path / "template.json"
    write_jsonl(
        test_path,
        [
            {"item_id": first, "hyps": {"system": "abcd"}},
            {"item_id": second, "hyps": {"system": "abcd"}},
        ],
    )
    write_jsonl(
        ref_path,
        [
            {
                "item_id": first,
                "task1_pred": {
                    "system": [
                        {
                            **assessment([major_span(0, 4)]),
                            "annotator_id": "a",
                            "context_models": [],
                        },
                        {
                            **assessment([major_span(0, 4)]),
                            "annotator_id": "b",
                            "context_models": [],
                        },
                    ]
                },
            },
            {
                "item_id": second,
                "task1_pred": {"system": assessment([major_span(0, 4)])},
            },
        ],
    )
    write_json(template_path, model_card_template())
    submission_path = tmp_path / "submissions" / "1"
    submission_path.mkdir(parents=True)
    write_json(submission_path / "model_card.json", model_card())
    write_jsonl(
        submission_path / "predictions.jsonl",
        [
            {
                "item_id": first,
                "task1_pred": {"system": assessment([])},
            },
            {
                "item_id": second,
                "task1_pred": {"system": assessment([major_span(0, 4)])},
            },
        ],
    )
    test_set = load_test_set(test_path)

    result = score_submission(
        discover_submissions(tmp_path / "submissions")[0][0],
        test_set,
        load_ground_truth(ref_path, test_set),
        load_model_card_template(template_path),
    )

    direction = result.direction_scores[0]
    assert direction.precision == pytest.approx(1.0)
    assert direction.recall == pytest.approx(0.5)
    assert direction.f1 == pytest.approx(2 / 3)
    assert direction.aligned_assessments == 3




def _report_score(
    tmp_path,
    submission_id,
    *,
    priority_f1,
    other_f1=None,
    full_official=False,
    baseline_name=None,
):
    is_baseline = baseline_name is not None
    descriptor = SubmissionDescriptor(
        path=(
            tmp_path / baseline_name
            if is_baseline
            else tmp_path / str(submission_id)
        ),
        submission_id="" if is_baseline else str(submission_id),
        username="baseline" if is_baseline else "user{}".format(submission_id),
        phase_id="" if is_baseline else "29413",
        suffix="" if is_baseline else "date_system",
        is_baseline=is_baseline,
    )
    coverage = CoverageReport(
        complete_directions=(),
        partial_directions=(),
        absent_directions=(),
        provided_pair_counts={},
        expected_pair_counts={},
        official_rows=0,
        challenge_rows=0,
        total_rows=0,
        all_official_complete=full_official,
        all_primary_complete=True,
        primary_only=other_f1 is None,
    )
    directions = [
        DirectionScore(
            language_pair="eng_Latn-ind_Latn",
            precision=priority_f1,
            recall=priority_f1,
            f1=priority_f1,
            aligned_assessments=10,
        )
    ]
    if other_f1 is not None:
        directions.append(
            DirectionScore(
                language_pair="eng_Latn-deu_Latn",
                precision=other_f1,
                recall=other_f1,
                f1=other_f1,
                aligned_assessments=20,
            )
        )
    return SubmissionScore(
        descriptor=descriptor,
        model_card=ModelCardMetadata(
            organization_name="Site",
            submission_name="System {}".format(submission_id),
            submission_description="A detector based on Qwen3-14B.",
            model_size="14B",
            api_provider="Test API",
            evaluator_characteristics={
                "reference_use": True,
                "supervised": False,
                "ensemble": False,
                "open_weight": True,
                "open_source": False,
            },
            valid=True,
        ),
        coverage=coverage,
        direction_scores=tuple(directions),
    )


















def test_selected_direction_must_be_complete(tmp_path):
    official, _, references, template = scoring_context(tmp_path)
    second = item_id(document="second")
    test_path = tmp_path / "test-two.jsonl"
    write_jsonl(
        test_path,
        [
            {"item_id": official, "hyps": {"system": "abcd"}},
            {"item_id": second, "hyps": {"system": "abcd"}},
        ],
    )
    test_set = load_test_set(test_path)
    make_submission(tmp_path, "1", official)

    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        score_submission(
            discover_submissions(tmp_path / "submissions")[0][0],
            test_set,
            references,
            template,
        )


def test_optimal_matching_is_forwarded_to_mpp(monkeypatch, tmp_path):
    official, test_set, references, template = scoring_context(tmp_path)
    make_submission(tmp_path, "1", official)
    submission = discover_submissions(tmp_path / "submissions")[0][0]
    captured = []

    def fake_compute(predictions, references, **options):
        captured.append(options)
        component = SimpleNamespace(
            precision_numerator=1.0,
            precision_denominator=1.0,
            recall_numerator=1.0,
            recall_denominator=1.0,
        )
        return SimpleNamespace(details=SimpleNamespace(segments_components=[component]))

    monkeypatch.setattr(scoring_module, "compute", fake_compute)
    score_submission(
        submission,
        test_set,
        references,
        template,
        matching_algorithm="optimal",
    )

    assert captured[0]["matching_algorithm"] == "optimal"


def test_mpp_cache_keeps_matching_algorithms_in_separate_files(monkeypatch, tmp_path):
    official, test_set, references, template = scoring_context(tmp_path)
    make_submission(tmp_path, "1", official)
    submission = discover_submissions(tmp_path / "submissions")[0][0]
    greedy_path = tmp_path / "greedy" / "mpp_cache.sqlite3"
    optimal_path = tmp_path / "optimal" / "mpp_cache.sqlite3"
    calls = []

    def fake_compute(predictions, references, **options):
        calls.append(options["matching_algorithm"])
        component = SimpleNamespace(
            precision_numerator=1.0 if options["matching_algorithm"] == "greedy" else 2.0,
            precision_denominator=2.0,
            recall_numerator=1.0,
            recall_denominator=2.0,
        )
        return SimpleNamespace(details=SimpleNamespace(segments_components=[component]))

    monkeypatch.setattr(scoring_module, "compute", fake_compute)
    with MppCache(
        greedy_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
        matching_algorithm="greedy",
    ) as greedy_cache, MppCache(
        optimal_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
        matching_algorithm="optimal",
    ) as optimal_cache:
        greedy = score_submission(
            submission, test_set, references, template,
            mpp_cache=greedy_cache, matching_algorithm="greedy",
        )
        optimal = score_submission(
            submission, test_set, references, template,
            mpp_cache=optimal_cache, matching_algorithm="optimal",
        )
        assert greedy_cache.statistics.computed_entries == 1
        assert optimal_cache.statistics.computed_entries == 1
        assert greedy.direction_scores[0].precision == pytest.approx(0.5)
        assert optimal.direction_scores[0].precision == pytest.approx(1.0)

    assert calls == ["greedy", "optimal"]

    def unexpected_compute(*args, **kwargs):
        raise AssertionError("both algorithm cache entries should be reusable")

    monkeypatch.setattr(scoring_module, "compute", unexpected_compute)
    with MppCache(
        greedy_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
        matching_algorithm="greedy",
    ) as greedy_cache, MppCache(
        optimal_path,
        test_set_path=test_set.path,
        ground_truth_path=references.path,
        matching_algorithm="optimal",
    ) as optimal_cache:
        score_submission(
            submission, test_set, references, template,
            mpp_cache=greedy_cache, matching_algorithm="greedy",
        )
        score_submission(
            submission, test_set, references, template,
            mpp_cache=optimal_cache, matching_algorithm="optimal",
        )
        assert greedy_cache.statistics.reused_entries == 1
        assert optimal_cache.statistics.reused_entries == 1


def test_batch_failure_keeps_paper_rankings_available(tmp_path):
    """One invalid submission does not prevent writing the two selected tables."""

    official, test_set, references, template = scoring_context(tmp_path)
    make_submission(tmp_path, "1", official)
    make_submission(tmp_path, "2", official, card=False)
    submissions = discover_submissions(tmp_path / "submissions")[0]
    result = score_submissions(submissions, test_set, references, template)
    paths = write_paper_score_reports(tmp_path / "out", result, None)
    assert result.failed and len(result.scores) == 1
    assert {path.name for path in paths} == {
        "priority_ranking.tex", "per_direction_rankings.tex",
    }
    assert "Test System" in (tmp_path / "out" / "per_direction_rankings.tex").read_text()


def test_paper_ranking_tables_style_and_model_aliases(tmp_path):
    """The paper tables retain ranking order, labels, and compact model names."""

    baseline = _report_score(
        tmp_path, 0, priority_f1=0.9, full_official=True,
        baseline_name="gemma_ref_task1",
    )
    participant = _report_score(
        tmp_path, 1, priority_f1=0.8, full_official=True,
    )
    participant = replace(
        participant,
        model_card=replace(
            participant.model_card,
            submission_name="System_1 & Co",
            closed_source_model_name="google/gemma-4-31B-it",
        ),
    )
    paths = write_paper_score_reports(
        tmp_path / "out",
        BatchScoreResult(scores=(baseline, participant), issues=()),
        None,
    )
    assert {path.name for path in paths} == {
        "priority_ranking.tex", "per_direction_rankings.tex",
    }
    priority = paths[0].read_text()
    assert "$90.00$" in priority and "$80.00$" in priority
    assert priority.index("$90.00$") < priority.index("$80.00$")
    assert r"System\_1 \& Co" in priority
    assert "Gemma 4 31B IT" in priority
    assert r"\label{tab:task1-priority-results}" in priority
