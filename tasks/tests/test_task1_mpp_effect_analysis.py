"""Tests for matched Task 1 baseline MPP effects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.analyze_task1_reference_thinking_effects import build_parser
from wmt26_mteval_task1.constants import PRIMARY_REFERENCE_TYPE_BY_DIRECTION
from wmt26_mteval_task1.models import (
    BatchScoreResult,
    DirectionScore,
    ModelCardMetadata,
)
from wmt26_mteval_task1.mpp_effect_analysis import (
    BASELINE_PAIR_SPECS,
    PRIORITY_DIRECTIONS,
    REFERENCE_EFFECT,
    _overall_effects,
    analyze_mpp_effects,
    infer_reasoning_label,
)
from wmt26_mteval_task1.mpp_effect_analysis_reporting import (
    _CESA_SCORES_BY_DIRECTION,
    _mean_direction_f1_effects,
    write_paper_mpp_effect_reports,
)
from wmt26_mteval_task1.score_statistics import SpanStatistics


def _fake_score(name: str, value: float, reference_based: bool, description: str):
    """Build a minimal scored-entry object for analysis-unit tests."""

    directions = tuple(
        DirectionScore(direction, value, value + 0.01, value + 0.02, 10)
        for direction in PRIORITY_DIRECTIONS
    ) + (DirectionScore("extra-direction", 0.99, 0.99, 0.99, 10),)
    metadata = ModelCardMetadata(
        submission_name=name,
        submission_description=description,
        evaluator_characteristics={"reference_use": reference_based},
        valid=True,
    )
    descriptor = SimpleNamespace(
        path=Path("baselines") / name,
        report_label=f"Baseline: {name}",
    )
    return SimpleNamespace(
        descriptor=descriptor,
        model_card=metadata,
        direction_scores=directions,
        coverage=SimpleNamespace(all_primary_complete=True),
        issues=(),
    )


def _fake_scores():
    """Build the twelve endpoints of all ten matched comparisons."""

    descriptions = {
        "gemma_ref_no_thinking_task1": "Reference-based Gemma 4 WITHOUT reasoning enabled.",
        "qwen_ref_no_thinking_task1": "Reference-based Qwen 3.6 WITHOUT reasoning enabled.",
        "gemma_task1": "Reference-free Gemma 4 with reasoning enabled.",
        "gemma_ref_task1": "Reference-based Gemma 4 with reasoning enabled.",
        "gemma4_no_thinking_task1": "Reference-free Gemma 4 WITHOUT reasoning enabled.",
        "qwen_task1": "Reference-free Qwen 3.6 with reasoning enabled.",
        "qwen_ref_task1": "Reference-based Qwen 3.6 with reasoning enabled.",
        "qwen36_no_thinking_task1": "Reference-free Qwen 3.6 WITHOUT reasoning enabled.",
        "xcomet_xl_no_ref_task1": "xCOMET XL without references.",
        "xcomet_xl_ref_task1": "xCOMET XL with references.",
        "xcomet_xxl_no_ref_task1": "xCOMET XXL without references.",
        "xcomet_xxl_ref_task1": "xCOMET XXL with references.",
    }
    reference_based = {
        "gemma_ref_no_thinking_task1",
        "qwen_ref_no_thinking_task1",
        "gemma_ref_task1",
        "qwen_ref_task1",
        "xcomet_xl_ref_task1",
        "xcomet_xxl_ref_task1",
    }
    values = {name: 0.1 + index / 100 for index, name in enumerate(descriptions)}
    return tuple(
        _fake_score(name, values[name], name in reference_based, descriptions[name])
        for name in descriptions
    )


def _fake_prediction_statistics(scores):
    """Build deterministic per-direction behavior summaries for fake scores."""

    output = {}
    for index, score in enumerate(scores, start=1):
        text_spans = 10 + index
        omissions = 2
        total_errors = text_spans + omissions
        major_errors = index % total_errors
        statistics = SpanStatistics(
            pair_count=10,
            assessment_count=10,
            text_span_count=text_spans,
            text_span_character_count=2 * text_spans,
            omission_count=omissions,
            minor_error_count=total_errors - major_errors,
            major_error_count=major_errors,
            minor_omission_count=1,
            major_omission_count=1,
        )
        output[score.descriptor.path.name] = {
            direction: statistics for direction in PRIORITY_DIRECTIONS
        }
    return output


def _patch_analysis(monkeypatch, scores):
    """Patch scoring and prediction summaries for an isolated analysis test."""

    monkeypatch.setattr(
        "wmt26_mteval_task1.mpp_effect_analysis.score_submissions",
        lambda *args, **kwargs: BatchScoreResult(scores, ()),
    )
    monkeypatch.setattr(
        "wmt26_mteval_task1.mpp_effect_analysis.collect_prediction_statistics",
        lambda *args, **kwargs: _fake_prediction_statistics(scores),
    )


def test_reasoning_inference_prefers_explicit_negative_phrase():
    """Recognize the model-card wording used by current no-thinking baselines."""

    assert infer_reasoning_label("Reasoning enabled.", "gemma_task1") == (
        "reasoning",
        "model-card description",
    )
    assert (
        infer_reasoning_label("WITHOUT reasoning enabled.", "gemma4_no_thinking_task1")[
            0
        ]
        == "non_reasoning"
    )
    assert infer_reasoning_label(
        "xCOMET XL without references.", "xcomet_xl_no_ref_task1"
    ) == (
        "non_reasoning",
        "inferred dedicated metric model",
    )


def test_reasoning_inference_rejects_unknown_non_xcomet_model():
    """Require explicit model-card evidence for ordinary language models."""

    with pytest.raises(ValueError, match="does not identify reasoning"):
        infer_reasoning_label("A language evaluator.", "new_baseline")


def test_effect_analysis_uses_only_priority_directions(monkeypatch):
    """Ignore an extra scored direction and pass the fixed priority scope to scoring."""

    scores = _fake_scores()
    observed = {}

    def fake_score_submissions(*args, **kwargs):
        """Record the scorer scope and return synthetic successful scores."""

        observed["directions"] = kwargs["directions"]
        return BatchScoreResult(scores, ())

    monkeypatch.setattr(
        "wmt26_mteval_task1.mpp_effect_analysis.score_submissions",
        fake_score_submissions,
    )
    monkeypatch.setattr(
        "wmt26_mteval_task1.mpp_effect_analysis.collect_prediction_statistics",
        lambda *args, **kwargs: _fake_prediction_statistics(scores),
    )
    result = analyze_mpp_effects(
        (),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert observed["directions"] == PRIORITY_DIRECTIONS
    assert len(result.pair_effects) == len(BASELINE_PAIR_SPECS)
    assert all(row.direction_count == 8 for row in result.pair_effects)
    assert not result.failed
    assert len(result.pair_effects) == 10
    assert sum(row.effect == "reference_effect" for row in result.pair_effects) == 6
    assert sum(row.effect == "thinking_effect" for row in result.pair_effects) == 4


def test_invalid_reference_setting_excludes_affected_pairs(monkeypatch):
    """Wrong reference declarations must not silently contribute to matched effects."""

    scores = _fake_scores()
    bad = next(
        score
        for score in scores
        if score.descriptor.path.name == "gemma_ref_no_thinking_task1"
    )
    bad.model_card.evaluator_characteristics["reference_use"] = False

    def fake_score_submissions(*args, **kwargs):
        """Return scored endpoints with one deliberately incompatible card."""
        return BatchScoreResult(scores, ())

    monkeypatch.setattr(
        "wmt26_mteval_task1.mpp_effect_analysis.score_submissions",
        fake_score_submissions,
    )
    monkeypatch.setattr(
        "wmt26_mteval_task1.mpp_effect_analysis.collect_prediction_statistics",
        lambda *args, **kwargs: _fake_prediction_statistics(scores),
    )
    result = analyze_mpp_effects(
        (), SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )
    assert result.failed
    assert not (
        {"gemma4_non_reasoning", "gemma4_reference_based"}
        & {row.pair_id for row in result.pair_effects}
    )
    assert len(result.pair_effects) == 8


def test_overall_effects_average_pair_deltas_equally():
    """Average complete matched pairs rather than aligned-assessment counts."""

    first = SimpleNamespace(
        effect="effect",
        control_values=(0.0, 0.0, 0.0),
        treatment_values=(1.0, 1.0, 1.0),
        delta_values=(1.0, 1.0, 1.0),
        control_behavior=(1.0, 2.0, 0.25),
        treatment_behavior=(2.0, 4.0, 1.0),
        delta_behavior=(1.0, 2.0, 0.75),
    )
    second = SimpleNamespace(
        effect="effect",
        control_values=(0.0, 0.0, 0.0),
        treatment_values=(0.0, 0.0, 0.0),
        delta_values=(0.0, 0.0, 0.0),
        control_behavior=(2.0, 4.0, 1.0),
        treatment_behavior=(2.0, 4.0, 1.0),
        delta_behavior=(0.0, 0.0, 0.0),
    )
    result = _overall_effects((first, second))

    assert result[0].pair_count == 2
    assert result[0].delta_values == pytest.approx((0.5, 0.5, 0.5))
    assert result[0].delta_behavior == pytest.approx((0.5, 1.0, 0.375))


def test_cli_defaults_to_greedy_and_effect_output_root():
    """Expose cache, algorithm, and report-root defaults for the new command."""

    arguments = build_parser().parse_args([])

    assert arguments.matching_algorithm == "greedy"
    assert str(arguments.output_dir) == "outputs/task1/reference_thinking_effects"
    assert arguments.baseline_selectors is None




def test_reference_direction_means_can_exclude_xcomet_pairs(monkeypatch):
    """Produce distinct all-pair and non-xCOMET reference-effect series."""

    scores = _fake_scores()
    _patch_analysis(monkeypatch, scores)
    result = analyze_mpp_effects(
        (), SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )

    all_pairs = _mean_direction_f1_effects(result, REFERENCE_EFFECT)
    without_xcomet = _mean_direction_f1_effects(
        result, REFERENCE_EFFECT, exclude_xcomet=True
    )

    assert [direction for direction, _ in all_pairs] == list(PRIORITY_DIRECTIONS)
    assert [direction for direction, _ in without_xcomet] == list(
        PRIORITY_DIRECTIONS
    )
    assert all_pairs[0][1] != without_xcomet[0][1]


def test_priority_reference_types_cover_every_plotted_direction():
    """Provide one supported display category for every priority direction."""

    assert set(PRIMARY_REFERENCE_TYPE_BY_DIRECTION) == set(PRIORITY_DIRECTIONS)
    assert set(PRIMARY_REFERENCE_TYPE_BY_DIRECTION.values()) == {
        "Native",
        "Post-edited",
        "Pseudo",
    }
    assert set(_CESA_SCORES_BY_DIRECTION) == set(PRIORITY_DIRECTIONS)
    assert _CESA_SCORES_BY_DIRECTION["ces_Latn-deu_Latn"] == 80.7
    assert _CESA_SCORES_BY_DIRECTION["zho_Hans-jpn_Jpan"] == 79.6


def test_effect_analysis_excludes_endpoint_failing_coverage(monkeypatch):
    """Keep diagnostic scores from leaking into matched effects after a coverage exclusion."""

    scores = _fake_scores()
    bad = scores[0]
    bad.coverage.all_primary_complete = False

    def fake_score_submissions(*args, **kwargs):
        """Return one scored endpoint whose termination gate failed."""

        return BatchScoreResult(scores, ())

    monkeypatch.setattr(
        "wmt26_mteval_task1.mpp_effect_analysis.score_submissions",
        fake_score_submissions,
    )
    monkeypatch.setattr(
        "wmt26_mteval_task1.mpp_effect_analysis.collect_prediction_statistics",
        lambda *args, **kwargs: _fake_prediction_statistics(scores),
    )
    result = analyze_mpp_effects(
        (), SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )
    excluded = {
        spec.pair_id
        for spec in BASELINE_PAIR_SPECS
        if bad.descriptor.path.name in (spec.control_baseline, spec.treatment_baseline)
    }
    assert not excluded & {row.pair_id for row in result.pair_effects}
    assert not result.failed


def test_paper_effect_reports_only_selected_artifacts(tmp_path, monkeypatch):
    """Matched effects yield three paper artifacts and three Markdown companions."""

    pytest.importorskip("matplotlib")
    import wmt26_mteval_task1.mpp_effect_analysis as analysis

    _patch_analysis(monkeypatch, _fake_scores())
    result = analysis.analyze_mpp_effects(
        (), SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )
    paths = write_paper_mpp_effect_reports(tmp_path, result)
    assert {path.name for path in paths} == {
        "task1-reference-thinking-effect.tex",
        "task1-reference-thinking-effect-spans.tex",
        "task1-llm-reference-effect.pdf",
    }
    assert {path.name for path in tmp_path.iterdir()} == {
        name for path in paths for name in (path.name, path.with_suffix(".md").name)
    }
    assert "Qwen 3.6" in paths[0].read_text()
    assert "Mean span length" in paths[1].read_text()
    assert paths[2].stat().st_size > 0
    assert "| Gemma 4" in paths[0].with_suffix(".md").read_text()
    assert "Errors per prediction" in paths[1].with_suffix(".md").read_text()
    markdown = paths[2].with_suffix(".md").read_text()
    assert "Mean ΔF1" in markdown
    assert "(pp)" not in markdown
