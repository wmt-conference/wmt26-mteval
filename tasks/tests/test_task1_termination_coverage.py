"""Coverage thresholds, ranking exclusions, and pooled shared-stop regression cases."""

from dataclasses import replace

import pytest
from span_mt_metrics_eval import compute

from _task1_helpers import write_jsonl, write_json, model_card, baseline_fixture
from wmt26_mteval_task1.coverage import build_coverage_report
from wmt26_mteval_task1.gpt_sol_stop import MEDIUM, XHIGH, analyze_gpt_sol_stop
from wmt26_mteval_task1.gpt_sol_stop_reporting import write_paper_gpt_sol_stop_report
from wmt26_mteval_task1.io import (
    load_test_set,
    load_ground_truth,
    load_model_card_template,
)
from wmt26_mteval_task1.models import TerminationCoverage
from wmt26_mteval_task1.reporting import write_paper_score_reports
from wmt26_mteval_task1.scoring import score_submissions
from wmt26_mteval_task1.termination import build_termination_coverage


@pytest.mark.parametrize("invalid,excluded", [(0, False), (1, False), (2, True)])
def test_threshold_counts_gold_pairs_and_preserves_metric_inputs(
    tmp_path, invalid, excluded
):
    """Use 20 gold pairs as denominator despite extra predictions and repeated human assessments."""

    descriptor, test_path, gold_path, template_path, predictions, test_rows = (
        baseline_fixture(tmp_path)
    )
    first = predictions[0]
    rows = [dict(first, item_id=first["item_id"] + str(i)) for i in range(100)]
    tests = [dict(test_rows[0], item_id=row["item_id"]) for row in rows]
    write_jsonl(test_path, tests)
    write_jsonl(gold_path, rows[:20])
    test = load_test_set(test_path)
    refs = load_ground_truth(gold_path, test)
    assessments = {key: dict(value) for key, value in refs.assessments.items()}
    assessments[rows[0]["item_id"]]["system"] *= 3
    refs = replace(refs, assessments=assessments)
    template = load_model_card_template(template_path)
    write_jsonl(descriptor.predictions_path, rows)
    original = score_submissions([descriptor], test, refs, template).scores[0]
    for i, row in enumerate(rows):
        row["stage1_finish_reason"] = {
            "system": None if i < invalid or i >= 20 else "stop"
        }
    write_jsonl(descriptor.predictions_path, rows)
    scored = score_submissions([descriptor], test, refs, template).scores[0]
    direction = next(iter(test.official_directions))
    finish = scored.coverage.termination_by_direction[direction]
    assert finish.gold_pairs == 20 and finish.invalid_pairs == invalid
    assert finish.excluded is excluded
    assert scored.direction_scores == original.direction_scores
    assert scored.coverage.provided_pair_counts[direction] == 100


def test_policy_excludes_reference_system_and_handles_no_gold(tmp_path):
    """Ignore reference-system, challenge, and unsupported pairs even with manually supplied gold."""

    _, test_path, gold_path, _, predictions, _ = baseline_fixture(tmp_path)
    test = load_test_set(test_path)
    refs = load_ground_truth(gold_path, test)
    first = predictions[0]["item_id"]
    direction = test.item_directions[first]
    # Treat this LP's only system as its translation-reference system.
    test = replace(test, reference_systems_by_direction={direction: "system"})
    statuses = {(first, "system"): "null", ("challenge", "system"): "length"}
    values = build_termination_coverage(test, refs, statuses)
    assert values[direction].gold_pairs == 0
    assert values[direction].failure_rate is None and not values[direction].excluded
    coverage = build_coverage_report(
        test,
        test.expected_pairs_by_direction,
        official_rows=8,
        challenge_rows=0,
        total_rows=8,
        termination_by_direction=values,
    )
    assert coverage.all_primary_complete
    assert TerminationCoverage(20, {"length": 1}).excluded is False


def test_ranking_reports_exclude_only_failed_lp_and_required_aggregates(tmp_path):
    """Ensure all ranking formats share the exclusion and report exact gold counts."""

    descriptor, test_path, gold_path, template_path, predictions, _ = baseline_fixture(
        tmp_path
    )
    predictions[0]["stage1_finish_reason"] = {"system": None}
    write_jsonl(descriptor.predictions_path, predictions)
    test = load_test_set(test_path)
    result = score_submissions(
        [descriptor],
        test,
        load_ground_truth(gold_path, test),
        load_model_card_template(template_path),
    )
    assert len(result.scores) == 1 and not result.failed
    output = tmp_path / "reports"
    write_paper_score_reports(output, result, None)
    failed = test.item_directions[predictions[0]["item_id"]]
    assert failed not in (output / "per_direction_rankings.tex").read_text()
    assert failed in result.scores[0].coverage.termination_excluded_directions


@pytest.mark.parametrize("algorithm", ["greedy", "optimal"])
def test_shared_stop_calls_once_per_evaluator_with_pair_weights(
    tmp_path, monkeypatch, algorithm
):
    """Retain matched reference occurrences, pool unequal LP sizes, and use reciprocal weights."""

    descriptor, test_path, gold_path, template_path, predictions, test_rows = (
        baseline_fixture(tmp_path)
    )
    extra = dict(predictions[1], item_id=predictions[1]["item_id"] + "extra")
    predictions.append(extra)
    test_rows.append(dict(test_rows[1], item_id=extra["item_id"]))
    write_jsonl(test_path, test_rows)
    write_jsonl(gold_path, predictions)
    test = load_test_set(test_path)
    refs = load_ground_truth(gold_path, test)
    refs = replace(
        refs,
        assessments={
            item: {
                system: values * (3 if item == extra["item_id"] else 1)
                for system, values in systems.items()
            }
            for item, systems in refs.assessments.items()
        },
    )
    descriptors = []
    for name in (MEDIUM, XHIGH):
        current = replace(descriptor, path=tmp_path / name)
        write_json(current.model_card_path, model_card())
        rows = [
            dict(row, stage1_finish_reason={"system": "stop"}) for row in predictions
        ]
        if name == XHIGH:
            rows[-2]["stage1_finish_reason"] = {"system": None}
        else:
            rows[-1] = dict(
                rows[-1], task1_pred={"system": {"errors": [], "omission": None}}
            )
        write_jsonl(current.predictions_path, reversed(rows))
        descriptors.append(current)
    calls = []

    def capture(predictions, references, **kwargs):
        """Capture the actual pooled metric inputs while executing the real matcher."""
        calls.append((predictions, references, kwargs))
        return compute(predictions, references, **kwargs)

    monkeypatch.setattr("wmt26_mteval_task1.gpt_sol_stop.compute", capture)
    result = analyze_gpt_sol_stop(
        descriptors,
        test,
        refs,
        load_model_card_template(template_path),
        matching_algorithm=algorithm,
    )
    assert len(calls) == 2
    assert calls[0][1] == calls[1][1]
    for _, _, options in calls:
        assert (
            options["averaging"] == "micro"
            and options["matching_algorithm"] == algorithm
        )
        assert "segment_weights" not in options
    assert len(result.pairs) == 8 and result.scores[0]["retained_assessments"] == 10
    statistics = {row["evaluator"]: row for row in result.prediction_statistics}
    assert statistics[MEDIUM]["prediction_assessments"] == 8
    assert statistics[XHIGH]["prediction_assessments"] == 8
    assert statistics[MEDIUM]["total_error_count"] == 7
    assert statistics[XHIGH]["total_error_count"] == 8
    assert statistics[MEDIUM]["mean_errors_per_assessment"] == pytest.approx(7 / 8)
    assert statistics[XHIGH]["mean_errors_per_assessment"] == pytest.approx(1)
    assert statistics[MEDIUM]["mean_severity_weight"] == pytest.approx(1)
    assert statistics[XHIGH]["mean_severity_weight"] == pytest.approx(1)
    assert result.scores[2]["f1"] > 0
    assert any(row["retained_pairs"] == 0 for row in result.directions)
    output = tmp_path / "comparison"
    assert write_paper_gpt_sol_stop_report(output, result).is_file()
    assert {path.name for path in output.iterdir()} == {
        "gpt_sol_stop.tex", "gpt_sol_stop.md",
    }
    latex = (output / "gpt_sol_stop.tex").read_text()
    assert r"\begin{tabular}{lrrrrrr}" in latex
    assert "GPT-5.6 Sol - medium thinking" in latex
    assert "GPT-5.6 Sol - xhigh thinking" in latex
    assert r"$\Delta$ (xhigh $-$ medium)" in latex
    assert "Errors per prediction" in latex
    markdown = (output / "gpt_sol_stop.md").read_text()
    assert "both models completed reasoning" in markdown
    assert "(pp)" not in markdown
    assert "Mean span length" in latex
    assert "Mean severity weight" in latex
    assert r"\label{tab:task1-gpt-sol-stop}" in latex
    with pytest.raises(ValueError, match="missing required"):
        analyze_gpt_sol_stop(
            descriptors[:1], test, refs, load_model_card_template(template_path)
        )
    rows = [dict(row, stage1_finish_reason={"system": "length"}) for row in predictions]
    write_jsonl(descriptors[1].predictions_path, rows)
    with pytest.raises(ValueError, match="no gold-annotated pairs"):
        analyze_gpt_sol_stop(
            descriptors, test, refs, load_model_card_template(template_path)
        )


def test_reference_quality_termination_exclusion_is_a_warning(tmp_path):
    """Exclude an otherwise complete comparison without turning termination into a schema error."""

    import json
    from test_task1_reference_quality import _analysis_context
    from wmt26_mteval_task1.reference_quality import analyze_reference_quality

    entries, test, refs, template, index = _analysis_context(tmp_path)
    path = entries[0].predictions_path
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["stage1_finish_reason"] = {"mt": None}
    write_jsonl(path, rows)
    result = analyze_reference_quality(entries, test, refs, template, index)
    assert not result.failed
    assert all(not scores for scores in result.scores_by_comparison.values())
    assert any(issue.stage == "termination" for issue in result.issues)
    assert all(issue.severity == "warning" for issue in result.issues)


def test_stop_cli_includes_supplied_lp_without_gold(tmp_path):
    """Exercise CLI discovery/defaults and retain zero-gold LPs in the count report."""

    from scripts.analyze_task1_gpt_sol_stop import main

    descriptor, test_path, gold_path, template_path, predictions, _ = baseline_fixture(
        tmp_path
    )
    write_jsonl(gold_path, predictions[:-1])
    for name in (MEDIUM, XHIGH):
        entry = replace(descriptor, path=tmp_path / "gpt" / name)
        write_json(entry.model_card_path, model_card())
        write_jsonl(
            entry.predictions_path,
            [dict(row, stage1_finish_reason={"system": "stop"}) for row in predictions],
        )
    output = tmp_path / "output"
    assert (
        main(
            [
                "--baselines-dir",
                str(tmp_path / "gpt"),
                "--run-specific-info",
                "deduplicated",
                "--test-set",
                str(test_path),
                "--ground-truth",
                str(gold_path),
                "--model-card-template",
                str(template_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    table = output / "greedy/deduplicated/gpt_sol_stop.tex"
    assert table.is_file()
    assert "xhigh minus medium" in table.read_text()
    assert {path.name for path in table.parent.iterdir()} == {
        "gpt_sol_stop.tex", "gpt_sol_stop.md",
    }
