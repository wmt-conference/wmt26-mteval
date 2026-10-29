"""Statistical oracle, support-policy, cache, and report integration checks."""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.score_task1_submissions import main
from wmt26_mteval_task1.models import BatchScoreResult, PermutationDirection
from wmt26_mteval_task1.rankings import ranking_scopes, entry_key, scope_score
from wmt26_mteval_task1.reporting import write_paper_score_reports
from wmt26_mteval_task1.scoring import (
    PreparedAssessment, aggregate_mpp_components, permutation_inputs_from_assessments,
    score_prepared_assessments, score_submissions,
)
from wmt26_mteval_task1.significance import (
    SignificanceConfig, assign_clusters, compute_significance, direction_draws,
    f1_from_totals, permutation_p_value, permuted_differences, validate_support,
)
from wmt26_mteval_task1.significance_cache import CHUNK_SIZE, SignificanceCache
from wmt26_mteval_task1.submissions import discover_submissions
from _task1_helpers import assessment, major_span, write_json, write_jsonl
from test_task1_scoring import scoring_context, make_submission, _report_score


def test_significance_cache_invalidates_changed_parquet(tmp_path):
    """A changed physical dataset must not reuse a logically named draw stream."""

    path = tmp_path / "significance.sqlite3"
    draws = np.zeros(CHUNK_SIZE)
    with SignificanceCache(path, data_fingerprint="first") as cache:
        key = cache.register({"direction": "test"})
        cache.store(key, 0, draws)
    with SignificanceCache(path, data_fingerprint="first") as cache:
        assert cache.load(key, 0) is not None
    with SignificanceCache(path, data_fingerprint="second") as cache:
        assert cache.load(key, 0) is None


@pytest.mark.parametrize("algorithm", ["greedy", "optimal"])
def test_component_swaps_equal_explicit_prediction_rescoring(tmp_path, algorithm):
    """Exhaustively swap two outputs, keeping duplicate human assessments intact."""
    official, _, _, _ = scoring_context(tmp_path)
    make_submission(tmp_path, "1", official)
    descriptor = discover_submissions(tmp_path / "submissions")[0][0]
    a = (
        PreparedAssessment("one", "system", "lp", None, 8, [major_span(0, 4)], [major_span(0, 3)], 0),
        PreparedAssessment("one", "system", "lp", None, 8, [major_span(0, 4)], [major_span(4, 8)], 1),
        PreparedAssessment("two", "system", "lp", None, 8, [], [major_span(0, 4)], 0),
    )
    b = tuple(replace(record, prediction=[] if record.item_id == "one" else [major_span(0, 4)]) for record in a)
    scored_a = score_prepared_assessments(descriptor, a, matching_algorithm=algorithm)
    scored_b = score_prepared_assessments(descriptor, b, matching_algorithm=algorithm)
    inputs_a = permutation_inputs_from_assessments(scored_a)["lp"]
    inputs_b = permutation_inputs_from_assessments(scored_b)["lp"]
    assert len(inputs_a.support) == 2
    assert inputs_a.support[0][2] == ((0, 0.5), (1, 0.5))
    masks = np.array(list(itertools.product([0.0, 1.0], repeat=2)))
    actual = permuted_differences(inputs_a.components, inputs_b.components, masks)
    expected = []
    for mask in masks:
        swaps = {"one": mask[0], "two": mask[1]}
        hybrid_a = [right if swaps[left.item_id] else left for left, right in zip(a, b)]
        hybrid_b = [left if swaps[left.item_id] else right for left, right in zip(a, b)]
        f1_a = aggregate_mpp_components(score_prepared_assessments(descriptor, hybrid_a, matching_algorithm=algorithm))[2]
        f1_b = aggregate_mpp_components(score_prepared_assessments(descriptor, hybrid_b, matching_algorithm=algorithm))[2]
        expected.append(f1_a - f1_b)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(actual, -actual[::-1], atol=1e-15)
    # Micro F1 is not the average of assessment F1 values.
    assert not np.isclose(float(f1_from_totals(inputs_a.components.sum(axis=0))),
                          np.mean(f1_from_totals(inputs_a.components)))


def test_exact_tail_ties_and_empty_sides():
    """Two perfect-versus-missed outputs have one extreme draw in four."""
    a = np.array([[1., 1., 1., 1.], [1., 1., 1., 1.]])
    b = np.array([[0., 0., 0., 1.], [0., 0., 0., 1.]])
    masks = np.array(list(itertools.product([0., 1.], repeat=2)))
    null = permuted_differences(a, b, masks)
    assert permutation_p_value(null, 1., "one-sided") == 0.25
    assert permutation_p_value(null, -1., "one-sided") == 1.
    assert permutation_p_value(permuted_differences(a, a, masks), 0.) == 1.
    np.testing.assert_array_equal(f1_from_totals(np.array([[0., 0., 0., 0.], [0., 0., 0., 1.]])), [1., 0.])


def test_greedy_clusters_nontransitivity_and_boundary():
    """A cluster boundary need not separate all cross-cluster comparisons."""
    matrix = np.array([[np.nan, .5, .05, .9], [np.nan, np.nan, .5, .01],
                       [np.nan, np.nan, np.nan, .6], [np.nan]*4])
    assert assign_clusters(matrix, .05) == [1, 1, 2, 2]
    assert assign_clusters(np.full((1, 1), np.nan), .05) == [1]
    assert assign_clusters(np.empty((0, 0)), .05) == []
    with pytest.raises(ValueError, match="incomplete"):
        assign_clusters(np.full((2, 2), np.nan), .05)


def _arrays():
    """Return two simple aligned component inputs for cache experiments."""
    support = (("a", "s", ((0, 1.),)), ("b", "s", ((0, 1.),)))
    return (PermutationDirection(support, np.array([[1., 1., 1., 1.], [1., 1., 1., 1.]])),
            PermutationDirection(support, np.array([[0., 0., 0., 1.], [0., 0., 0., 1.]])))


def test_cache_prefix_extension_and_invalidation(tmp_path):
    """Warm runs and extensions exactly match fresh streams; changed inputs miss."""
    a, b = _arrays()
    path = tmp_path / "cache.sqlite3"
    with SignificanceCache(path) as cache:
        prefix, key = direction_draws(a, b, SignificanceConfig(20), cache, {"direction": "lp"})
        assert cache.computed_chunks == 1
    with SignificanceCache(path) as cache:
        extended, same_key = direction_draws(a, b, SignificanceConfig(CHUNK_SIZE + 10, alpha=.1), cache, {"direction": "lp"})
        assert same_key == key
        assert cache.reused_chunks == 1 and cache.computed_chunks == 1
        np.testing.assert_array_equal(prefix, extended[:20])
    with SignificanceCache(tmp_path / "fresh.sqlite3") as cache:
        fresh, _ = direction_draws(a, b, SignificanceConfig(CHUNK_SIZE + 10), cache, {"direction": "lp"})
        np.testing.assert_array_equal(fresh, extended)
    with SignificanceCache(path) as cache:
        for metadata, config, other in [
            ({"direction": "new-lp"}, SignificanceConfig(20), b),
            ({"direction": "lp"}, SignificanceConfig(20, seed=5), b),
            ({"direction": "lp"}, SignificanceConfig(20), replace(b, components=a.components)),
            ({"direction": "lp", "predictions_sha256": "changed"}, SignificanceConfig(20), b),
        ]:
            _, changed_key = direction_draws(a, other, config, cache, metadata)
            assert changed_key != key
        assert cache.computed_chunks == 4
        _, same_key = direction_draws(a, b, SignificanceConfig(20), cache, {"direction": "lp"}, recompute=True)
        assert same_key == key and cache.computed_chunks == 5


def test_interrupted_cache_resumes_completed_chunks(tmp_path, monkeypatch):
    """A failed second chunk leaves the first transaction available for reuse."""
    a, b = _arrays()
    with SignificanceCache(tmp_path / "cache.sqlite3") as cache:
        original_store = cache.store
        def interrupted_store(key, index, draws):
            """Simulate process failure before committing the second chunk."""
            if index == 1:
                raise RuntimeError("interrupted")
            original_store(key, index, draws)
        monkeypatch.setattr(cache, "store", interrupted_store)
        with pytest.raises(RuntimeError, match="interrupted"):
            direction_draws(a, b, SignificanceConfig(300), cache, {})
    with SignificanceCache(tmp_path / "cache.sqlite3") as cache:
        direction_draws(a, b, SignificanceConfig(300), cache, {})
        assert cache.reused_chunks == 1 and cache.computed_chunks == 1


def test_support_mismatch_is_not_intersected():
    """Different reference occurrence sets and missing outputs are rejected."""
    a, b = _arrays()
    for bad in (replace(b, support=b.support[:1]),
                replace(b, support=(("a", "s", ((1, 1.),)), b.support[1]))):
        with pytest.raises(ValueError, match="support"):
            validate_support(a, bad)
    with pytest.raises(ValueError, match="components"):
        validate_support(a, replace(b, components=np.full((2, 4), np.nan)))


def test_common_support_controls_scores_and_eligibility(tmp_path):
    """Czech→Ukrainian affects neither aggregate score nor aggregate eligibility."""
    common, excluded = "eng_Latn-ind_Latn", "ces_Latn-ukr_Cyrl"
    free = _report_score(tmp_path, 1, priority_f1=.4, other_f1=.99, full_official=True)
    based = _report_score(tmp_path, 2, priority_f1=.5, full_official=True)
    free = replace(free, direction_scores=(free.direction_scores[0], replace(free.direction_scores[1], language_pair=excluded)),
                   coverage=replace(free.coverage, complete_directions=(common,), partial_directions=(excluded,), all_official_complete=False))
    based = replace(based, coverage=replace(based.coverage, complete_directions=(common,)))
    batch = BatchScoreResult((free, based), (), directions_without_references=(excluded,),
                             selected_directions=(common, excluded), gold_directions=(common, excluded))
    aggregate = next(scope for scope in ranking_scopes(batch) if scope.name == "all_directions")
    assert aggregate.directions == (common,)
    assert [entry_key(score.descriptor) for score in aggregate.entries] == [("submission", "2"), ("submission", "1")]
    assert scope_score(free, aggregate.directions) == .4
    assert any(scope.name == "direction:" + excluded for scope in ranking_scopes(batch))
    missing = replace(based, direction_scores=())
    assert len(next(scope for scope in ranking_scopes(replace(batch, scores=(free, missing))) if scope.name == "all_directions").entries) == 1
    unscored_required = replace(batch, selected_directions=(common, excluded, "eng_Latn-deu_Latn"))
    assert not next(scope for scope in ranking_scopes(unscored_required) if scope.name == "all_directions").entries


def _batch(tmp_path):
    """Create two differently scoring evaluators on a reference-available direction."""
    official, test_set, references, template = scoring_context(tmp_path)
    make_submission(tmp_path, "1", official)
    second = make_submission(tmp_path, "2", official)
    second_card = json.loads((second / "model_card.json").read_text())
    for entry in second_card:
        if entry["question"] == "Submission name":
            entry["answer"] = "Test System 2"
    write_json(second / "model_card.json", second_card)
    write_jsonl(second / "predictions.jsonl", [{"item_id": official, "task1_pred": {"system": assessment([])}}])
    submissions = discover_submissions(tmp_path / "submissions")[0]
    batch = score_submissions(submissions, test_set, references, template, collect_significance_inputs=True)
    return batch, test_set, references, template


def test_significance_clusters_and_paper_table_reuse(tmp_path):
    """Cached tests produce stable clusters in the per-direction paper table."""
    batch, test_set, references, _ = _batch(tmp_path)
    config = SignificanceConfig(300)
    kwargs = {"test_set_path": test_set.path, "ground_truth_path": references.path}
    cold = compute_significance(batch, config, tmp_path / "sig.sqlite3", **kwargs)
    warm = compute_significance(replace(batch, scores=batch.scores[::-1]), config, tmp_path / "sig.sqlite3", **kwargs)
    assert cold.pairs == warm.pairs and cold.clusters == warm.clusters
    assert warm.manifest["cache"]["computed_chunks"] == 0
    assert len(cold.pairs) == 2  # one direction and its aggregate
    assert cold.pairs[0].p_value == cold.pairs[1].p_value
    output = tmp_path / "out"
    paths = write_paper_score_reports(output, batch, cold)
    assert {path.name for path in paths} == {
        "priority_ranking.tex", "per_direction_rankings.tex",
    }
    assert r"\multicolumn{1}{c}{Cluster}" in paths[1].read_text()
    singleton = compute_significance(replace(batch, scores=batch.scores[:1]), config, tmp_path / "sig.sqlite3", **kwargs)
    assert not singleton.pairs and singleton.clusters["all_directions"] == {("submission", "1"): 1}
    write_paper_score_reports(output, batch, None)
    assert not (output / "significance_manifest.json").exists()


def test_paper_rankings_reject_duplicate_submission_names(tmp_path):
    """Fail publication when two submissions would share one display name."""

    batch, _, _, _ = _batch(tmp_path)
    first, second = batch.scores
    duplicate = replace(
        second,
        model_card=replace(
            second.model_card,
            submission_name=first.model_card.submission_name,
        ),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        write_paper_score_reports(
            tmp_path / "rankings", replace(batch, scores=(first, duplicate)), None
        )


def test_cli_paper_rankings_and_validation(tmp_path):
    """The CLI writes the paper rankings with supplied input paths."""
    _, test_set, references, template = _batch(tmp_path)
    arguments = ["--submissions-dir", str(tmp_path / "submissions"), "--baseline",
                 "--baselines-dir", "baselines/task1_deduplicated", "--run-specific-info", "deduplicated",
                 "--test-set", str(test_set.path), "--ground-truth", str(references.path),
                 "--model-card-template", str(template.path), "--output-dir", str(tmp_path / "out"),
                 "--mpp-cache-root", str(tmp_path / "cache"),
                 "--significance-permutations", "20"]
    assert main(arguments) == 0
    output = tmp_path / "out" / "greedy" / "deduplicated"
    assert {path.name for path in output.iterdir()} == {
        "priority_ranking.tex", "per_direction_rankings.tex",
        "priority_ranking.md", "per_direction_rankings.md",
    }
    markdown = (output / "priority_ranking.md").read_text()
    assert "| Rank | Cluster | Auto-evaluator |" in markdown
    assert "| P | R | F1 |" in markdown
    assert "percentages" not in markdown
    for flag, value in [("--significance-permutations", "0"), ("--significance-seed", "-1"), ("--significance-alpha", "nan")]:
        assert main(arguments + [flag, value]) == 2


def test_aggregate_averages_direction_draws_and_new_entries_reuse_pairs(tmp_path):
    """Unequal direction sizes retain equal weights, including reversed ordering."""
    batch, test_set, references, _ = _batch(tmp_path)
    direction = "eng_Latn-ind_Latn"
    support = tuple((str(index), "system", ((0, 1.),)) for index in range(20))
    entries = []
    for index, score in enumerate(batch.scores):
        components = np.tile([0., 0., 0., 1.] if index == 0 else [1., 1., 1., 1.], (20, 1))
        extra = replace(score.direction_scores[0], language_pair=direction,
                        precision=1., recall=float(index), f1=float(index))
        entries.append(replace(score, direction_scores=score.direction_scores + (extra,),
                               permutation_inputs={**score.permutation_inputs, direction: PermutationDirection(support, components)},
                               coverage=replace(score.coverage, complete_directions=score.coverage.complete_directions + (direction,))))
    batch = replace(batch, scores=tuple(entries), selected_directions=batch.selected_directions + (direction,),
                    gold_directions=batch.gold_directions + (direction,))
    config = SignificanceConfig(100)
    path = tmp_path / "cache.sqlite3"
    kwargs = {"test_set_path": test_set.path, "ground_truth_path": references.path}
    result = compute_significance(batch, config, path, **kwargs)
    aggregate = next(pair for pair in result.pairs if pair.scope == "all_directions")
    assert aggregate.better == ("submission", "2")
    assert aggregate.observed_difference == pytest.approx(1 / 6)
    with SignificanceCache(path) as cache:
        draws = [cache.load(stream["fingerprint"], 0)[:100] for stream in result.manifest["streams"]]
    expected_null = -np.mean(draws, axis=0)
    assert aggregate.p_value == permutation_p_value(expected_null, aggregate.observed_difference)
    third = replace(entries[0], descriptor=replace(entries[0].descriptor, submission_id="3"))
    extended = compute_significance(replace(batch, scores=(*batch.scores, third)), config, path, **kwargs)
    assert extended.manifest["cache"]["reused_chunks"] == 2
    assert extended.manifest["cache"]["computed_chunks"] == 4
    changed_alpha = compute_significance(batch, replace(config, alpha=.1), path, **kwargs)
    assert changed_alpha.manifest["cache"]["computed_chunks"] == 0


def test_termination_exclusion_applies_to_significance_scopes(tmp_path):
    """An excluded direction removes its entry from both relevant ranking scopes."""
    from wmt26_mteval_task1.models import TerminationCoverage
    batch, _, _, _ = _batch(tmp_path)
    score = batch.scores[0]
    direction = score.direction_scores[0].language_pair
    excluded = replace(score, coverage=replace(
        score.coverage, complete_directions=(), partial_directions=(direction,),
        termination_by_direction={direction: TerminationCoverage(1, {"null": 1})},
    ))
    scopes = ranking_scopes(replace(batch, scores=(excluded, batch.scores[1])))
    assert all(entry_key(entry.descriptor) != ("submission", "1") for scope in scopes for entry in scope.entries)


def test_transfer_optimization_matches_direct_sums():
    """Positive totals use the fast path without changing the tested statistic."""
    generator = np.random.default_rng(19)
    a = generator.uniform(.1, 2., size=(400, 4))
    b = generator.uniform(.1, 2., size=(400, 4))
    masks = generator.integers(0, 2, size=(256, len(a))).astype(float)
    masks[0] = 0.
    masks[1] = 1.
    expected = (f1_from_totals((1-masks) @ a + masks @ b)
                - f1_from_totals(masks @ a + (1-masks) @ b))
    np.testing.assert_allclose(permuted_differences(a, b, masks), expected,
                               rtol=1e-12, atol=1e-15)
    # Tiny potentially empty components must retain the direct-sum convention.
    a[:, 0] = 0.
    b[:, 0] = 1e-20
    expected = (f1_from_totals((1-masks) @ a + masks @ b)
                - f1_from_totals(masks @ a + (1-masks) @ b))
    np.testing.assert_array_equal(permuted_differences(a, b, masks), expected)
