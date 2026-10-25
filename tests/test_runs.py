import os
from pathlib import Path

import pytest

from wmt26_mteval.grand import Grand, ManifestEntry
from wmt26_mteval.runs import RunSpec, baseline_run_id, list_runs


def entry(kind, namespace, folder, source_id=2, canonical=True):
    """A canonical manifest entry laid out like the real ones."""
    return ManifestEntry(
        source_id=source_id,
        column=f"{kind}__{namespace}__{folder}__predictions",
        path=f"{kind}s/{namespace}/{folder}/predictions.jsonl",
        kind=kind,
        task_namespace=namespace,
        version="original",
        run_id=folder,
        bytes=0,
        sha256="0" * 64,
        records=0,
        canonical_source_id=source_id if canonical else 0,
        canonical_column="",
    )


class FakeGrand:
    """Stands in for ``Grand``: ``list_runs`` only calls ``canonical_entries()``."""

    def __init__(self, entries):
        self._entries = entries

    def canonical_entries(self):
        return list(self._entries)


@pytest.fixture(scope="module")
def mini_runs(mini_grand_path):
    return list_runs(Grand(mini_grand_path, expected_sha256=None))


def test_list_runs_on_mini(mini_grand_path):
    runs = list_runs(Grand(mini_grand_path, expected_sha256=None))
    subs = [r for r in runs if r.kind == "submission"]
    base = [r for r in runs if r.kind == "baseline"]
    assert len(subs) == 50 and len(base) == 35  # 23 official + 12 alt
    assert sum(r.variant == "alt" for r in base) == 12
    x = [r for r in base if r.run_id == "xcomet_xl_ref"]
    assert [r.variant for r in x] == ["official"]
    assert not any("pre_dedup" in r.column for r in runs)


def test_run_counts_by_task(mini_runs):
    def tally(kind):
        counts = {}
        for r in mini_runs:
            if r.kind == kind:
                counts[(r.task, r.variant)] = counts.get((r.task, r.variant), 0) + 1
        return counts

    assert tally("submission") == {
        ("task1", "official"): 17,
        ("task2", "official"): 20,
        ("task3", "official"): 13,
    }
    # 12 deduplicated + 4 xCOMET originals are official; the 12 LLM originals are alt.
    assert tally("baseline") == {
        ("task1", "official"): 16,
        ("task1", "alt"): 12,
        ("task2", "official"): 7,
    }


def test_runs_are_sorted_and_unique(mini_runs):
    keys = [(r.kind, r.run_id, r.variant) for r in mini_runs]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)


def test_submission_run_spec(mini_runs):
    run = next(r for r in mini_runs if r.run_id == "875921")
    assert run == RunSpec(
        run_id="875921",
        kind="submission",
        task="task1",
        variant="official",
        column="submission__task1__875921__predictions",
        source_path="submissions/task1/875921/predictions.jsonl",
    )


@pytest.mark.parametrize(
    "run_id, variant, task, folder",
    [
        ("gemma", "official", "task1", "task1_deduplicated/gemma_task1__deduplicated"),
        ("gemma", "alt", "task1", "task1/gemma_task1"),
        (
            "gemma_ref_no_thinking",
            "official",
            "task1",
            "task1_deduplicated/gemma_ref_no_thinking_task1__deduplicated",
        ),
        ("gemma_ref", "alt", "task1", "task1/gemma_ref_task1"),
        ("xcomet_xl_ref", "official", "task1", "task1/xcomet_xl_ref_task1"),
        ("xcomet_xxl_no_ref", "official", "task1", "task1/xcomet_xxl_no_ref_task1"),
        ("872013", "official", "task2", "task2/872013"),
        ("873002", "official", "task2", "task2/873002"),
    ],
)
def test_baseline_run_specs(mini_runs, run_id, variant, task, folder):
    (run,) = [
        r for r in mini_runs if r.kind == "baseline" and (r.run_id, r.variant) == (run_id, variant)
    ]
    namespace, name = folder.split("/")
    assert run.task == task
    assert run.source_path == f"baselines/{namespace}/{name}/predictions.jsonl"
    assert run.column == f"baseline__{namespace}__{name}__predictions"


def test_every_run_column_is_a_canonical_column(mini_grand_path, mini_runs):
    canonical = {e.column for e in Grand(mini_grand_path, expected_sha256=None).canonical_entries()}
    assert {r.column for r in mini_runs} <= canonical


@pytest.mark.parametrize(
    "folder, run_id",
    [
        ("gemma_task1__deduplicated", "gemma"),
        ("gemma_task1", "gemma"),
        ("gemma_ref_task1", "gemma_ref"),
        ("gemma_ref_task1__pre_dedup_pre_enhy_fix", "gemma_ref"),
        ("xcomet_xl_ref_task1", "xcomet_xl_ref"),
        ("gpt_56_sol_medium_no_ref_task1__deduplicated", "gpt_56_sol_medium_no_ref"),
        ("872013", "872013"),
        ("task1_gemma", "task1_gemma"),  # only a trailing suffix is stripped
    ],
)
def test_baseline_run_id(folder, run_id):
    assert baseline_run_id(folder) == run_id


def test_original_is_alt_only_when_a_deduplicated_run_exists():
    runs = list_runs(
        FakeGrand(
            [
                entry("baseline", "task1", "gemma_task1", 2),
                entry("baseline", "task1_deduplicated", "gemma_task1__deduplicated", 3),
                entry("baseline", "task1", "xcomet_xl_ref_task1", 4),
                entry("baseline", "task1", "qwen_task1", 5),  # no deduplicated sibling
            ]
        )
    )
    assert [(r.run_id, r.variant) for r in runs] == [
        ("gemma", "alt"),
        ("gemma", "official"),
        ("qwen", "official"),
        ("xcomet_xl_ref", "official"),
    ]


def test_non_run_entries_are_skipped():
    runs = list_runs(
        FakeGrand(
            [
                entry("ground_truth", "ground_truth", "x", 1),
                entry("test_set", "test_set", "y", 0),
                entry("baseline", "task1_prededup_and_enhy_fix", "a_task1__pre_dedup_pre_enhy_fix"),
                entry("submission", "task3", "111111", 5),
            ]
        )
    )
    assert [(r.kind, r.run_id, r.task) for r in runs] == [("submission", "111111", "task3")]


def test_unknown_baseline_namespace_fails_loudly():
    with pytest.raises(ValueError, match="task9"):
        list_runs(FakeGrand([entry("baseline", "task9", "abc_task9")]))


def test_duplicate_run_ids_fail_loudly():
    with pytest.raises(ValueError, match="duplicate"):
        list_runs(
            FakeGrand(
                [
                    entry("baseline", "task1", "gemma_task1", 2),
                    entry("baseline", "task1", "gemma_task1", 3),
                ]
            )
        )


@pytest.fixture(scope="module")
def real_grand():
    path = os.environ.get("GRAND")
    if not path or not Path(path).is_file():
        pytest.skip("GRAND is not set to the real wmt26-grand.parquet")
    return Grand(Path(path))


@pytest.mark.fulldata
def test_real_run_inventory(real_grand):
    runs = list_runs(real_grand)
    counts = {}
    for run in runs:
        counts[(run.kind, run.variant)] = counts.get((run.kind, run.variant), 0) + 1
    assert counts == {
        ("submission", "official"): 50,
        ("baseline", "official"): 23,
        ("baseline", "alt"): 12,
    }
    assert len({(r.kind, r.run_id, r.variant) for r in runs}) == 85
