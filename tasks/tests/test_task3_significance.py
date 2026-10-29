"""Check original permutation/clustering behavior and release permission adapters."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wmt26_mteval_task3 import significance as sig
from wmt26_mteval_task3.io import DataError


def test_original_permutation_gives_one_for_identical_systems() -> None:
    """Identical per-source predictions must yield unit p-values under every swap."""
    counts = np.array([[[2, 0, 1, 1], [1, 1, 0, 2]],
                       [[2, 0, 1, 1], [1, 1, 0, 2]]])
    items = {name: sig._Item("source---target", ("official", "source", "news", "doc", name))
             for name in ("a", "b")}
    matrix = sig._permutation_p_values([counts], [["a", "b"]], items, 25, 42)
    np.testing.assert_array_equal(matrix, np.ones((2, 2)))


def test_original_greedy_cluster_checks_every_current_member() -> None:
    """A candidate differing from any current member must start a new cluster."""
    systems = [sig._System(key, "org", "model", set(), {}) for key in ("a", "b", "c")]
    scores = {key: sig._Scores(mcc, 0.5, 0.5, 0.5)
              for key, mcc in zip(("a", "b", "c"), (0.8, 0.7, 0.6))}
    results = sig._ranked_results(systems, scores,
                                  {("a", "b"): 0.2, ("a", "c"): 0.01, ("b", "c"): 0.9}, 0.05)
    assert [(row.rank, row.cluster) for row in results] == [(1, 1), (2, 1), (3, 2)]


def test_significance_adapter_does_not_bypass_task1_opt_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the upstream permission gate even when significance runs independently."""
    directory = tmp_path / "submissions/task1/1"
    directory.mkdir(parents=True)
    card = [{"question": "Organization name", "answer": "Example"},
            {"question": "Submission name", "answer": "Example"},
            {"question": "Allow automatic conversion from Task 1 to Task 3 (true|false)", "answer": False}]
    (directory / "model_card.json").write_text(json.dumps(card))
    (directory / "predictions.jsonl").write_text('{}\n')
    families = {name: [] for name in ("submissions", "task1_baselines_converted", "task1_converted", "task2_converted")}
    families["task1_converted"] = [{"entry_id": "1", "model_card": "submissions/task1/1/model_card.json",
                                    "predictions": "submissions/task1/1/predictions.jsonl"}]
    context = (SimpleNamespace(items={}, item_ids_by_pair={}), SimpleNamespace(labels={}), {"families": families})
    monkeypatch.setattr(sig, "load_context", lambda root: context)
    with pytest.raises(DataError, match="conversion no longer permitted"):
        sig.load_dataset(tmp_path, {"task1:1"})
