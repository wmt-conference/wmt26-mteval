from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts.stage_task2_inputs import build_parser, stage


ROOT = Path(__file__).resolve().parents[1]
GRAND = ROOT / "wmt26-grand.parquet"
# Task 3 pins the reconstructed bytes of every file it shares with Task 2.
TASK3_CATALOG = ROOT / "src" / "wmt26_mteval_task3" / "parquet_inputs.json"


def test_defaults_match_the_paths_of_the_task2_readme():
    arguments = build_parser().parse_args([])

    assert arguments.grand == Path("wmt26-grand.parquet")
    assert arguments.model_cards == Path("model_cards")
    assert arguments.out == Path(".")


def test_refuses_when_a_target_folder_exists(tmp_path):
    (tmp_path / "submissions").mkdir()

    # The parquet path does not exist: the check must come before anything is read.
    with pytest.raises(FileExistsError, match="submissions"):
        stage(tmp_path / "missing.parquet", ROOT / "model_cards", tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["submissions"]


@pytest.mark.skipif(not GRAND.is_file(), reason="integrated WMT26 parquet file is unavailable")
def test_stages_every_record_from_the_grand_parquet(tmp_path):
    counts = stage(GRAND, ROOT / "model_cards", tmp_path)

    manifest = json.loads(pq.ParquetFile(GRAND).schema_arrow.metadata[b"wmt26_source_manifest"])
    expected = {entry["path"]: entry["records"] for entry in manifest if entry["path"] in counts}
    assert counts == expected
    assert "human_annotations/ground-truth.jsonl" in counts
    assert any(path.startswith("submissions/task2/") for path in counts)
    assert any(path.startswith("baselines/task1_prededup_and_enhy_fix/") for path in counts)
    for path in counts:
        lines = (tmp_path / path).read_text(encoding="utf-8").splitlines()
        assert len(lines) == counts[path]
    pinned = json.loads(TASK3_CATALOG.read_text(encoding="utf-8"))["sources"]
    shared = sorted(set(pinned) & set(counts))
    assert "human_annotations/ground-truth.jsonl" in shared
    assert "test-set-files/mteval-test26.jsonl" in shared
    for path in shared:
        digest = hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
        assert digest == pinned[path]["sha256"], path
