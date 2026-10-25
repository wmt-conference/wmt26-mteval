import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wmt26_mteval.build import _git_state, build_all
from wmt26_mteval.grand import ChecksumError, Grand
from wmt26_mteval.runs import list_runs
from wmt26_mteval.schemas import SCHEMAS

CONFIGS = {
    "items",
    "pairs",
    "submissions",
    "baselines",
    "systems",
    "mt_systems",
    "challenge_sets",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory, mini_grand_path, repo_root):
    """One build of the mini grand file, shared by the read-only tests: ``(out_dir, manifest)``."""
    out = tmp_path_factory.mktemp("build") / "hf"
    return out, build_all(mini_grand_path, repo_root, out, expected_sha256=None)


def _paths(manifest: dict) -> dict[str, list[str]]:
    return {c: [f["path"] for f in v["files"]] for c, v in manifest["configs"].items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_mini(tmp_path, mini_grand_path, repo_root):
    m = build_all(mini_grand_path, repo_root, tmp_path, expected_sha256=None)
    assert set(m["configs"]) == CONFIGS
    assert not (tmp_path / "wmt26-grand.parquet").exists()
    for c, info in m["configs"].items():
        assert info["rows"] == sum(
            pq.ParquetFile(tmp_path / f["path"]).metadata.num_rows for f in info["files"]
        )


def test_build_is_deterministic(tmp_path, mini_grand_path, repo_root):
    a = build_all(mini_grand_path, repo_root, tmp_path / "a", expected_sha256=None)
    b = build_all(mini_grand_path, repo_root, tmp_path / "b", expected_sha256=None)

    def strip(m):
        return {c: [f["sha256"] for f in v["files"]] for c, v in m["configs"].items()}

    assert strip(a) == strip(b)


def test_layout_follows_the_naming_scheme(built, mini_grand_path, mini_pairs):
    out, manifest = built
    runs = list_runs(Grand(mini_grand_path, expected_sha256=None))
    expected = {
        "items": ["data/items/evaluation-00000.parquet"],
        "pairs": sorted(
            f"data/pairs/evaluation-{lp}.parquet" for lp in set(mini_pairs.column("lp").to_pylist())
        ),
        "submissions": sorted(
            f"data/submissions/evaluation-{r.task}-{r.run_id}.parquet"
            for r in runs
            if r.kind == "submission"
        ),
        "baselines": sorted(
            f"data/baselines/evaluation-{r.run_id}-{r.variant}.parquet"
            for r in runs
            if r.kind == "baseline"
        ),
        "systems": ["data/systems/evaluation-00000.parquet"],
        "mt_systems": ["data/mt_systems/evaluation-00000.parquet"],
        "challenge_sets": ["data/challenge_sets/evaluation-00000.parquet"],
    }
    assert _paths(manifest) == expected
    on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    listed = {path for paths in expected.values() for path in paths}
    # the card sits next to the manifest; there is no copy of the grand file
    assert on_disk == listed | {"README.md", "build_manifest.json"}


@pytest.mark.parametrize("config", sorted(SCHEMAS))
def test_files_keep_their_schema_and_metadata(built, config):
    out, manifest = built
    want = SCHEMAS[config]
    for entry in manifest["configs"][config]["files"]:
        got = pq.read_schema(out / entry["path"])
        # Parquet names the element of a list `element`, not `item`; the type is the same.
        assert got.names == want.names
        assert [f.type for f in got] == [f.type for f in want]
        assert [f.nullable for f in got] == [f.nullable for f in want]
        assert [f.metadata for f in got] == [f.metadata for f in want]
        assert got.metadata == want.metadata


def test_files_are_zstd_compressed_and_dictionary_encoded(built):
    out, manifest = built
    for info in manifest["configs"].values():
        for entry in info["files"]:
            metadata = pq.ParquetFile(out / entry["path"]).metadata
            row_group = metadata.row_group(0)
            columns = [row_group.column(i) for i in range(row_group.num_columns)]
            assert {c.compression for c in columns} == {"ZSTD"}
            assert any("DICTIONARY" in encoding for c in columns for encoding in c.encodings)


def test_pairs_have_one_file_per_language_pair_in_order(built, mini_pairs):
    out, manifest = built
    parts = []
    for entry in manifest["configs"]["pairs"]["files"]:
        table = pq.read_table(out / entry["path"])
        lps = set(table.column("lp").to_pylist())
        assert [entry["path"]] == [f"data/pairs/evaluation-{lp}.parquet" for lp in lps]
        keys = list(zip(table.column("item_id").to_pylist(), table.column("system").to_pylist()))
        assert keys == sorted(keys)
        parts.append(table)
    assert len(parts) > 1
    merged = pa.concat_tables(parts).sort_by([("item_id", "ascending"), ("system", "ascending")])
    assert merged.equals(mini_pairs)


def test_metadata_configs_equal_their_builders(built, mini_pairs, mini_systems):
    out, manifest = built
    systems = pq.read_table(out / "data/systems/evaluation-00000.parquet")
    assert systems.equals(mini_systems)
    mt_systems = pq.read_table(out / "data/mt_systems/evaluation-00000.parquet")
    assert mt_systems.column("system").to_pylist() == sorted(set(mini_pairs["system"].to_pylist()))
    challenge_sets = pq.read_table(out / "data/challenge_sets/evaluation-00000.parquet")
    assert challenge_sets.num_rows == 7


def test_manifest_file_matches_the_returned_dict(built):
    out, manifest = built
    text = (out / "build_manifest.json").read_text(encoding="utf-8")
    assert text == json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    assert json.loads(text) == manifest


def test_manifest_records_provenance(built, mini_grand_path):
    _, manifest = built
    assert set(manifest) == {
        "grand_sha256",
        "git_commit",
        "git_dirty",
        "pyarrow_version",
        "configs",
        "stats",
    }
    assert manifest["grand_sha256"] == _sha256(mini_grand_path)  # computed, not assumed
    assert manifest["pyarrow_version"] == pa.__version__
    assert manifest["git_commit"] is None or len(manifest["git_commit"]) == 40
    assert isinstance(manifest["git_dirty"], bool) == (manifest["git_commit"] is not None)


def test_manifest_files_are_sorted_relative_and_checksummed(built):
    out, manifest = built
    for info in manifest["configs"].values():
        paths = [f["path"] for f in info["files"]]
        assert paths == sorted(paths)
        assert info["rows"] == sum(f["rows"] for f in info["files"])
        for entry in info["files"]:
            assert set(entry) == {"path", "sha256", "rows"}
            assert not Path(entry["path"]).is_absolute() and "\\" not in entry["path"]
            assert entry["sha256"] == _sha256(out / entry["path"])
            assert entry["rows"] == pq.ParquetFile(out / entry["path"]).metadata.num_rows


def test_the_verified_checksum_goes_into_the_manifest(tmp_path, mini_grand_path, repo_root):
    sha = _sha256(mini_grand_path)
    manifest = build_all(mini_grand_path, repo_root, tmp_path / "hf", expected_sha256=sha)
    assert manifest["grand_sha256"] == sha


def test_a_wrong_checksum_is_refused_before_anything_is_written(
    tmp_path, mini_grand_path, repo_root
):
    out = tmp_path / "hf"
    with pytest.raises(ChecksumError):
        build_all(mini_grand_path, repo_root, out)  # the default expects the real grand file
    assert not out.exists()


def test_a_missing_output_directory_is_created(tmp_path, mini_grand_path, repo_root):
    out = tmp_path / "deeper" / "hf"
    build_all(mini_grand_path, repo_root, out, expected_sha256=None)
    assert (out / "build_manifest.json").is_file()


def test_an_empty_output_directory_is_accepted(tmp_path, mini_grand_path, repo_root):
    build_all(mini_grand_path, repo_root, tmp_path, expected_sha256=None)
    assert (tmp_path / "build_manifest.json").is_file()


def test_a_non_empty_output_directory_is_refused(tmp_path, mini_grand_path, repo_root):
    stale = tmp_path / "data" / "pairs" / "evaluation-old.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale shard")
    with pytest.raises(FileExistsError, match=str(tmp_path)):
        build_all(mini_grand_path, repo_root, tmp_path, expected_sha256=None)
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [stale]
    assert stale.read_bytes() == b"stale shard"


def test_an_output_path_that_is_a_file_is_refused(tmp_path, mini_grand_path, repo_root):
    target = tmp_path / "hf"
    target.write_text("not a directory")
    with pytest.raises(FileExistsError, match=str(target)):
        build_all(mini_grand_path, repo_root, target, expected_sha256=None)


def _git(repo: Path, *args: str) -> str:
    config = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
    done = subprocess.run(
        ["git", "-C", str(repo), *config, "-c", "commit.gpgsign=false", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_state(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))  # no parent repository
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git_state(repo) == (None, None)  # not a repository
    _git(repo, "init", "-q")
    assert _git_state(repo) == (None, None)  # no commit yet
    _git(repo, "commit", "-q", "--allow-empty", "-m", "first")
    head = _git(repo, "rev-parse", "HEAD")
    assert _git_state(repo) == (head, False)
    (repo / "new.txt").write_text("uncommitted")
    assert _git_state(repo) == (head, True)


def test_git_state_without_git_on_the_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert _git_state(tmp_path) == (None, None)


def _run_cli(repo_root: Path, hash_seed: str, *args: str) -> subprocess.CompletedProcess:
    script = repo_root / "scripts" / "build_hf.py"
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    return subprocess.run(
        [sys.executable, str(script), *args], env=env, capture_output=True, text=True
    )


def test_cli_builds_identically_across_processes(tmp_path, mini_grand_path, repo_root):
    manifests = []
    for seed in ("1", "2"):
        out = tmp_path / f"seed{seed}"
        done = _run_cli(
            repo_root, seed, "--grand", str(mini_grand_path), "--out", str(out), "--no-checksum"
        )
        assert done.returncode == 0, done.stderr
        assert all(config in done.stdout for config in CONFIGS)
        assert (out / "README.md").is_file()
        manifests.append(json.loads((out / "build_manifest.json").read_text(encoding="utf-8")))
    assert manifests[0] == manifests[1]
    assert manifests[0]["configs"]["items"]["rows"] == 16


def test_cli_checks_the_checksum_unless_told_not_to(tmp_path, mini_grand_path, repo_root):
    out = tmp_path / "hf"
    done = _run_cli(repo_root, "0", "--grand", str(mini_grand_path), "--out", str(out))
    assert done.returncode != 0
    assert "sha256" in done.stderr
    assert not out.exists()


def test_cli_refuses_a_non_empty_output_directory(tmp_path, mini_grand_path, repo_root):
    (tmp_path / "leftover.txt").write_text("x")
    done = _run_cli(
        repo_root, "0", "--grand", str(mini_grand_path), "--out", str(tmp_path), "--no-checksum"
    )
    assert done.returncode != 0
    assert str(tmp_path) in done.stderr
    assert [p.name for p in tmp_path.iterdir()] == ["leftover.txt"]
