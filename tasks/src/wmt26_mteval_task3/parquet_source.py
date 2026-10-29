"""Prepare Task 3 records from the grand parquet using the audited Task 1 reader.

Only input layout/serialization changes here. Existing Task 3 scoring and
statistical routines continue to consume their original JSON object schemas.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from wmt26_mteval_task1.parquet_source import Task1ParquetSource

from .io import DataError, load_json, sha256_file
from .model_cards import load_answers
from .reproduction import load_targets


CATALOG_PATH = Path(__file__).with_name("parquet_inputs.json")


class Task3ParquetSource(Task1ParquetSource):
    """Extend the existing source-order reconstruction to all three task namespaces."""

    def __init__(self, path: Path, model_cards_dir: Path) -> None:
        """Read the grand source manifest without activating or changing Task 1 defaults."""
        super().__init__(path, model_cards_dir)
        metadata = pq.ParquetFile(path).schema_arrow.metadata
        manifest = json.loads(metadata[b"wmt26_source_manifest"])
        self.entries = {Path(entry["path"]): entry for entry in manifest}
        if len(self.entries) != len(manifest):
            raise DataError("Duplicate logical paths in parquet source manifest")
        for logical in self.entries:
            if logical.is_absolute() or ".." in logical.parts:
                raise DataError("Parquet source path escapes its logical root")


def selected_sources(source: Task3ParquetSource) -> dict[str, dict[str, Any]]:
    """Resolve only the frozen shared inputs and five explicit historical system pools.

    Return staged relative paths mapped to their source-manifest entries. Ignore
    raw annotations (not in parquet) and all unselected evaluator snapshots.
    """
    from .preparation import FAMILY_DIRECTORIES

    targets = load_targets()
    selected = {}
    for shared in targets["shared_inputs"]:
        if shared["member"] == "annotations.json":
            continue
        entry = source.entries.get(Path(shared["member"]))
        if entry is None or entry["sha256"] != shared["sha256"]:
            raise DataError(f"Parquet shared-source identity changed: {shared['member']}")
        selected[shared["member"]] = entry
    for family in targets["families"]:
        for frozen in family["entries"]:
            matches = [source.entries[Path(prefix) / frozen["prediction_filename"]]
                       for prefix in frozen["dump_candidates"]
                       if Path(prefix) / frozen["prediction_filename"] in source.entries]
            if len(matches) != 1 or matches[0]["sha256"] != frozen["historical_prediction_sha256"]:
                raise DataError(f"Parquet prediction identity changed: {family['name']}/{frozen['entry_id']}")
            relative = (f"{FAMILY_DIRECTORIES[family['name']]}/{frozen['entry_id']}/"
                        f"{frozen['prediction_filename']}")
            if relative in selected and selected[relative] != matches[0]:
                raise DataError("Conflicting parquet sources for the same staged payload")
            selected[relative] = matches[0]
    return selected


def materialize_sources(source: Task3ParquetSource, output: Path,
                        selected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reconstruct selected JSON records in original line order and hash their new bytes.

    JSON whitespace may differ from the archived source. Preserve record values,
    mapping order and list order; pin reconstructed bytes separately from the
    manifest's historical source hashes. This function never reads legacy files.
    """
    files = {}
    for relative, entry in selected.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("wb") as handle:
            for _, record in source.iter_records(Path(entry["path"])):
                handle.write((json.dumps(record, ensure_ascii=False, allow_nan=False,
                                         separators=(",", ":")) + "\n").encode("utf-8"))
                count += 1
        files[relative] = {
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "source_path": entry["path"], "logical_sha256": entry["sha256"],
            "records": count,
        }
        print(f"Reconstructed {relative}: {count} records", flush=True)
    return files


def prepare_parquet_inputs(data_file: Path, output: Path,
                           model_cards_dir: Path = Path("model_cards")) -> dict[str, Any]:
    """Atomically stage the pinned parquet and complete public cards, with no repairs.

    Verify physical parquet identity, historical source identities and independent
    reconstructed-file hashes. Copy only selected model_card.json files from the
    task-specific public card roots; never read neighboring predictions or gold.
    """
    from .preparation import FAMILY_DIRECTORIES

    if output.exists():
        raise DataError(f"Refusing existing input directory: {output}")
    catalog = load_json(CATALOG_PATH)
    source = Task3ParquetSource(data_file, model_cards_dir)
    if source.parquet_sha256 != catalog["parquet_sha256"]:
        raise DataError("Parquet identity changed; audit the new data snapshot before replay")
    selected = selected_sources(source)
    if set(selected) != set(catalog["sources"]):
        raise DataError("Parquet source selection differs from the audited snapshot")
    # Fail before expensive reconstruction if public cards are absent/incomplete.
    cards = {}
    for family in load_targets()["families"]:
        for entry in family["entries"]:
            relative_dir = FAMILY_DIRECTORIES[family["name"]] + "/" + entry["entry_id"]
            relative = relative_dir + "/model_card.json"
            if relative in cards:
                continue
            if family["name"] == "submissions":
                task, member = 3, f"submissions/task3/{entry['entry_id']}/model_card.json"
            elif family["name"] == "task1_converted":
                task, member = 1, f"submissions/task1/{entry['entry_id']}/model_card.json"
            elif family["name"] == "task2_converted":
                task = 2
                member = str(Path(selected[relative_dir + "/" + entry["prediction_filename"]]["path"]).parent / "model_card.json")
            else:
                task, member = 2, entry["model_card_member"]
            path = model_cards_dir / f"task{task}" / member
            answers = load_answers(path)
            if set(family["required_card_questions"]) - answers.keys():
                raise DataError(f"Public card lacks conversion metadata: {path}")
            cards[relative] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        files = materialize_sources(source, stage, selected)
        for relative, expected in catalog["sources"].items():
            if files[relative] != expected:
                raise DataError(f"Reconstructed parquet records changed: {relative}")
        for relative, card in cards.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(card, destination)
            files[relative] = {"sha256": sha256_file(destination), "bytes": destination.stat().st_size,
                               "source_card": str(card.relative_to(model_cards_dir))}
        families = {
            family["name"]: [{"entry_id": entry["entry_id"],
                              "predictions": f"{FAMILY_DIRECTORIES[family['name']]}/{entry['entry_id']}/{entry['prediction_filename']}",
                              "model_card": f"{FAMILY_DIRECTORIES[family['name']]}/{entry['entry_id']}/model_card.json"}
                             for entry in family["entries"]]
            for family in load_targets()["families"]
        }
        manifest = {"schema_version": 2,
                    "source": {"kind": "parquet", "file": data_file.name,
                               "sha256": source.parquet_sha256, "bytes": data_file.stat().st_size},
                    "files": files, "families": families, "metadata_repairs": [],
                    "parquet_catalog_sha256": sha256_file(CATALOG_PATH)}
        (stage / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "metadata_repairs.json").write_text("[]\n")
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest
