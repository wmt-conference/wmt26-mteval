"""Prepare isolated Task 3 inputs from grand parquet and public model cards.

The legacy ZIP path remains explicit and supports authorized missing-field
repair. Parquet preparation uses complete public cards with no metadata repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .io import DataError, load_json, sha256_file
from .model_cards import load_answers
from .reproduction import load_targets


FAMILY_DIRECTORIES = {
    "submissions": "submissions/task3",
    "task1_converted": "submissions/task1",
    "task2_converted": "submissions/task2",
    "task1_baselines_converted": "baselines",
    "task2_baselines_converted": "baselines",
}


def _repair_card(card: list[dict[str, Any]], questions: list[str], source: Path | None,
                 expected_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Restore only absent conversion fields from a hash-verified original card.

    Existing answers, including false opt-ins and blank thresholds, are never
    overwritten. Return the staged card and the exact non-personal repair rows.
    """
    present = {row["question"] for row in card}
    missing = [question for question in questions if question not in present]
    if not missing:
        return card, []
    if source is None or not source.is_file():
        raise DataError(f"Missing conversion metadata {missing}; supply explicit local card roots")
    if sha256_file(source) != expected_hash:
        raise DataError("Local metadata card does not match the frozen original card hash")
    # Validate duplicate questions without copying private questions or answers.
    load_answers(source)
    original = {row["question"]: row for row in load_json(source)}
    if any(question not in original for question in missing):
        raise DataError("Original card lacks a required conversion question")
    repairs = [{key: original[question][key] for key in ("question", "required", "answer")}
               for question in missing]
    return [*card, *repairs], repairs


def _source_card(family: str, entry_id: str, task1_root: Path | None,
                 task2_root: Path | None) -> Path | None:
    """Resolve a numeric-ID original card only within the explicitly supplied root."""
    if family == "task1_converted" and task1_root is not None:
        matches = [p for p in [task1_root / entry_id / "model_card.json",
                               *task1_root.glob(f"{entry_id}_*/model_card.json")]
                   if p.is_file()]
        if len(matches) != 1:
            raise DataError(f"Expected one original Task 1 card for {entry_id}")
        return matches[0]
    if family == "task2_converted" and task2_root is not None:
        return task2_root / f"{entry_id}_model_card.json"
    return None


def prepare_inputs(dump: Path, output: Path, *, task1_cards: Path | None = None,
                   task2_cards: Path | None = None, include_raw: bool = False,
                   allow_repacked_dump: bool = False) -> dict[str, Any]:
    """Atomically stage selected ZIP payloads, repaired cards and a provenance manifest.

    Refuse existing destinations. A repacked ZIP is allowed only explicitly;
    frozen shared-reference and prediction hashes must still match exactly.
    """
    targets = load_targets()
    if output.exists():
        raise DataError(f"Refusing existing input directory: {output}")
    dump_hash = sha256_file(dump)
    if dump_hash != targets["dump"]["sha256"] and not allow_repacked_dump:
        raise DataError("ZIP identity changed; inspect it before using --allow-repacked-dump")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "archive": {"file": dump.name, "sha256": dump_hash, "bytes": dump.stat().st_size},
        "files": {}, "families": {}, "metadata_repairs": [],
        "targets_sha256": hashlib.sha256(json.dumps(targets, sort_keys=True).encode()).hexdigest(),
    }
    try:
        with zipfile.ZipFile(dump) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise DataError("Duplicate member names in ZIP")

            def copy_member(member: str, relative: str, expected: str) -> None:
                """Copy one explicitly named member and bind its exact released bytes."""
                if relative in manifest["files"]:
                    if manifest["files"][relative]["sha256"] != expected:
                        raise DataError("Conflicting selected payloads")
                    return
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                actual = sha256_file(path)
                if actual != expected:
                    raise DataError(f"{member}: frozen payload hash mismatch")
                manifest["files"][relative] = {"sha256": actual, "bytes": path.stat().st_size,
                                               "source_member": member}

            for shared in targets["shared_inputs"]:
                if shared["member"] == "annotations.json" and not include_raw:
                    continue
                copy_member(shared["member"], shared["member"], shared["sha256"])
            for family in targets["families"]:
                selected = []
                for entry in family["entries"]:
                    prefixes = [p for p in entry["dump_candidates"] if p + "/model_card.json" in names]
                    if len(prefixes) != 1:
                        raise DataError(f"Ambiguous/missing selected directory: {entry['entry_id']}")
                    prefix = prefixes[0]
                    relative_dir = FAMILY_DIRECTORIES[family["name"]] + "/" + entry["entry_id"]
                    prediction_name = entry["prediction_filename"]
                    prediction = relative_dir + "/" + prediction_name
                    copy_member(prefix + "/" + prediction_name, prediction,
                                entry["historical_prediction_sha256"])
                    card_relative = relative_dir + "/model_card.json"
                    if card_relative not in manifest["files"]:
                        card_member = entry.get("model_card_member", prefix + "/model_card.json")
                        raw = archive.read(card_member)
                        card = json.loads(raw)
                        if not isinstance(card, list):
                            raise DataError(f"{card_member}: invalid card")
                        missing = set(family["required_card_questions"]) - {row["question"] for row in card}
                        source = _source_card(family["name"], entry["entry_id"], task1_cards, task2_cards) if missing else None
                        card, repairs = _repair_card(card, family["required_card_questions"], source,
                                                     entry["historical_model_card_sha256"])
                        path = stage / card_relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n")
                        load_answers(path)
                        manifest["files"][card_relative] = {
                            "sha256": sha256_file(path), "bytes": path.stat().st_size,
                            "source_member": card_member,
                            "source_member_sha256": hashlib.sha256(raw).hexdigest(),
                        }
                        if repairs:
                            manifest["metadata_repairs"].append({
                                "family": family["name"], "entry_id": entry["entry_id"],
                                "member": card_member, "restored_entries": repairs,
                                "original_card_sha256": entry["historical_model_card_sha256"],
                            })
                    selected.append({"entry_id": entry["entry_id"], "predictions": prediction,
                                     "model_card": card_relative})
                manifest["families"][family["name"]] = selected
        (stage / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "metadata_repairs.json").write_text(json.dumps(manifest["metadata_repairs"], indent=2) + "\n")
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def validate_inputs(root: Path) -> dict[str, Any]:
    """Reject edited/stale staged files and catalogs before any result calculation."""
    manifest = load_json(root / "input_manifest.json")
    version = manifest.get("schema_version")
    if version not in {1, 2}:
        raise DataError("Unsupported prepared-input manifest")
    parquet_sources = None
    if version == 2:
        from .parquet_source import CATALOG_PATH
        catalog_path = CATALOG_PATH
        catalog = load_json(catalog_path)
        if (manifest.get("parquet_catalog_sha256") != sha256_file(catalog_path)
                or manifest.get("source", {}).get("kind") != "parquet"
                or manifest["source"].get("sha256") != catalog["parquet_sha256"]):
            raise DataError("Prepared parquet source/catalog identity changed")
        parquet_sources = catalog["sources"]
        for relative, expected in parquet_sources.items():
            if manifest["files"].get(relative) != expected:
                raise DataError(f"Prepared parquet reconstruction differs from audited source: {relative}")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise DataError("Prepared input escapes its root")
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise DataError(f"Prepared input changed: {relative}")
    targets = load_targets()
    for shared in targets["shared_inputs"]:
        path = root / shared["member"]
        if shared["member"] == "annotations.json" and not path.exists():
            continue
        if parquet_sources is not None:
            expected = parquet_sources.get(shared["member"])
            if expected is None or expected["logical_sha256"] != shared["sha256"]:
                raise DataError(f"Parquet shared source differs from frozen target: {shared['member']}")
            digest = expected["sha256"]
        else:
            digest = shared["sha256"]
        if sha256_file(path) != digest:
            raise DataError(f"Shared input differs from frozen target: {shared['member']}")
    for family in targets["families"]:
        actual = manifest["families"][family["name"]]
        if [e["entry_id"] for e in actual] != [e["entry_id"] for e in family["entries"]]:
            raise DataError("Prepared selection differs from frozen targets")
        for entry, frozen in zip(actual, family["entries"]):
            expected_dir = FAMILY_DIRECTORIES[family["name"]] + "/" + entry["entry_id"]
            if (entry["predictions"] != expected_dir + "/" + frozen["prediction_filename"]
                    or entry["model_card"] != expected_dir + "/model_card.json"
                    or any(entry[key] not in manifest["files"] for key in ("predictions", "model_card"))):
                raise DataError("Prepared payload paths differ from the frozen release layout")
            if parquet_sources is not None:
                expected = parquet_sources[entry["predictions"]]
                if expected["logical_sha256"] != frozen["historical_prediction_sha256"]:
                    raise DataError("Parquet prediction source differs from frozen target")
                digest = expected["sha256"]
            else:
                digest = frozen["historical_prediction_sha256"]
            if sha256_file(root / entry["predictions"]) != digest:
                raise DataError("Prepared prediction differs from frozen target")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Prepare parquet/public cards by default, or an explicitly supplied legacy ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, nargs="?", help="legacy ZIP; omit to use parquet")
    parser.add_argument("--data-file", type=Path, default=Path("wmt26-grand.parquet"))
    parser.add_argument("--model-cards-dir", type=Path, default=Path("model_cards"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/task3"))
    parser.add_argument("--metadata-task1-dir", type=Path)
    parser.add_argument("--metadata-task2-dir", type=Path)
    parser.add_argument("--include-raw-annotations", action="store_true")
    parser.add_argument("--allow-repacked-dump", action="store_true")
    args = parser.parse_args(argv)
    if args.dump is None:
        if (args.metadata_task1_dir or args.metadata_task2_dir or args.include_raw_annotations
                or args.allow_repacked_dump):
            parser.error("Metadata repair, raw annotations and repacked-dump flags apply only to a legacy ZIP")
        from .parquet_source import prepare_parquet_inputs
        manifest = prepare_parquet_inputs(args.data_file, args.output_dir, args.model_cards_dir)
    else:
        manifest = prepare_inputs(args.dump, args.output_dir, task1_cards=args.metadata_task1_dir,
                                  task2_cards=args.metadata_task2_dir, include_raw=args.include_raw_annotations,
                                  allow_repacked_dump=args.allow_repacked_dump)
    print(f"Prepared {len(manifest['files'])} files; repaired {len(manifest['metadata_repairs'])} cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
