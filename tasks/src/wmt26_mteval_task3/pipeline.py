"""Score the five frozen Task 3 tracks from prepared single-dump inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .canonical_reference import load_canonical_human_reference
from .inventory import load_official_inventory
from .io import DataError, load_json, sha256_file
from .model_cards import load_base_model_overrides
from .models import SubmissionSpec
from .preparation import validate_inputs
from .reproduction import load_targets
from . import reporting, task1_reporting, task2_reporting
from .submissions import evaluate_submission
from .task1_conversion import (
    build_task1_display_names, convert_task1_and_evaluate, inspect_task1_decision,
)
from .task2_conversion import (
    build_conversion_display_names, convert_and_evaluate, inspect_conversion_decision,
)


PACKAGE_ROOT = Path(__file__).resolve().parent


def load_context(root: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Validate prepared provenance and load the common official inventory/evidence."""
    manifest = validate_inputs(root)
    inventory = load_official_inventory(root / "test-set-files/mteval-test26.jsonl")
    reference = load_canonical_human_reference(root / "human_annotations/ground-truth.jsonl", inventory)
    if reference.stats != load_targets()["human_reference"]:
        raise DataError("Human population differs from the frozen snapshot")
    return inventory, reference, manifest


def conversion_decisions(root: Path, task: int, baseline: bool = False) -> tuple[Path, list[Any], dict[Any, str]]:
    """Read declared conversion rules, preserving opt-outs and organizer exceptions."""
    family_name = f"task{task}_baselines_converted" if baseline else f"task{task}_converted"
    family = next(f for f in load_targets()["families"] if f["name"] == family_name)
    directory = root / ("baselines" if baseline else f"submissions/task{task}")
    overrides = load_base_model_overrides(PACKAGE_ROOT / "base_model_overrides.json")
    decisions = []
    for entry in family["entries"]:
        entry_id = entry["entry_id"] if baseline else int(entry["entry_id"])
        decision = inspect_task1_decision(entry_id, directory) if task == 1 else inspect_conversion_decision(
            entry_id, directory, None if baseline else overrides["task2_participant"])
        if baseline:
            if task == 2 and (decision.threshold != 85 or decision.inclusive is not True):
                raise DataError("Historical organizer Task 2 rule must be score >= 85")
            decision = replace(decision, uses_reference=False, status="converted",
                               reason="Configured organizer baseline; model-card permission is not applied")
        decisions.append(decision)
    counts = dict(Counter(d.status for d in decisions))
    if counts != family["historical_summary"]["status_counts"]:
        raise DataError(f"{family_name}: conversion-status population changed: {counts}")
    names = build_task1_display_names(decisions) if task == 1 else build_conversion_display_names(decisions)
    return directory, decisions, names


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a nonempty generated CSV with deterministic field and row order."""
    if not rows:
        raise DataError(f"No rows for {path}")
    reporting.write_csv(path, rows, list(rows[0]))


def write_run_manifest(output: Path, inputs: Path, *, role: str = "five-family scoring") -> None:
    """Bind generated files, implementation/resources and environment to staged inputs."""
    implementation = {str(p.relative_to(PACKAGE_ROOT)): sha256_file(p)
                      for p in sorted(PACKAGE_ROOT.rglob("*"))
                      if p.suffix in {".py", ".json", ".tex", ".csv", ".toml"}}
    outputs = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*"))
               if p.is_file() and p.suffix in {".csv", ".md", ".tex", ".pdf", ".json"}
               and p.name not in {"run_manifest.json", "input_audit.json", "reproduction_check.json"}}
    environment = {name: importlib.metadata.version(name)
                   for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "pyarrow")}
    input_manifest = load_json(inputs / "input_manifest.json")
    reporting.write_json(output / "run_manifest.json", {
        "schema_version": 1, "role": role,
        "input_manifest_sha256": sha256_file(inputs / "input_manifest.json"),
        "python": sys.version.split()[0], "environment": environment,
        "implementation": implementation, "outputs": outputs,
        "metadata_policy": ("complete public cards; no repairs"
                            if input_manifest.get("schema_version") == 2
                            else "temporary explicitly authorized card repair, recorded in input manifest"),
    })


def score_families(inputs: Path, output: Path) -> None:
    """Regenerate all family metrics/tables from predictions, never from target CSVs."""
    inventory, reference, manifest = load_context(inputs)
    summaries, _ = reporting.human_reference_rows(inventory, reference)
    direct_family = load_targets()["families"][0]
    specs = [SubmissionSpec(int(e["entry_id"]), "",
                            manifest["files"][f"submissions/task3/{e['entry_id']}/{e['prediction_filename']}"]["sha256"])
             for e in direct_family["entries"]]
    overrides = load_base_model_overrides(PACKAGE_ROOT / "base_model_overrides.json")
    metadata, raw_metrics = [], []
    for spec in specs:
        print(f"Scoring direct Task 3 {spec.submission_id}", flush=True)
        info, rows = evaluate_submission(spec, inputs / "submissions/task3", inventory, reference,
                                        base_model_overrides=overrides["task3_direct"])
        metadata.append(info)
        raw_metrics.extend(rows)
    names = reporting.build_display_names(metadata)
    metrics = reporting._add_reference_columns(raw_metrics, summaries, metadata, names)
    tables = reporting.table_rows(specs, metrics, summaries, metadata, names)
    directory = output / "submissions"
    _write_rows(directory / "human_reference_summary.csv", summaries)
    _write_rows(directory / "metrics_by_language.csv", metrics)
    _write_rows(directory / "table_by_language.csv", tables)
    _write_rows(directory / "submission_validation.csv", [
        {"submission_id": m.spec.submission_id, "display_name": names[m.spec.submission_id],
         "organization_name": m.organization_name, "submission_name": m.submission_name,
         "base_model": m.base_model, "prediction_sha256": m.prediction_sha256,
         "model_card_sha256": m.model_card_sha256, "attempted_pairs": ";".join(m.attempted_pairs),
         "status": "valid"} for m in metadata])
    reporting.write_markdown_tables(directory / "table_by_language.md", tables, summaries)
    for task, baseline in ((1, False), (2, False), (1, True), (2, True)):
        source, decisions, names = conversion_decisions(inputs, task, baseline)
        family = f"task{task}_baselines_converted" if baseline else f"task{task}_converted"
        print(f"Scoring {family}", flush=True)
        convert = convert_task1_and_evaluate if task == 1 else convert_and_evaluate
        results = [convert(d, names[d.submission_id], source, inventory, reference)
                   for d in decisions if d.status == "converted"]
        renderer = task1_reporting if task == 1 else task2_reporting
        id_column = "entry_key" if baseline else "submission_id"
        metrics = renderer._metric_rows(results, summaries, id_column)
        tables = renderer._table_rows(results, metrics, summaries, id_column)
        info = renderer._conversion_info_rows(decisions, names, results, id_column)
        directory = output / family
        for name, rows in (("metrics_by_language", metrics), ("table_by_language", tables),
                           ("conversion_info", info)):
            _write_rows(directory / f"{name}.csv", rows)
        reporting.write_markdown_tables(directory / "table_by_language.md", tables, summaries)
        renderer._write_conversion_markdown(directory / "conversion_info.md", info, id_column)
    write_run_manifest(output, inputs)


def main(argv: list[str] | None = None) -> int:
    """Score the frozen families using standard release input/output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/task3"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task3"))
    args = parser.parse_args(argv)
    score_families(args.data_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
