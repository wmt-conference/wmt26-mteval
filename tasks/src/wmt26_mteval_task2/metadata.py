"""Read the evaluator catalog shipped with the package."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib.resources import files

BASELINE = "baseline"
EXPLICIT_TASK2_PARTICIPANT = "explicit-task2-participant"


@dataclass(frozen=True)
class EntryMetadata:
    """Identity, model, and display information for a submitted evaluator."""

    submission_id: str
    submission_name: str
    entry_type: str
    uses_references: bool
    based_on_models: str
    paper_name: str = ""
    fine_tuned: bool = False


def load_catalog() -> dict[str, dict[str, EntryMetadata]]:
    """Validate the packaged TSV and return entries indexed by track and ID."""
    catalog = {"direct": {}, "conversion": {}}
    resource = files("wmt26_mteval_task2").joinpath("evaluators.tsv")
    with resource.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            track, identity = row["track"], row["id"]
            if track not in catalog or not identity or identity in catalog[track]:
                raise ValueError(f"Invalid or duplicate catalog identity: {track}/{identity}")
            for flag in ("reference_based", "baseline", "fine_tuned"):
                if row[flag] not in {"true", "false"}:
                    raise ValueError(f"Invalid boolean in evaluator catalog: {identity}/{flag}")
            kind = BASELINE if row["baseline"] == "true" else (
                EXPLICIT_TASK2_PARTICIPANT if track == "direct" else "converted-task1-participant")
            catalog[track][identity] = EntryMetadata(
                identity, row["name"], kind, row["reference_based"] == "true",
                row["model"], row["paper_name"], row["fine_tuned"] == "true")
    return catalog


_CATALOG = load_catalog()
DIRECT_ENTRIES = _CATALOG["direct"]
CONVERSION_ENTRIES = _CATALOG["conversion"]
