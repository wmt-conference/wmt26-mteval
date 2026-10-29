"""Audit the sole release ZIP and check generated CSVs against frozen targets.

This module deliberately does not score submissions or recover missing card
answers from another checkout. The target catalog contains expected results,
not replacement evaluation inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any, BinaryIO


class ReproductionError(ValueError):
    """A release input or generated result violates a reproduction target."""


def load_targets() -> dict[str, Any]:
    """Return the packaged target catalog without consulting an analysis checkout."""
    return json.loads(files("wmt26_mteval_task3").joinpath("targets.json").read_text())


def stream_sha256(handle: BinaryIO) -> str:
    """Return the SHA-256 of a binary stream, reading in bounded-size chunks."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def csv_projection(path: Path, columns: list[str], keys: list[str]) -> dict[str, Any]:
    """Hash a row-order-independent CSV projection; reject duplicate result keys.

    Values remain CSV strings, preserving the frozen numerical precision. Paths,
    display names and provenance columns need not match when they are not selected.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns + keys) - set(reader.fieldnames or [])
        if missing:
            raise ReproductionError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    identities = [tuple(row[key] for key in keys) for row in rows]
    if len(identities) != len(set(identities)):
        raise ReproductionError(f"{path}: duplicate result keys")
    rows.sort(key=lambda row: tuple(row[key] for key in keys))
    projected = [[row[column] for column in columns] for row in rows]
    data = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
    return {"rows": len(rows), "projection_sha256": hashlib.sha256(data.encode()).hexdigest()}


def verify_results(root: Path, targets: dict[str, Any]) -> dict[str, Any]:
    """Check every required CSV under an output root; report all missing/mismatched files."""
    checks = []
    for target in targets["artifacts"]:
        path = root / target["path"]
        check: dict[str, Any] = {"path": target["path"], "matches": False}
        try:
            actual = csv_projection(path, target["columns"], target["keys"])
            check.update(actual)
            check["matches"] = all(actual[key] == target[key] for key in actual)
            if "historical_file_sha256" in target:
                with path.open("rb") as handle:
                    check["byte_identical_to_historical"] = stream_sha256(handle) == target["historical_file_sha256"]
        except (OSError, ReproductionError, KeyError) as error:
            check["error"] = str(error)
        checks.append(check)
    for target in targets.get("presentation_artifacts", []):
        check = {"path": target["path"], "type": "presentation", "matches": False}
        try:
            with (root / target["path"]).open("rb") as handle:
                check["sha256"] = stream_sha256(handle)
            check["matches"] = check["sha256"] == target["sha256"]
        except OSError as error:
            check["error"] = str(error)
        checks.append(check)
    return {"matches": all(check["matches"] for check in checks), "checks": checks}


def card_answers(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    """Read unique question/answer entries; never return them in an audit report."""
    card = json.loads(archive.read(member))
    if not isinstance(card, list):
        raise ReproductionError(f"{member}: expected a list of model-card entries")
    answers = {}
    for row in card:
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            raise ReproductionError(f"{member}: malformed model-card entry")
        question = row["question"]
        if question in answers:
            raise ReproductionError(f"{member}: duplicate model-card question")
        if "answer" not in row:
            raise ReproductionError(f"{member}: missing answer field")
        answers[question] = row["answer"]
    return answers


def audit_dump(path: Path, targets: dict[str, Any]) -> dict[str, Any]:
    """Inspect only the supplied ZIP for identities, payloads and missing metadata.

    Selected payload hashes come from the old aggregate manifests. Historical
    baseline predictions and their model cards have separate paths in the ZIP;
    the catalog pins those paths rather than silently choosing a newer snapshot.
    """
    with path.open("rb") as handle:
        archive_hash = stream_sha256(handle)
    checks = []
    cards = []
    blockers = []
    if archive_hash != targets["dump"]["sha256"]:
        blockers.append({"reason": "archive differs from the inspected release snapshot"})
    with zipfile.ZipFile(path) as archive:
        inventory = Counter(archive.namelist())
        duplicates = sorted(name for name, count in inventory.items() if count > 1)
        if duplicates:
            raise ReproductionError(f"ZIP contains duplicate member names: {duplicates}")
        for target in targets["shared_inputs"]:
            member = target["member"]
            check: dict[str, Any] = {"member": member, "matches": False}
            if member in inventory:
                with archive.open(member) as handle:
                    digest = stream_sha256(handle)
                check.update(sha256=digest, bytes=archive.getinfo(member).file_size)
                check["matches"] = digest == target["sha256"]
            checks.append(check)
        for family in targets["families"]:
            for entry in family["entries"]:
                candidates = [prefix for prefix in entry["dump_candidates"]
                              if prefix + "/model_card.json" in inventory]
                if len(candidates) != 1:
                    blockers.append({"family": family["name"], "entry_id": entry["entry_id"],
                                     "reason": "expected one selected payload directory",
                                     "found": len(candidates)})
                    continue
                prefix = candidates[0]
                member = prefix + "/" + entry["prediction_filename"]
                check = {"member": member, "matches": False}
                if member in inventory:
                    with archive.open(member) as handle:
                        digest = stream_sha256(handle)
                    check.update(sha256=digest, bytes=archive.getinfo(member).file_size)
                    expected = entry.get("historical_prediction_sha256")
                    # None means inventoried, not verified against historical bytes.
                    check["matches"] = digest == expected if expected else None
                checks.append(check)
                card_member = entry.get("model_card_member", prefix + "/model_card.json")
                answers = card_answers(archive, card_member)
                missing = [question for question in family["required_card_questions"]
                           if question not in answers]
                blank = [question for question in family["required_card_questions"]
                         if question in answers and answers[question] in (None, "")]
                cards.append({"family": family["name"], "entry_id": entry["entry_id"],
                              "member": card_member,
                              "missing_questions": missing, "blank_questions": blank})
        for family in targets["families"]:
            missing_cards = [card for card in cards if card["family"] == family["name"]
                             and card["missing_questions"]]
            if missing_cards:
                blockers.append({"family": family["name"],
                                 "reason": "required conversion metadata removed",
                                 "affected_cards": len(missing_cards)})
    failed = [check["member"] for check in checks if check["matches"] is False]
    if failed:
        blockers.append({"reason": "missing or changed frozen payloads", "members": failed})
    return {
        "schema_version": 1,
        "archive": {"file": path.name, "bytes": path.stat().st_size, "sha256": archive_hash,
                    "matches_inspected_snapshot": archive_hash == targets["dump"]["sha256"]},
        "ready_for_unmodified_historical_runners": False,
        "scope": "input feasibility only; not a scoring or numerical reproduction run",
        "checks": checks,
        "cards": cards,
        "blockers": blockers,
        "remaining_gaps": targets["remaining_gaps"],
    }


def main(argv: list[str] | None = None) -> int:
    """Audit an explicit ZIP or verify result CSVs; exit nonzero on unmet targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    audit = modes.add_parser("audit", help="audit the sole authorized data ZIP")
    audit.add_argument("dump", type=Path)
    verify = modes.add_parser("verify", help="check generated CSVs against frozen targets")
    verify.add_argument("output_root", type=Path)
    args = parser.parse_args(argv)
    targets = load_targets()
    try:
        report = audit_dump(args.dump, targets) if args.mode == "audit" else verify_results(
            args.output_root, targets)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("matches", report.get("ready_for_unmodified_historical_runners")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
