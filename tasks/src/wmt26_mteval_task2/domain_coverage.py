"""Official test-set membership and text-verified reference exclusions."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from .io import _parse_item_id

PRIMARY_DIRECTIONS = (
    "ces_Latn-deu_Latn", "ces_Latn-vie_Latn", "eng_Latn-arz_Arab", "eng_Latn-ekk_Latn",
    "eng_Latn-ind_Latn", "eng_Latn-isl_Latn", "eng_Latn-sme_Latn",
    "zho_Hans-jpn_Jpan",
)



DIRECTION_LABELS = dict(zip(PRIMARY_DIRECTIONS, (
    "Czech→German", "Czech→Vietnamese", "English→Arabic (Egyptian)", "English→Estonian",
    "English→Indonesian", "English→Icelandic", "English→Northern Sámi",
    "Chinese→Japanese",
)))



@dataclass(frozen=True)
class DomainIndex:
    """Official item membership, domain labels, and reference exclusions."""

    path: Path
    test_sets: dict[str, dict[str, dict[str, str]]]
    item_domains: dict[str, str]
    designated_references: dict[str, str]



def file_sha256(path: Path) -> str:
    """Hash a file in bounded memory for reproducible manifests."""

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()



def load_domain_index(path: Path, directions=PRIMARY_DIRECTIONS) -> DomainIndex:
    """Read official domains and derive the same reference policy as Task 1.

    A designated candidate must match ref.text on every official item in its
    direction. Multiple consistent matches are ambiguous and fail explicitly.
    Only system membership and labels are retained, never translation text.
    """

    selected = tuple(sorted(set(directions)))
    if not selected:
        raise ValueError("Select at least one direction")
    test_sets = {direction: {} for direction in selected}
    domains: dict[str, str] = {}
    matches: dict[str, set[str]] = {}
    available = dict.fromkeys(selected, True)
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            context = f"{path}:{number}"
            if not isinstance(record, dict):
                raise ValueError(f"{context}: expected an object")
            item = record.get("item_id")
            set_id, direction = _parse_item_id(item, context)
            if set_id != "official" or direction not in test_sets:
                continue
            if item in domains:
                raise ValueError(f"{context}: duplicate official item {item}")
            hypotheses = record.get("hyps")
            if not isinstance(hypotheses, dict) or not hypotheses or any(
                not isinstance(system, str) or not system or not isinstance(text, str)
                for system, text in hypotheses.items()
            ):
                raise ValueError(f"{context}: invalid hypotheses")
            reference = record.get("ref")
            if reference is not None and not isinstance(reference, dict):
                raise ValueError(f"{context}: ref must be an object or null")
            text = reference.get("text") if reference is not None else None
            if text is not None and not isinstance(text, str):
                raise ValueError(f"{context}: ref.text must be a string or null")
            available[direction] &= text is not None
            matching = {system for system, value in hypotheses.items() if value == text}
            matches[direction] = matches.get(direction, matching) & matching
            test_sets[direction][item] = dict.fromkeys(hypotheses, "")
            domains[item] = item.split("_###_")[3]
    if any(not items for items in test_sets.values()):
        raise ValueError("Selected direction absent from official test set")
    references = {}
    for direction in selected:
        if available[direction]:
            if len(matches[direction]) > 1:
                raise ValueError(f"{direction}: multiple consistent designated candidates")
            if matches[direction]:
                references[direction] = next(iter(matches[direction]))
    return DomainIndex(path, test_sets, domains, references)
