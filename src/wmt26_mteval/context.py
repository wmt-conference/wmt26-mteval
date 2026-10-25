"""Language normalisation and the per-item columns shared by every config."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from wmt26_mteval.ids import challenge_set, domain, parse_item_id

LANG_CODES_PATH = Path("data") / "metadata" / "lang_codes.tsv"
PRIORITY_PATH = Path("data") / "task1_domain_coverage_sets.json"


class LangCodes:
    """Maps the language codes found in item ids to normalised ``xxx_Xxxx[_REGION]`` codes."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = dict(mapping)

    @classmethod
    def load(cls, path: Path) -> LangCodes:
        """Read a tab-separated file with a ``orig`` / ``normalised`` header."""
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["orig", "normalised"]:
                raise ValueError(f"{path}: header must be 'orig<TAB>normalised'")
            mapping: dict[str, str] = {}
            for row in reader:
                if row["orig"] in mapping:
                    raise ValueError(f"{path}: duplicate language code {row['orig']!r}")
                mapping[row["orig"]] = row["normalised"]
        return cls(mapping)

    def normalise(self, code: str) -> str:
        """Normalised form of ``code``; ``KeyError`` if the code is not in the table."""
        try:
            return self._mapping[code]
        except KeyError:
            raise KeyError(f"unknown language code {code!r}") from None

    def short(self, norm: str, role: Literal["src", "tgt"]) -> str:
        """Paper-style short code: the first three letters, with the Chinese variants kept apart.

        ``zho_Hant_TW`` is always ``zho_TW``; ``zho_Hans`` is ``zho_CN`` as a target only.
        """
        if role not in ("src", "tgt"):
            raise ValueError(f"role must be 'src' or 'tgt', not {role!r}")
        if norm == "zho_Hant_TW":
            return "zho_TW"
        if norm == "zho_Hans" and role == "tgt":
            return "zho_CN"
        return norm[:3]


@dataclass
class Context:
    """Everything needed to derive the shared per-item columns from an ``item_id``."""

    lang: LangCodes
    priority: frozenset[str]

    @classmethod
    def from_repo(cls, repo: Path) -> Context:
        """Load the language table and priority directions from a repository checkout."""
        repo = Path(repo)
        lang = LangCodes.load(repo / LANG_CODES_PATH)
        coverage = json.loads((repo / PRIORITY_PATH).read_text(encoding="utf-8"))
        return cls(lang, frozenset(coverage["priority_directions"]))

    def describe(self, item_id: str) -> dict:
        """The shared columns for one item.

        ``src_lang`` and ``tgt_lang`` are normalised; the ``*_orig`` values are the codes as they
        appear in the id.  ``para_idx`` is ``None`` for FECE and QEbreak items.
        """
        i = parse_item_id(item_id)
        src = self.lang.normalise(i.src_lang_orig)
        tgt = self.lang.normalise(i.tgt_lang_orig)
        return {
            "collection": i.collection,
            "challenge_set": challenge_set(i),
            "domain": domain(i),
            "src_lang": src,
            "tgt_lang": tgt,
            "src_lang_orig": i.src_lang_orig,
            "tgt_lang_orig": i.tgt_lang_orig,
            "lp": f"{self.lang.short(src, 'src')}-{self.lang.short(tgt, 'tgt')}",
            "is_priority": i.collection == "official" and f"{src}-{tgt}" in self.priority,
            "doc_id": i.doc_id,
            "para_idx": i.para_idx,
        }
