"""Parsing of ``item_id`` and the small classifiers built on it.

An item id has six fields joined by ``_###_``::

    <collection>_###_<src>_###_<tgt>_###_<field4>_###_<doc>_###_<para>
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SEPARATOR = "_###_"

NAMED_CHALLENGE_SETS = ("GAMBITP", "FECE", "QEbreak", "AGGREE", "LIGATUR")
REFQUALITY_SETS = ("RefQuality-PD", "RefQuality-PE")

# FECE and QEbreak close their ids with a segment hash, not a paragraph number.  Some of those
# hashes are all digits (35 in the release data), so the rule is by challenge set, never by int().
_SEGMENT_HASH_SETS = frozenset({"FECE", "QEbreak"})

_DIGITS = re.compile(r"[0-9]+")
_CHALLENGE_HYP = re.compile(r"[A-Za-z]+_hyp[0-9]+")

# The source type "undefined" (and a missing type) is not in this table: it becomes ``unknown`` when
# a reference text is present and ``none`` when there is none (see ``map_ref_type``).
_REF_TYPES = {"human": "native", "postedit": "postedit", "pseudo": "pseudo"}
UNDEFINED = "undefined"


@dataclass(frozen=True)
class ItemId:
    """The six fields of an item id; ``para_idx`` is ``None`` for FECE and QEbreak items."""

    collection: str
    src_lang_orig: str
    tgt_lang_orig: str
    field4: str
    doc_id: str
    para_idx: int | None


def parse_item_id(s: str) -> ItemId:
    """Split an item id into its fields; ``ValueError`` if it is not a well-formed id."""
    parts = s.split(SEPARATOR)
    if len(parts) != 6:
        raise ValueError(f"item_id must have 6 fields separated by {SEPARATOR!r}: {s!r}")
    collection, src, tgt, field4, doc_id, last = parts
    if field4 in _SEGMENT_HASH_SETS:
        if not last:
            raise ValueError(f"item_id has an empty segment hash: {s!r}")
        para_idx = None
    elif _DIGITS.fullmatch(last):
        para_idx = int(last)
    else:
        raise ValueError(f"item_id paragraph index is not an integer: {s!r}")
    return ItemId(collection, src, tgt, field4, doc_id, para_idx)


def challenge_set(i: ItemId) -> str | None:
    """Name of the challenge set, or ``None`` for official items.

    A challenge item that belongs to no known set raises ``ValueError`` rather than getting a null.
    """
    if i.collection == "official":
        return None
    if i.collection != "challenge":
        raise ValueError(f"unknown collection {i.collection!r}")
    if i.field4 in NAMED_CHALLENGE_SETS:
        return i.field4
    prefix = i.doc_id.split("_", 1)[0]
    if prefix in REFQUALITY_SETS:
        return prefix
    raise ValueError(f"challenge item belongs to no known challenge set: {i}")


def domain(i: ItemId) -> str | None:
    """The domain (``field4``) for official and RefQuality items, otherwise ``None``."""
    if i.collection == "official":
        return i.field4
    return i.field4 if challenge_set(i) in REFQUALITY_SETS else None


def map_ref_type(ref: dict | None) -> tuple[str, str | None]:
    """Return ``(ref_type, ref_type_orig)`` for a ``ref`` cell.

    ``ref_type`` is ``native``, ``postedit`` or ``pseudo`` for the three recorded types, and
    ``none`` exactly when the item has no reference text: the ref is absent, or its text is null
    and its type is null or ``"undefined"``.  A reference text whose type is not recorded (type
    ``"undefined"`` or null) is ``unknown`` (Ruling R23).  ``ref_type_orig`` keeps the source
    spelling: ``None`` for an absent ref or a null type, ``"undefined"`` for the explicit marker.
    """
    if ref is None:
        return "none", None
    orig = ref.get("type")
    if orig is None or orig == UNDEFINED:
        return ("none" if ref.get("text") is None else "unknown"), orig
    try:
        return _REF_TYPES[orig], orig
    except KeyError:
        raise ValueError(f"unknown reference type {orig!r}") from None


# The values of ``system_type``: the reference systems are the three between ``mt`` and
# ``challenge_hyp``.
SYSTEM_TYPES = ("mt", "human_scratch", "human_postedit", "pseudo_ref", "challenge_hyp")


def classify_mt_system(name: str) -> str:
    """Classify a hypothesis system name.

    One of ``SYSTEM_TYPES``: ``mt``, ``human_scratch``, ``human_postedit``, ``pseudo_ref`` or
    ``challenge_hyp``.
    """
    if name == "Human (from scratch)":
        return "human_scratch"
    if name == "Human (postediting)":
        return "human_postedit"
    if name == "PseudoRef":
        return "pseudo_ref"
    if _CHALLENGE_HYP.fullmatch(name):
        return "challenge_hyp"
    return "mt"


def is_reference_candidate(hyp: str, ref_text: str | None) -> bool:
    """True when the hypothesis equals the reference, ignoring surrounding whitespace."""
    return ref_text is not None and hyp.strip() == ref_text.strip()
