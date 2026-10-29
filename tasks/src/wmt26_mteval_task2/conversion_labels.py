"""Frozen direction order and catalog selection for the conversion track."""

from .metadata import CONVERSION_ENTRIES

# The original permutation run used this direction order. Retaining it also
# preserves the last bits of floating-point aggregate reductions.
CONVERSION_DIRECTIONS = (
    "ces_Latn-deu_Latn", "ces_Latn-vie_Latn", "zho_Hans-jpn_Jpan",
    "eng_Latn-arz_Arab", "eng_Latn-ekk_Latn", "eng_Latn-isl_Latn",
    "eng_Latn-ind_Latn", "eng_Latn-sme_Latn",
)
CONVERSION_FINE_TUNED = frozenset(
    identity for identity, entry in CONVERSION_ENTRIES.items() if entry.fine_tuned)
