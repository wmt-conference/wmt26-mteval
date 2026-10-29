"""Paper direction labels and evaluator selection rules."""
from .metadata import DIRECT_ENTRIES

PRIMARY_DIRECTIONS = (
    "ces_Latn-deu_Latn",
    "ces_Latn-vie_Latn",
    "eng_Latn-arz_Arab",
    "eng_Latn-ekk_Latn",
    "eng_Latn-ind_Latn",
    "eng_Latn-isl_Latn",
    "eng_Latn-sme_Latn",
    "zho_Hans-jpn_Jpan",
)


EXCLUDED_DIRECT_ENTRY_IDS = frozenset({
    "867433", "867450", "867457", "870660", "870671", "884877", "884879",
    "884882", "884884", "884885",
})


PLACEHOLDER_ENTRY_IDS = frozenset({"gpt56_sol_xhigh_no_ref_task2"})


PER_DIRECTION_DISPLAY_EXCLUSIONS = {
    "ces_Latn-ukr_Cyrl": frozenset({
        "gemma_ref_task2", "xcomet_xl_ref_task2", "xcomet_xxl_ref_task2",
    }),
}


TEX_DIRECTION_LABELS = {
    "ces_Latn-deu_Latn": "Czech$\\rightarrow$German",
    "ces_Latn-ukr_Cyrl": "Czech$\\rightarrow$Ukrainian",
    "ces_Latn-vie_Latn": "Czech$\\rightarrow$Vietnamese",
    "eng_Latn-arz_Arab": "English$\\rightarrow$Arabic (Egyptian)",
    "eng_Latn-bel_Cyrl": "English$\\rightarrow$Belarusian",
    "eng_Latn-ces_Latn": "English$\\rightarrow$Czech",
    "eng_Latn-deu_Latn": "English$\\rightarrow$German",
    "eng_Latn-ekk_Latn": "English$\\rightarrow$Estonian",
    "eng_Latn-hye_Armn": "English$\\rightarrow$Armenian",
    "eng_Latn-ind_Latn": "English$\\rightarrow$Indonesian",
    "eng_Latn-isl_Latn": "English$\\rightarrow$Icelandic",
    "eng_Latn-kaz_Cyrl": "English$\\rightarrow$Kazakh",
    "eng_Latn-lij_Latn": "English$\\rightarrow$Ligurian",
    "eng_Latn-lld_Latn": "English$\\rightarrow$Ladin",
    "eng_Latn-rus_Cyrl": "English$\\rightarrow$Russian",
    "eng_Latn-sme_Latn": "English$\\rightarrow$Northern S{\\'a}mi",
    "eng_Latn-tha_Thai": "English$\\rightarrow$Thai",
    "eng_Latn-ukr_Cyrl": "English$\\rightarrow$Ukrainian",
    "eng_Latn-zho_Hans": "English$\\rightarrow$Chinese (Simplified)",
    "eng_Latn-zho_Hant_TW": "English$\\rightarrow$Chinese (Traditional, Taiwan)",
    "zho_Hans-jpn_Jpan": "Chinese (Simplified)$\\rightarrow$Japanese",
}


TEX_NAMES = {key: entry.paper_name for key, entry in DIRECT_ENTRIES.items()}
TEX_MODELS = {key: entry.based_on_models for key, entry in DIRECT_ENTRIES.items()}
FINE_TUNED_IDS = frozenset(key for key, entry in DIRECT_ENTRIES.items() if entry.fine_tuned)
