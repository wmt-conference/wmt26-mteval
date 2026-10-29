"""Shared constants for WMT26 Task 1 processing."""

from __future__ import annotations

from typing import Final

ITEM_ID_SEPARATOR = "_###_"
OFFICIAL_SET_ID = "official"
CHALLENGE_SET_ID = "challenge"
ITEM_SET_IDS = frozenset({OFFICIAL_SET_ID, CHALLENGE_SET_ID})

PRIMARY_LANGUAGE_PAIRS = frozenset(
    {
        "ces_Latn-deu_Latn",
        "ces_Latn-vie_Latn",
        "eng_Latn-arz_Arab",
        "eng_Latn-ekk_Latn",
        "eng_Latn-ind_Latn",
        "eng_Latn-isl_Latn",
        "eng_Latn-sme_Latn",
        "zho_Hans-jpn_Jpan",
    }
)

PRIMARY_REFERENCE_TYPE_BY_DIRECTION: Final[dict[str, str]] = {
    "ces_Latn-deu_Latn": "Native",
    "ces_Latn-vie_Latn": "Native",
    "eng_Latn-arz_Arab": "Post-edited",
    "eng_Latn-ekk_Latn": "Native",
    "eng_Latn-ind_Latn": "Pseudo",
    "eng_Latn-isl_Latn": "Native",
    "eng_Latn-sme_Latn": "Native",
    "zho_Hans-jpn_Jpan": "Post-edited",
}

SEVERITY_WEIGHTS = {"minor": 0.25, "major": 1.0}

MATCHING_ALGORITHMS = ("greedy", "optimal")

MPP_METRIC_OPTIONS = {
    "measure": "MPP",
    "matching": "one_to_one",
    "matching_algorithm": "greedy",
    "averaging": "micro",
    "severity_weights": SEVERITY_WEIGHTS,
}


def mpp_metric_options(matching_algorithm: str = "greedy") -> dict[str, object]:
    """Return MPP options for one supported matching algorithm."""

    if matching_algorithm not in MATCHING_ALGORITHMS:
        raise ValueError(
            f"unsupported MPP matching algorithm {matching_algorithm!r}; "
            "expected one of " + ", ".join(MATCHING_ALGORITHMS)
        )
    return {**MPP_METRIC_OPTIONS, "matching_algorithm": matching_algorithm}

ORGANIZATION_NAME_QUESTION = "Organization name"
SUBMISSION_NAME_QUESTION = "Submission name"
SUBMISSION_DESCRIPTION_QUESTION = "Submission description (a short paragraph)"
REFERENCE_DESCRIPTION_QUESTION = (
    "If your system uses reference translations, please briefly describe which "
    "types of references you use (native|postedit|pseudo) and how you use them."
)
MODEL_SIZE_QUESTION = "Model size"
CLOSED_SOURCE_MODEL_QUESTION = "Closed-source model name (e.g. GPT-5.6-luna)"
API_PROVIDER_QUESTION = (
    "Closed-source API provider (Who is the provider/host for this API? e.g. "
    "AWS/Azure/OpenAI/OpenRouter)"
)
SYSTEM_LINK_QUESTION = "Provide link to the submitted system if you have any"

EVALUATOR_CHARACTERISTIC_QUESTIONS = {
    "reference_use": (
        "Does your system use the provided reference translations in any form? "
        "(true|false)"
    ),
    "ensemble": "Is an ensemble (true|false)",
    "supervised": "Is supervised (did you fine-tune your model?) (true|false)",
    "open_weight": "Is based on open-weight models (true|false)",
    "open_source": "Is the submitted system open-source? (true|false)",
}
