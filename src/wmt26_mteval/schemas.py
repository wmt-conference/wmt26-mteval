"""Arrow schemas and field descriptions for every config of the release.

This module is the single source of truth for the columns of each config.  Each field carries a
short ``description`` in its metadata, which the documentation renders as the field glossary.  A
config is added by defining its fields below and registering the schema in ``SCHEMAS``; fields
that several configs share are defined once and reused.
"""

from __future__ import annotations

import pyarrow as pa


def _field(name: str, type_: pa.DataType, description: str) -> pa.Field:
    """A nullable field whose metadata carries its short description."""
    return pa.field(name, type_, metadata={"description": description})


ITEM_ID = _field(
    "item_id",
    pa.string(),
    "Unique identifier of the test item: six fields (collection, source language, target "
    "language, domain or challenge-set name, document, paragraph or segment) joined by `_###_`.",
)

LP = _field(
    "lp",
    pa.string(),
    "Language pair as a short `src-tgt` code, such as `ces-deu` or `eng-zho_CN`.",
)

# The columns parsed from ``item_id`` that every per-item config repeats, in this order.
SHARED_FIELDS: tuple[pa.Field, ...] = (
    _field(
        "collection",
        pa.string(),
        "Either `official` for the main test set or `challenge` for a challenge set.",
    ),
    _field(
        "challenge_set",
        pa.string(),
        "Name of the challenge set the item belongs to (`GAMBITP`, `FECE`, `QEbreak`, `AGGREE`, "
        "`LIGATUR`, `RefQuality-PD` or `RefQuality-PE`), or null for official items.",
    ),
    _field(
        "domain",
        pa.string(),
        "Text domain (`news`, `social`, `speech`, `software`, `edu` or `factchecking`) for "
        "official and RefQuality items, or null for every other challenge item.",
    ),
    _field(
        "src_lang",
        pa.string(),
        "Source language, normalised to a three-letter language code, a script and, where "
        "needed, a region (for example `eng_Latn` or `zho_Hant_TW`).",
    ),
    _field(
        "tgt_lang",
        pa.string(),
        "Target language, normalised to a three-letter language code, a script and, where "
        "needed, a region (for example `deu_Latn` or `zho_Hans`).",
    ),
    _field(
        "src_lang_orig",
        pa.string(),
        "Source language code exactly as it is written in `item_id` (for example `en` or "
        "`eng_Latn`).",
    ),
    _field(
        "tgt_lang_orig",
        pa.string(),
        "Target language code exactly as it is written in `item_id` (for example `de_DE` or "
        "`deu_Latn`).",
    ),
    LP,
    _field(
        "is_priority",
        pa.bool_(),
        "True for official items whose language pair is one of the eight priority directions, and "
        "false for every other item, challenge items included.",
    ),
    _field(
        "doc_id",
        pa.string(),
        "Identifier of the source document the item belongs to, the fifth field of `item_id`.",
    ),
    _field(
        "para_idx",
        pa.int32(),
        "Zero-based index of the paragraph within its document, taken from the last field of "
        "`item_id`, and null for FECE and QEbreak, whose item ids end in a segment hash.",
    ),
)

SRC = _field("src", pa.string(), "Source text of the item, exactly as in the test set.")
REF_TEXT = _field(
    "ref_text",
    pa.string(),
    "Reference translation, exactly as in the test set, or null when the item has no reference.",
)
REF_TYPE = _field(
    "ref_type",
    pa.string(),
    "How the reference was produced: `native` (human translation from scratch), `postedit` "
    "(post-edited machine translation), `pseudo` (machine-generated pseudo-reference), "
    "`unknown` (a reference text is present but its type is not recorded) or `none` (no "
    "reference text).",
)

SYSTEM = _field(
    "system",
    pa.string(),
    "Name of the system that produced the hypothesis, exactly as in the source data.",
)

SYSTEM_TYPE = _field(
    "system_type",
    pa.string(),
    "Kind of system: `mt`, `human_scratch`, `human_postedit`, `pseudo_ref` or `challenge_hyp`.",
)

RUN_ID = _field(
    "run_id",
    pa.string(),
    "Identifier of the evaluator run: the numeric submission id, or the name of the "
    "organisers' baseline (for example `gemma_ref` or `872013`).",
)

# The columns of the ``submissions`` and ``baselines`` configs: one row per run, item and system.
T1_ERROR_TYPE = pa.struct(
    [
        pa.field("start", pa.int32()),
        pa.field("end", pa.int32()),
        pa.field("severity", pa.string()),
        pa.field("category", pa.string()),
    ]
)

PREDICTION_FIELDS: tuple[pa.Field, ...] = (
    RUN_ID,
    _field(
        "task",
        pa.string(),
        "Task the run was entered for: `task1` (error spans, with Task 2 scores as well for the "
        "organisers' Task 1 baselines), `task2` (segment scores) or `task3` (error-free "
        "judgements).",
    ),
    ITEM_ID,
    LP,
    SYSTEM,
    _field(
        "t1_errors",
        pa.list_(T1_ERROR_TYPE),
        "Task 1 error spans the run predicts for the translation, each with `start` and `end` "
        "offsets into the hypothesis, a `severity` and a `category` (which may be null), or null "
        "where the run makes no Task 1 prediction for the row.",
    ),
    _field(
        "t1_omission",
        pa.string(),
        "Omission severity (`minor` or `major`) that the run predicts for the translation, or "
        "null when it predicts no omission or makes no Task 1 prediction.",
    ),
    _field(
        "t1_instruction_fault",
        pa.string(),
        "Instruction fault that the run reports for the translation, or null when it reports "
        "none or makes no Task 1 prediction (null in every row of this release).",
    ),
    _field(
        "t2_score",
        pa.float64(),
        "Task 2 quality score the run predicts for the translation, copied unchanged from the "
        "source (some runs score below 0 or above 100), or null where the run makes no Task 2 "
        "prediction.",
    ),
    _field(
        "t3_error_free",
        pa.bool_(),
        "Task 3 judgement that the translation is free of errors, or null where the run makes "
        "no Task 3 prediction.",
    ),
)


def _flag(true: str, false: str) -> str:
    """Description of a Boolean card answer: what true and false mean, and when it is null."""
    return (
        f"True when the model card says {true}, false when it says {false}, and null when the "
        "answer is missing or is neither yes nor no."
    )


def _conversion(source: str, target: str) -> str:
    """Description of one of the three task-conversion permissions on the model card."""
    return (
        f"True when the team allows the organisers to convert its {source} output into a "
        f"{target} prediction automatically, false when it does not, and null when the model "
        "card gives no readable answer."
    )


# The columns of the ``systems`` config: one row per official run, from its scrubbed model card.
SYSTEMS_FIELDS: tuple[pa.Field, ...] = (
    RUN_ID,
    _field(
        "name",
        pa.string(),
        "Name the team gave the submission in its model card, followed by the run id in "
        "brackets where several runs share that name, so that every name is unique.",
    ),
    _field(
        "canonical_name",
        pa.string(),
        "Name under which the findings paper lists the system, curated by the organisers and "
        "never suffixed with the run id.",
    ),
    _field(
        "org",
        pa.string(),
        "Organisation or organisations that entered the run, as written in the model card.",
    ),
    _field(
        "description",
        pa.string(),
        "Short free-text description of the system, copied from the model card.",
    ),
    _field(
        "reference_usage",
        pa.string(),
        "Free-text account of which kinds of reference translation the system uses and how, "
        "copied from the model card.",
    ),
    _field(
        "model_size",
        pa.string(),
        "Size of the model behind the system, as the team wrote it in the model card.",
    ),
    _field(
        "closed_model_name",
        pa.string(),
        "Name of the closed-source model the system calls, as written in the model card.",
    ),
    _field(
        "api_provider",
        pa.string(),
        "Provider or host of the API through which the closed-source model is called, as "
        "written in the model card.",
    ),
    _field(
        "link",
        pa.string(),
        "Link to the submitted system that the team volunteered for publication in its model "
        "card, or an empty string where it gave none.",
    ),
    _field(
        "underlying_model",
        pa.string(),
        "Model or models behind the system as the findings paper reports them, curated by the "
        "organisers (`—` for the metrics that use no model, and `not stated in the paper` "
        "where the paper gives none).",
    ),
    _field(
        "paper_citation",
        pa.string(),
        "Citation of the paper that describes the system, in the author-year form of the "
        "findings paper, or `[Citation missing]` where the findings paper has none.",
    ),
    _field(
        "is_baseline",
        pa.bool_(),
        "True for the organisers' baselines and false for participant submissions.",
    ),
    _field(
        "uses_reference",
        pa.bool_(),
        _flag("that the system uses the provided reference translations", "that it does not"),
    ),
    _field(
        "ensemble",
        pa.bool_(),
        _flag("that the system is an ensemble", "that it is not"),
    ),
    _field(
        "fine_tuned",
        pa.bool_(),
        _flag("that the team fine-tuned its model", "that it did not"),
    ),
    _field(
        "open_weight",
        pa.bool_(),
        _flag("that the system is based on open-weight models", "that it is not"),
    ),
    _field(
        "open_source",
        pa.bool_(),
        _flag("that the submitted system is open-source", "that it is not"),
    ),
    _field(
        "threshold_inclusive",
        pa.bool_(),
        "True when a Task 2 score equal to `t3_threshold` counts as error-free, false when only "
        "a higher score does, and null when the model card says neither or both.",
    ),
    _field("allow_t1_to_t2", pa.bool_(), _conversion("Task 1", "Task 2")),
    _field("allow_t1_to_t3", pa.bool_(), _conversion("Task 1", "Task 3")),
    _field("allow_t2_to_t3", pa.bool_(), _conversion("Task 2", "Task 3")),
    _field(
        "covers_all_priority",
        pa.bool_(),
        "True when the run has at least one row for every one of the eight priority language "
        "pairs among the official items.",
    ),
    _field(
        "t3_threshold",
        pa.float64(),
        "Task 2 score that separates the team's Task 2 predictions that are converted to "
        "error-free judgements from those that are not: a higher score counts as error-free, and "
        "a score equal to the threshold does so when `threshold_inclusive` is true. Null where "
        "the model card gives no numeric threshold.",
    ),
    _field(
        "tasks",
        pa.list_(pa.string()),
        "Tasks the run has predictions for: the one task of a submission, `task1` and `task2` "
        "for the organisers' Task 1 baselines, and `task2` for the seven segment-level metric "
        "baselines.",
    ),
    _field(
        "lps",
        pa.list_(pa.string()),
        "Language pairs, as short codes sorted alphabetically, in which the run has at least "
        "one row, challenge sets included.",
    ),
    _field(
        "n_items",
        pa.int64(),
        "Number of distinct test items for which the run has at least one row.",
    ),
    _field(
        "n_pairs",
        pa.int64(),
        "Number of rows (one per item and system) that the run has in its prediction table.",
    ),
)

# The columns of the ``mt_systems`` config: one row per system that has hypotheses in ``pairs``.
MT_SYSTEMS_FIELDS: tuple[pa.Field, ...] = (
    SYSTEM,
    SYSTEM_TYPE,
    _field(
        "n_items",
        pa.int64(),
        "Number of distinct test items for which the system has a hypothesis.",
    ),
    _field(
        "lps",
        pa.list_(pa.string()),
        "Language pairs, as short codes sorted alphabetically, in which the system has hypotheses.",
    ),
)

# The columns of the ``challenge_sets`` config: one row per challenge set.
CHALLENGE_SETS_FIELDS: tuple[pa.Field, ...] = (
    _field(
        "name",
        pa.string(),
        "Name of the challenge set, as in the `challenge_set` column of `items` and `pairs`.",
    ),
    _field(
        "full_name",
        pa.string(),
        "Title of the paper that introduced the challenge set, copied from the reference list "
        "of the findings paper, or a descriptive name for the organisers' reference-swap sets, "
        "which have no published title.",
    ),
    _field(
        "authors",
        pa.string(),
        "Authors of the challenge set, as in the reference list of the findings paper.",
    ),
    _field(
        "citation",
        pa.string(),
        "Citation of the paper that describes the challenge set, in the author-year form of the "
        "findings paper.",
    ),
    _field(
        "what_it_tests",
        pa.string(),
        "Short description of the phenomenon that the challenge set probes, from the findings "
        "paper.",
    ),
    _field(
        "hyp_semantics",
        pa.string(),
        "What the hypotheses of the set are, and which `system` names they appear under, or "
        "`not stated in the paper` where the paper does not say.",
    ),
    _field(
        "ref_semantics",
        pa.string(),
        "What the reference of the set is, or `not stated in the paper` where the paper does "
        "not say.",
    ),
    _field(
        "lps",
        pa.list_(pa.string()),
        "Language pairs, as short codes sorted alphabetically, in which the set has items.",
    ),
    _field(
        "n_items",
        pa.int64(),
        "Number of distinct items in the challenge set.",
    ),
    _field(
        "n_pairs",
        pa.int64(),
        "Number of rows (one per item and system) that the challenge set has in `pairs`.",
    ),
)

SCHEMAS: dict[str, pa.Schema] = {
    "items": pa.schema(
        [
            ITEM_ID,
            *SHARED_FIELDS,
            SRC,
            REF_TEXT,
            REF_TYPE,
            _field(
                "ref_type_orig",
                pa.string(),
                "Reference type exactly as recorded in the source data (`human`, `postedit`, "
                "`pseudo` or `undefined`), or null when the data records no type.",
            ),
            _field(
                "asr",
                pa.string(),
                "Automatic speech recognition transcript of the source audio, present for "
                "speech items only.",
            ),
            _field(
                "video_path",
                pa.string(),
                "Relative path of the source video file as recorded in the test set, present for "
                "speech items only.",
            ),
            _field(
                "screenshot_path",
                pa.string(),
                "Relative path of the screenshot of the original post, present for some social "
                "media items only.",
            ),
            _field(
                "n_systems",
                pa.int32(),
                "Number of systems (machine translation, human translation, pseudo-reference or "
                "challenge hypothesis) that have a hypothesis for the item.",
            ),
        ]
    ),
    "pairs": pa.schema(
        [
            ITEM_ID,
            *SHARED_FIELDS,
            SYSTEM,
            SYSTEM_TYPE,
            SRC,
            _field(
                "hyp",
                pa.string(),
                "Hypothesis translation, exactly as in the source data, including empty and "
                "whitespace-only outputs.",
            ),
            REF_TEXT,
            REF_TYPE,
            _field(
                "is_reference_candidate",
                pa.bool_(),
                "True when the hypothesis equals the reference apart from surrounding "
                "whitespace, so that it may be a copy of the reference; it marks the reference "
                "itself (a `human_scratch`, `human_postedit` or `pseudo_ref` row) and also any MT "
                "output or challenge hypothesis that equals it, so filter on `system_type` to "
                "tell them apart.",
            ),
            _field(
                "hyp_is_empty",
                pa.bool_(),
                "True when the hypothesis is empty or contains only whitespace.",
            ),
        ]
    ),
    "submissions": pa.schema(PREDICTION_FIELDS),
    "baselines": pa.schema(
        [
            *PREDICTION_FIELDS,
            _field(
                "variant",
                pa.string(),
                "Snapshot of the baseline: `official` for the released predictions, or `alt` for "
                "the original predictions from before deduplication, with Task 1 fields only; the "
                "label is provisional and will be checked against the paper in v1.0.",
            ),
        ]
    ),
    "systems": pa.schema(SYSTEMS_FIELDS),
    "mt_systems": pa.schema(MT_SYSTEMS_FIELDS),
    "challenge_sets": pa.schema(CHALLENGE_SETS_FIELDS),
}
