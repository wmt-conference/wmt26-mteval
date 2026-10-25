"""Arrow schemas and field descriptions for every config of the release.

This module is the single source of truth for the columns of each config.  Each field carries a
one-sentence ``description`` in its metadata, which the documentation renders as the field
glossary.  A config is added by defining its fields below and registering the schema in
``SCHEMAS``; fields that several configs share are defined once and reused.
"""

from __future__ import annotations

import pyarrow as pa


def _field(name: str, type_: pa.DataType, description: str) -> pa.Field:
    """A nullable field whose metadata carries its one-sentence description."""
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
        "True for official items whose language pair is one of the eight priority directions.",
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
    "(post-edited machine translation), `pseudo` (machine-generated pseudo-reference) or "
    "`none` (no reference).",
)

SYSTEM = _field(
    "system",
    pa.string(),
    "Name of the system that produced the hypothesis, exactly as in the source data.",
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
    _field(
        "run_id",
        pa.string(),
        "Identifier of the evaluator run: the numeric submission id, or the name of the "
        "organisers' baseline (for example `gemma_ref` or `872013`).",
    ),
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
        "offsets into the hypothesis, a `severity` and a `category`, or null where the run makes "
        "no Task 1 prediction for the row.",
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
            _field(
                "system_type",
                pa.string(),
                "Kind of system: `mt`, `human_scratch`, `human_postedit`, `pseudo_ref` or "
                "`challenge_hyp`.",
            ),
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
                "whitespace, so that it may be a copy of the reference.",
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
                "the original predictions from before deduplication, with Task 1 fields only.",
            ),
        ]
    ),
}
