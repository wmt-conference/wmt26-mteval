from __future__ import annotations

import pytest

from wmt26_mteval_task1.constants import (
    API_PROVIDER_QUESTION,
    CLOSED_SOURCE_MODEL_QUESTION,
    EVALUATOR_CHARACTERISTIC_QUESTIONS,
    MODEL_SIZE_QUESTION,
    REFERENCE_DESCRIPTION_QUESTION,
    SUBMISSION_DESCRIPTION_QUESTION,
    SYSTEM_LINK_QUESTION,
)
from wmt26_mteval_task1.io import (
    load_model_card_template,
    load_ground_truth,
    load_test_set,
    normalize_assessment,
    parse_item_id,
    validate_model_card,
)

from _task1_helpers import (
    assessment,
    item_id,
    major_span,
    model_card,
    model_card_template,
    write_json,
    write_jsonl,
)


def test_parse_item_id_preserves_exact_languages():
    parsed = parse_item_id(item_id("zho_Hant_TW"), "test")

    assert parsed.set_id == "official"
    assert parsed.language_pair == "eng_Latn-zho_Hant_TW"


@pytest.mark.parametrize(
    "value",
    [None, "official_bad", "wmt26_###_a_###_b_###_d_###_x_###_0"],
)
def test_parse_item_id_rejects_noncanonical_values(value):
    with pytest.raises(ValueError):
        parse_item_id(value, "test")


def test_test_set_ignores_challenge_payload_validation(tmp_path):
    official = item_id()
    challenge = item_id(set_id="challenge", document="challenge")
    path = tmp_path / "test.jsonl"
    write_jsonl(
        path,
        [
            {"item_id": official, "hyps": {"system": "abcd"}},
            {"item_id": challenge, "hyps": []},
        ],
    )

    index = load_test_set(path)

    assert index.challenge_row_count == 1
    assert index.official_row_count == 1
    assert index.translation_lengths[official] == {"system": 4}


def test_strict_model_card_uses_template_required_fields(tmp_path):
    template_path = tmp_path / "template.json"
    card_path = tmp_path / "card.json"
    write_json(template_path, model_card_template())
    card = model_card()
    card[0]["answer"] = "  "
    write_json(card_path, card)
    template = load_model_card_template(template_path)

    with pytest.raises(ValueError, match="Organization name.*empty answer"):
        validate_model_card(card_path, template)


@pytest.mark.parametrize("question", ["Participant name", "Email", "Data Collection Consent: test"])
def test_model_card_rejects_private_fields(tmp_path, question):
    """Reject legacy identity or consent fields in cards and templates."""

    template_path = tmp_path / "template.json"
    card_path = tmp_path / "card.json"
    template_entries = model_card_template()
    card_entries = model_card()
    write_json(template_path, template_entries + [{"question": question, "required": False}])
    with pytest.raises(ValueError, match="private question"):
        load_model_card_template(template_path)

    write_json(template_path, template_entries)
    write_json(card_path, card_entries + [{"question": question, "answer": "redacted"}])
    with pytest.raises(ValueError, match="private question"):
        validate_model_card(card_path, load_model_card_template(template_path))


def test_model_card_metadata_includes_ranking_report_fields(tmp_path):
    """The strict reader retains all non-sensitive ranking metadata."""

    template_path = tmp_path / "template.json"
    card_path = tmp_path / "card.json"
    template = model_card_template()
    card = model_card(reference_use="yes")
    text_answers = {
        SUBMISSION_DESCRIPTION_QUESTION: "A detector based on Qwen3-14B.",
        REFERENCE_DESCRIPTION_QUESTION: "Uses native references in the prompt.",
        MODEL_SIZE_QUESTION: "14B",
        CLOSED_SOURCE_MODEL_QUESTION: "",
        API_PROVIDER_QUESTION: "Example provider",
        SYSTEM_LINK_QUESTION: "https://huggingface.co/example/model",
    }
    for question, answer in text_answers.items():
        template.append({"question": question, "required": False, "answer": ""})
        card.append({"question": question, "answer": answer})
    boolean_answers = {
        "reference_use": "yes",
        "ensemble": "false",
        "supervised": "true",
        "open_weight": "true",
        "open_source": "false",
    }
    for key, question in EVALUATOR_CHARACTERISTIC_QUESTIONS.items():
        if key == "reference_use":
            continue
        template.append({"question": question, "required": True, "answer": ""})
        card.append({"question": question, "answer": boolean_answers[key]})

    write_json(template_path, template)
    write_json(card_path, card)
    metadata = validate_model_card(
        card_path,
        load_model_card_template(template_path),
    )

    assert metadata.submission_description == "A detector based on Qwen3-14B."
    assert metadata.reference_description == "Uses native references in the prompt."
    assert metadata.model_size == "14B"
    assert metadata.closed_source_model_name is None
    assert metadata.api_provider == "Example provider"
    assert metadata.system_link == "https://huggingface.co/example/model"
    assert metadata.evaluator_characteristics == {
        "reference_use": True,
        "ensemble": False,
        "supervised": True,
        "open_weight": True,
        "open_source": False,
    }


def test_reference_lists_are_normalized_and_omissions_get_sentinels(tmp_path):
    official = item_id()
    test_path = tmp_path / "test.jsonl"
    ref_path = tmp_path / "ground-truth.jsonl"
    write_jsonl(test_path, [{"item_id": official, "hyps": {"system": "abcd"}}])
    write_jsonl(
        ref_path,
        [
            {
                "item_id": official,
                "task1_pred": {
                    "system": [
                        {
                            **assessment([major_span(0, 2)], omission="minor"),
                            "annotator_id": "human",
                            "context_models": [],
                        }
                    ]
                },
            }
        ],
    )

    references = load_ground_truth(ref_path, load_test_set(test_path))

    assert references.annotator_ids[official]["system"] == ("human",)
    assert references.assessments[official]["system"][0] == [
        {"start": 0, "end": 2, "side": "target", "severity": "major"},
        {"start": 4, "end": 5, "side": "target", "severity": "minor"},
    ]


def test_duplicate_reference_annotators_are_preserved(tmp_path):
    """Load repeated assessments without changing their annotator identity."""

    official = item_id()
    test_path = tmp_path / "test.jsonl"
    ref_path = tmp_path / "ground-truth.jsonl"
    write_jsonl(test_path, [{"item_id": official, "hyps": {"system": "abcd"}}])
    repeated = {
        **assessment([major_span(0, 2)]),
        "annotator_id": "human",
        "context_models": [],
    }
    write_jsonl(
        ref_path,
        [
            {
                "item_id": official,
                "task1_pred": {
                    "system": [
                        repeated,
                        {
                            **assessment([major_span(2, 4)]),
                            "annotator_id": "human",
                            "context_models": [],
                        },
                    ]
                },
            }
        ],
    )

    references = load_ground_truth(ref_path, load_test_set(test_path))

    assert references.annotator_ids[official]["system"] == ("human", "human")
    assert references.assessments[official]["system"] == (
        [{"start": 0, "end": 2, "side": "target", "severity": "major"}],
        [{"start": 2, "end": 4, "side": "target", "severity": "major"}],
    )


def test_unified_screens_load_as_task1_references(tmp_path):
    """Derive Task 1 annotator and context metadata from nested screens."""

    official = item_id()
    test_path = tmp_path / "test.jsonl"
    ref_path = tmp_path / "ground-truth.jsonl"
    write_jsonl(
        test_path,
        [{"item_id": official, "hyps": {"A": "abcd", "B": "b", "C": "c"}}],
    )
    write_jsonl(
        ref_path,
        [
            {
                "item_id": official,
                "screens": [
                    {
                        "screen_id": "eng_Latn_###_deu_Latn_###_0",
                        "annotator_id": "human",
                        "task1_pred": {
                            "B": assessment([]),
                            "A": assessment([major_span(0, 2)], omission="minor"),
                            "C": assessment([]),
                        },
                        "task2_pred": {"B": 50, "A": 20, "C": 90},
                    }
                ],
            }
        ],
    )

    references = load_ground_truth(ref_path, load_test_set(test_path))

    assert references.annotator_ids[official]["A"] == ("human",)
    assert references.assessments[official]["A"][0] == [
        {"start": 0, "end": 2, "side": "target", "severity": "major"},
        {"start": 4, "end": 5, "side": "target", "severity": "minor"},
    ]


def test_zero_length_spans_are_ignored_but_out_of_bounds_fail():
    spans, ignored = normalize_assessment(assessment([major_span(2, 2)]), 4, "test")
    assert spans == []
    assert ignored == 1

    with pytest.raises(ValueError, match="exceeds translation length"):
        normalize_assessment(assessment([major_span(5, 5)]), 4, "test")
