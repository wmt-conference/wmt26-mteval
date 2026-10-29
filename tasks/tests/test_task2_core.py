from __future__ import annotations

from wmt26_mteval_task2.conversion import task1_to_task2_score
from wmt26_mteval_task2.metadata import DIRECT_ENTRIES, load_catalog
from wmt26_mteval_task2.metrics import score_pdp
from wmt26_mteval_task2.models import Screen
from wmt26_mteval_task2.paper_labels import (
    PER_DIRECTION_DISPLAY_EXCLUSIONS,
    PLACEHOLDER_ENTRY_IDS,
    EXCLUDED_DIRECT_ENTRY_IDS,
)
from wmt26_mteval_task2.preprocessing import aggregate_human_evidence


def test_task1_to_task2_conversion_preserves_published_edge_cases():
    """Count minor/major spans and omissions, including full and empty targets."""

    assert task1_to_task2_score(
        {
            "errors": [
                {"start": 1, "end": 2, "severity": "major"},
                {"start": 4, "end": 5, "severity": "minor"},
            ],
            "omission": "minor",
        },
        10,
    ) == 85.0
    assert task1_to_task2_score(
        {"errors": [{"start": 0, "end": 3, "severity": "major"}]}, 4
    ) == 0.0
    assert task1_to_task2_score(
        {"errors": [], "omission": "major"}, 0
    ) == 0.0


def test_screen_and_source_evidence_average_repeated_annotations():
    """Keep screen-pair means separate from item/system means."""

    item = "official_###_ces_Latn_###_deu_Latn_###_news_###_doc1_###_1"
    direction = "ces_Latn-deu_Latn"
    screens = [
        Screen("s1", item, direction, "annotator1", ("A", "B", "C"),
               {"A": 60, "B": 20, "C": 40}),
        Screen("s2", item, direction, "annotator2", ("A", "B", "C"),
               {"A": 80, "B": 40, "C": 30}),
    ]

    evidence = aggregate_human_evidence(
        {item: {"A": "a", "B": "b", "C": "c"}}, screens, direction
    )

    assert evidence.segment_scores[item] == {"A": 70.0, "B": 30.0, "C": 35.0}
    assert evidence.co_screen_scores[(item, "A", "B")] == (70.0, 30.0)
    assert evidence.statistics.extra_item_system_scores == 3
    assert evidence.statistics.extra_pair_screens == 3


def test_screen_pdp_uses_only_human_co_screen_pairs():
    """PDP compares metric and cESA differences within each retained screen pair."""

    item = "official_###_ces_Latn_###_deu_Latn_###_news_###_doc1_###_1"
    evidence = aggregate_human_evidence(
        {item: {"A": "a", "B": "b", "C": "c"}},
        [Screen("s1", item, "ces_Latn-deu_Latn", "u1", ("A", "B", "C"),
                {"A": 90, "B": 50, "C": 20})],
        "ces_Latn-deu_Latn",
    )
    predictions = {item: {"A": 9.0, "B": 5.0, "C": 2.0}}

    value, support = score_pdp(evidence, predictions, screen_based=True)

    assert value == 1.0
    assert support == 6  # three pairs, both ordered orientations


def test_paper_catalog_and_explicit_exclusion_lists_are_unique_and_present():
    """Check paper evaluator metadata and direction-specific configuration."""

    catalog = load_catalog()
    assert len(catalog["direct"]) == len(DIRECT_ENTRIES)
    assert len(catalog["direct"]) == len(set(catalog["direct"]))
    assert "gpt56_sol_xhigh_no_ref_task2" in PLACEHOLDER_ENTRY_IDS
    assert {"gemma_ref_task2", "xcomet_xl_ref_task2", "xcomet_xxl_ref_task2"} <= set(
        PER_DIRECTION_DISPLAY_EXCLUSIONS["ces_Latn-ukr_Cyrl"]
    )
    assert EXCLUDED_DIRECT_ENTRY_IDS
