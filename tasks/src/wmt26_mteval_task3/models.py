"""Typed data shared across the Task 3 analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedItemId:
    """Store typed parsed item id state shared by the original analysis routines."""
    set_id: str
    source_language: str
    target_language: str
    domain: str
    document_id: str
    segment_id: str

    @property
    def language_pair(self) -> str:
        """Compute language pair from the supplied analysis state and return its representation."""
        return f"{self.source_language}---{self.target_language}"


@dataclass(frozen=True)
class TestItem:
    """Store typed test item state shared by the original analysis routines."""
    item_id: str
    parsed_id: ParsedItemId
    source: str
    hypotheses: dict[str, str]


@dataclass
class OfficialInventory:
    """Store typed official inventory state shared by the original analysis routines."""
    items: dict[str, TestItem]
    item_ids_by_pair: dict[str, set[str]]
    label_counts_by_pair: dict[str, int]
    challenge_items: dict[str, TestItem]
    challenge_item_ids_by_pair: dict[str, set[str]]


@dataclass(frozen=True)
class SubmissionSpec:
    """Store typed submission spec state shared by the original analysis routines."""
    submission_id: int
    archive_sha256: str
    prediction_sha256: str

    @property
    def key(self) -> str:
        """Compute key from the supplied analysis state and return its representation."""
        return str(self.submission_id)


@dataclass(frozen=True)
class NormalizedSpan:
    """Store typed normalized span state shared by the original analysis routines."""
    start_i: int
    end_i: int
    category: Any
    severity: Any
    span_type: str
    offset_unit: str = "utf16_code_unit"

    def as_dict(self) -> dict[str, Any]:
        """Compute as dict from the supplied analysis state and return its representation."""
        return {
            "start_i": self.start_i,
            "end_i": self.end_i,
            "category": self.category,
            "severity": self.severity,
            "span_type": self.span_type,
            "offset_unit": self.offset_unit,
        }


@dataclass(frozen=True)
class HumanJudgment:
    """Store typed human judgment state shared by the original analysis routines."""
    language_pair: str
    item_id: str
    system: str
    annotator_id: str
    submit_time: float
    score: float
    error_free: bool
    spans: tuple[NormalizedSpan, ...]
    group_name: str
    record_index: int
    item_index: int

    @property
    def key(self) -> tuple[str, str]:
        """Compute key from the supplied analysis state and return its representation."""
        return self.item_id, self.system

    @property
    def ordinary_span_count(self) -> int:
        """Compute ordinary span count from the supplied analysis state and return its representation."""
        return sum(span.span_type == "ordinary" for span in self.spans)

    @property
    def omission_count(self) -> int:
        """Compute omission count from the supplied analysis state and return its representation."""
        return sum(span.span_type == "omission" for span in self.spans)

    @property
    def instruction_error_count(self) -> int:
        """Compute instruction error count from the supplied analysis state and return its representation."""
        return sum(span.span_type == "instruction_error" for span in self.spans)


@dataclass
class HumanReference:
    """Store typed human reference state shared by the original analysis routines."""
    labels: dict[tuple[str, str], HumanJudgment]
    withheld_latest_ties: set[tuple[str, str]]
    judgment_counts: dict[tuple[str, str], int]
    annotator_counts: dict[tuple[str, str], int]
    stats: dict[str, int]
    assessments: dict[tuple[str, str], tuple[HumanJudgment, ...]] = field(
        default_factory=dict
    )

    def judgments_for(self, key: tuple[str, str]) -> tuple[HumanJudgment, ...]:
        """Return every retained human assessment for one item-system pair."""
        if self.assessments:
            return self.assessments.get(key, ())
        judgment = self.labels.get(key)
        return () if judgment is None else (judgment,)


@dataclass(frozen=True)
class SubmissionMetadata:
    """Store typed submission metadata state shared by the original analysis routines."""
    spec: SubmissionSpec
    archive_name: str
    archive_sha256: str
    prediction_sha256: str
    model_card_sha256: str
    submission_name: str
    organization_name: str
    attempted_pairs: tuple[str, ...]
    base_model: str = ""
