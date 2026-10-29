"""Binary classification metrics for per-language Task 3 evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support


@dataclass
class PairObservations:
    """Store typed pair observations state shared by the original analysis routines."""
    gold: list[bool] = field(default_factory=list)
    predictions: list[bool] = field(default_factory=list)
    item_ids: set[str] = field(default_factory=set)

    def add(self, item_id: str, gold: bool, prediction: bool) -> None:
        """Append one human/prediction observation and record its item identity."""
        self.gold.append(gold)
        self.predictions.append(prediction)
        self.item_ids.add(item_id)

    def metrics(self) -> dict[str, Any]:
        """Return confusion counts, MCC, precision, recall and F1 for accumulated observations."""
        if not self.gold:
            raise ValueError("Cannot compute metrics without observations")
        true_positive = sum(
            gold and prediction
            for gold, prediction in zip(self.gold, self.predictions)
        )
        false_positive = sum(
            not gold and prediction
            for gold, prediction in zip(self.gold, self.predictions)
        )
        false_negative = sum(
            gold and not prediction
            for gold, prediction in zip(self.gold, self.predictions)
        )
        true_negative = sum(
            not gold and not prediction
            for gold, prediction in zip(self.gold, self.predictions)
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.gold,
            self.predictions,
            pos_label=True,
            average="binary",
            zero_division=0,
        )
        mcc = (
            0.0
            if len(set(self.gold)) < 2 or len(set(self.predictions)) < 2
            else float(matthews_corrcoef(self.gold, self.predictions))
        )
        return {
            "annotated_items": len(self.item_ids),
            "annotated_labels": len(self.gold),
            "gold_error_free": sum(self.gold),
            "gold_error": len(self.gold) - sum(self.gold),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "mcc": mcc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
