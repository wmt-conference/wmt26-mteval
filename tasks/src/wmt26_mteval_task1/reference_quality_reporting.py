"""Render the paper reference-type and quality figure."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from wmt26_mteval_task1.markdown_reporting import markdown_table, write_markdown_companion
from wmt26_mteval_task1.reference_effect_plotting import ReferenceEffectRow, plot_reference_effects
from wmt26_mteval_task1.reference_quality import (
    ALL_REFERENCE_BASED, COMPARISONS, NATIVE_VS_PSEUDO, NATIVE_VS_POSTEDIT,
    OFFICIAL_VS_PSEUDO, POSTEDIT_VS_PSEUDO, ReferenceQualityResult, WITHOUT_XCOMET,
)
from wmt26_mteval_task1.reporting_policy import is_xcomet_baseline


_CONDITION_FIELDS = {
    OFFICIAL_VS_PSEUDO: ("native_or_postedited", "pseudo"),
    NATIVE_VS_POSTEDIT: ("native", "postedit"),
    NATIVE_VS_PSEUDO: ("native", "pseudo"),
    POSTEDIT_VS_PSEUDO: ("postedit", "pseudo"),
}

_CESA_SCORES_BY_DIRECTION = {
    "ces_Latn-vie_Latn": {"native": 87.4, "postedit": 90.6},
    "ces_Latn-deu_Latn": {"native": 80.7, "postedit": 79.8, "pseudo": 78.4},
    "eng_Latn-isl_Latn": {"native": 87.5, "pseudo": 79.1},
    "eng_Latn-ekk_Latn": {"native": 81.0, "pseudo": 77.1},
    "zho_Hans-jpn_Jpan": {"postedit": 79.6, "pseudo": 72.2},
    "eng_Latn-deu_Latn": {"postedit": 73.0, "pseudo": 59.7},
}

_UNIFIED_NON_XCOMET_COMPARISONS = (
    NATIVE_VS_POSTEDIT,
    NATIVE_VS_PSEUDO,
    POSTEDIT_VS_PSEUDO,
)

_REFERENCE_DISPLAY_NAMES = {
    "native": "Native",
    "postedit": "Post-edited",
    "pseudo": "Pseudo",
}

def _display_reference_type(reference_type: str) -> str:
    """Return the controlled user-facing label for a reference type."""

    if reference_type == "human":
        return "Native"
    return _REFERENCE_DISPLAY_NAMES.get(reference_type, reference_type)


def _score_map(result: ReferenceQualityResult, cohort: str) -> Mapping[str, tuple]:
    """Return per-entry scores for one cohort."""

    if cohort in result.scores_by_cohort_and_comparison:
        return result.scores_by_cohort_and_comparison[cohort]
    base = result.scores_by_comparison
    if cohort == ALL_REFERENCE_BASED:
        return base
    return {
        comparison: tuple(
            score
            for score in scores
            if not is_xcomet_baseline(score.descriptor)
        )
        for comparison, scores in base.items()
    }


def _common_map(
    result: ReferenceQualityResult,
    cohort: str,
) -> Mapping[str, tuple[str, ...]]:
    """Return the fixed analysis directions for one cohort."""

    if cohort in result.common_directions_by_cohort_and_comparison:
        return result.common_directions_by_cohort_and_comparison[cohort]
    return result.common_directions_by_comparison


def _format_float(value: float) -> str:
    """Format a unit-scale metric as a two-decimal percentage."""

    return f"{100 * float(value):.2f}"


def _cohort_direction_rows(
    result: ReferenceQualityResult,
    cohort: str,
) -> list[dict[str, Any]]:
    """Aggregate per-entry direction scores for one cohort and comparison."""

    scores = _score_map(result, cohort)
    common = _common_map(result, cohort)
    rows: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        values = tuple(scores.get(comparison, ()))
        for direction_name in common.get(comparison, ()):
            direction_values = tuple(
                direction
                for score in values
                for direction in score.directions
                if direction.language_pair == direction_name
            )
            if not direction_values:
                continue
            mean = lambda field: sum(
                field(direction) for direction in direction_values
            ) / len(direction_values)
            first = direction_values[0]
            rows.append(
                {
                    "cohort": cohort,
                    "comparison": comparison,
                    "language_pair": direction_name,
                    "model_count": len(direction_values),
                    "left_reference_type": _display_reference_type(first.left_reference_type),
                    "right_reference_type": _display_reference_type(first.right_reference_type),
                    "left_precision": _format_float(mean(lambda d: d.left.precision)),
                    "left_recall": _format_float(mean(lambda d: d.left.recall)),
                    "left_f1": _format_float(mean(lambda d: d.left.f1)),
                    "right_precision": _format_float(mean(lambda d: d.right.precision)),
                    "right_recall": _format_float(mean(lambda d: d.right.recall)),
                    "right_f1": _format_float(mean(lambda d: d.right.f1)),
                    "delta_precision": _format_float(
                        mean(lambda d: d.right.precision - d.left.precision)
                    ),
                    "delta_recall": _format_float(
                        mean(lambda d: d.right.recall - d.left.recall)
                    ),
                    "delta_f1": _format_float(
                        mean(lambda d: d.right.f1 - d.left.f1)
                    ),
                }
            )
    return rows


def _matplotlib() -> Any:
    """Load Matplotlib with a non-interactive backend for report generation."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires Matplotlib. Install the project with "
            "'python -m pip install -e .' to generate the paper figure."
        ) from exc
    return plt


def _cesa_label(language_pair: str, comparison: str) -> str:
    """Return supplied cESA scores for a plotted reference comparison.

    Missing metadata is displayed explicitly so future RefQuality directions can
    still be plotted without silently inventing a cESA value.
    """

    left_condition, right_condition = _CONDITION_FIELDS[comparison]
    scores = _CESA_SCORES_BY_DIRECTION.get(language_pair, {})
    left_score = scores.get(left_condition)
    right_score = scores.get(right_condition)
    if left_score is None or right_score is None:
        return "cESA: not provided"
    return (
        f"cESA: {_REFERENCE_DISPLAY_NAMES[left_condition]} {left_score:.1f} "
        f"vs {_REFERENCE_DISPLAY_NAMES[right_condition]} {right_score:.1f}"
    )


def _unified_non_xcomet_rows(
    direction_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select and order non-xCOMET directional ΔF1 rows for the unified plot."""

    rows_by_key = {
        (str(row["language_pair"]), str(row["comparison"])): row
        for row in direction_rows
        if row["cohort"] == WITHOUT_XCOMET
        and row["comparison"] in _UNIFIED_NON_XCOMET_COMPARISONS
    }
    ordered_directions = tuple(_CESA_SCORES_BY_DIRECTION) + tuple(
        sorted(
            {
                language_pair
                for language_pair, _comparison in rows_by_key
                if language_pair not in _CESA_SCORES_BY_DIRECTION
            }
        )
    )
    selected: list[dict[str, Any]] = []
    for language_pair in ordered_directions:
        for comparison in _UNIFIED_NON_XCOMET_COMPARISONS:
            row = rows_by_key.get((language_pair, comparison))
            if row is None or row.get("delta_f1") in (None, ""):
                continue
            left_condition, right_condition = _CONDITION_FIELDS[comparison]
            selected.append(
                {
                    "language_pair": language_pair,
                    "comparison": comparison,
                    "comparison_label": (
                        f"{_REFERENCE_DISPLAY_NAMES[left_condition]} vs "
                        f"{_REFERENCE_DISPLAY_NAMES[right_condition]}"
                    ),
                    "cesa_label": _cesa_label(language_pair, comparison),
                    "delta_f1": float(row["delta_f1"]),
                }
            )
    return selected


def plot_column_reference_quality(
    path: Path, direction_rows: Iterable[Mapping[str, Any]], plt: Any,
    *, formats: tuple[str, ...] = ("pdf", "png"),
) -> tuple[Path, ...]:
    """Render cohort direction rows in the requested paper-figure formats."""

    rows = []
    for row in _unified_non_xcomet_rows(direction_rows):
        left, right = _CONDITION_FIELDS[row["comparison"]]
        scores = _CESA_SCORES_BY_DIRECTION.get(row["language_pair"], {})
        cesa = (f"cESA {scores[left]:.1f} → {scores[right]:.1f}"
                if left in scores and right in scores else "cESA not provided")
        rows.append(ReferenceEffectRow(
            row["language_pair"], row["comparison_label"].replace(" vs ", " → "),
            cesa, row["delta_f1"],
        ))
    return plot_reference_effects(path, rows, plt,
                                  definition="Reference A → B; ΔF1 = B − A",
                                  formats=formats)


def write_paper_reference_quality_report(
    output_dir: Path, result: ReferenceQualityResult
) -> Path:
    """Write the saved seven-row reference-type and quality PDF from matched scores."""

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = _cohort_direction_rows(result, WITHOUT_XCOMET)
    paths = plot_column_reference_quality(
        plots_dir / "task1-reference-quality-effects",
        rows,
        _matplotlib(),
        formats=("pdf",),
    )
    if len(paths) != 1:
        raise ValueError("reference-quality paper figure has no matched rows")
    plotted = _unified_non_xcomet_rows(rows)
    write_markdown_companion(
        paths[0], "Task 1: reference type and quality effects",
        markdown_table(
            ("Translation direction", "Reference A → B", "Reference quality",
             "Mean ΔF1"),
            (
                (row["language_pair"], row["comparison_label"].replace(" vs ", " → "),
                 row["cesa_label"], f"{row['delta_f1']:+.2f}")
                for row in plotted
            ),
        ) + "\n\nΔF1 = B − A; Gemma 4 and Qwen 3.6 baselines only.",
    )
    return paths[0]
