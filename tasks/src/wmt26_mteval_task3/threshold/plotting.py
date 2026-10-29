"""Matplotlib plots for pooled and per-language threshold analyses."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ..io import DataError

from .core import AnalysisResult
from .reporting import fmt_mcc, fmt_threshold

BLUE = "#1f77b4"
GREEN = "#2b8a3e"
ORANGE = "#d17a00"
RED = "#c62828"
GRAY = "#555555"

plt.rcParams.update(
    {
        "axes.edgecolor": "#bbbbbb",
        "axes.labelcolor": "#333333",
        "axes.linewidth": 0.7,
        "font.family": "sans-serif",
        "font.size": 9,
        "pdf.fonttype": 42,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    }
)


def select_language_pairs(
    results: Iterable[AnalysisResult], requested: list[str] | None
) -> list[str]:
    """Validate requested plot pairs or return all available priority pairs."""
    available = sorted(
        {
            language.row["language_pair"]
            for result in results
            for language in result.languages
        }
    )
    if not requested:
        return available
    selected = list(dict.fromkeys(requested))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise DataError(f"Unknown plot language pair(s): {', '.join(unknown)}")
    return selected


def plot_slug(language_pair: str) -> str:
    """Return a filename-safe language-pair identifier."""
    return language_pair.replace("---", "-to-")


def save_pdf(figure: plt.Figure, path: Path) -> None:
    """Save a tightly bounded vector PDF and close its Matplotlib figure."""
    figure.savefig(
        path,
        format="pdf",
        facecolor="white",
        metadata={
            "Creator": "WMT26 Task 2 threshold analysis",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def write_delta_plot(
    path: Path,
    rows: list[dict[str, Any]],
    title: str,
    series: tuple[
        tuple[str, str, str],
        tuple[str, str, str],
    ],
    symmetric: bool,
) -> None:
    """Plot declared versus optimized MCC changes for the supplied system rows."""
    ordered = sorted(rows, key=lambda row: (row["track"], -row[series[1][0]]))
    values = [float(row[field]) for row in rows for field, _, _ in series]
    if symmetric:
        bound = max(0.02, max(abs(value) for value in values) * 1.1)
        lower, upper = -bound, bound
        ticks = (-bound, -bound / 2, 0.0, bound / 2, bound)
    else:
        lower = min(-0.01, min(values) * 1.1)
        upper = max(0.02, max(values) * 1.1)
        ticks = (0.0, upper / 4, upper / 2, 3 * upper / 4, upper)

    figure, axis = plt.subplots(figsize=(15, 1.75 + 0.38 * len(ordered)))
    y_positions = list(range(len(ordered)))
    bar_height = 0.26
    for offset, (field, label, color) in zip((-0.16, 0.16), series):
        axis.barh(
            [position + offset for position in y_positions],
            [float(row[field]) for row in ordered],
            height=bar_height,
            color=color,
            label=label,
        )

    labels = []
    for row in ordered:
        label = f"{row['track'][0].upper()} {row['entry_id']}  {row['submission_name']}"
        labels.append(label if len(label) <= 54 else label[:51] + "...")
    axis.set_yticks(y_positions, labels)
    axis.tick_params(axis="y", length=0, labelsize=8)
    axis.set_xlim(lower, upper)
    axis.set_xticks(ticks, [f"{tick:+.3f}" for tick in ticks])
    axis.axvline(0, color=GRAY, linewidth=0.8)
    axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.invert_yaxis()
    axis.set_title(title, loc="left", fontsize=14, pad=42)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        ncol=2,
        borderaxespad=0,
    )
    for index in range(1, len(ordered)):
        if ordered[index - 1]["track"] != ordered[index]["track"]:
            axis.axhline(index - 0.5, color="#888888", linewidth=0.7)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    figure.subplots_adjust(left=0.34, right=0.98, top=0.86, bottom=0.09)
    save_pdf(figure, path)


def write_summary_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write the original pooled and macro sensitivity summary figures."""
    write_delta_plot(
        output_dir / "threshold_improvement.pdf",
        rows,
        "Task 2 threshold retuning: pooled MCC change",
        (
            ("apparent_pooled_mcc_delta", "Same-data optimum", BLUE),
            ("cv_pooled_mcc_delta_vs_declared", "5-fold item-grouped CV", ORANGE),
        ),
        symmetric=True,
    )
    write_delta_plot(
        output_dir / "macro_threshold_improvement.pdf",
        rows,
        "Macro MCC change from threshold retuning",
        (
            (
                "pooled_optimum_macro_language_mcc_delta",
                "One pooled-optimal threshold",
                BLUE,
            ),
            (
                "language_optimal_macro_mcc_delta_vs_declared",
                "Per-language thresholds",
                GREEN,
            ),
        ),
        symmetric=False,
    )


def write_paper_gain_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Compare macro MCC changes from pooled and per-language thresholds."""
    pooled_field = "pooled_optimum_macro_language_mcc_delta"
    language_field = "language_optimal_macro_mcc_delta_vs_declared"
    values = [
        float(row[field])
        for row in rows
        for field in (pooled_field, language_field)
    ]
    lower = min(-0.01, min(values) - 0.005)
    upper = max(values) + 0.012

    figure, axis = plt.subplots(figsize=(3.35, 3.05))
    axis.plot(
        (lower, upper),
        (lower, upper),
        color=GRAY,
        linestyle=":",
        linewidth=1,
        label="Equal change",
        zorder=1,
    )
    for track, label, marker, color in (
        ("participant", "Participants", "o", BLUE),
        ("baseline", "Organizer baselines", "s", ORANGE),
    ):
        selected = [row for row in rows if row["track"] == track]
        axis.scatter(
            [float(row[pooled_field]) for row in selected],
            [float(row[language_field]) for row in selected],
            s=24,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            label=label,
            zorder=2,
        )

    tick_upper = math.ceil(upper / 0.05) * 0.05
    ticks = [index * 0.05 for index in range(int(round(tick_upper / 0.05)) + 1)]
    axis.set_xlim(lower, tick_upper)
    axis.set_ylim(lower, tick_upper)
    axis.set_xticks(ticks, [f"{value:+.2f}" for value in ticks])
    axis.set_yticks(ticks, [f"{value:+.2f}" for value in ticks])
    axis.set_xlabel("Pooled-optimal threshold\n$\\Delta$ macro-averaged MCC", fontsize=7.5)
    axis.set_ylabel(
        "Per-language thresholds\n$\\Delta$ macro-averaged MCC", fontsize=7.5
    )
    axis.axhline(0, color="#aaaaaa", linewidth=0.6, zorder=0)
    axis.axvline(0, color="#aaaaaa", linewidth=0.6, zorder=0)
    axis.grid(color="#e5e5e5", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(length=2, labelsize=6.5)
    axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=6.5,
        borderaxespad=0.3,
        handletextpad=0.45,
        labelspacing=0.35,
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.2)
    save_pdf(figure, path)


def threshold_x(
    threshold: float,
    score_divisor: float,
    score_min: float,
    score_max: float,
) -> float:
    """Transform raw score thresholds to the original display scale."""
    return min(score_max, max(score_min, threshold / score_divisor))


def plottable_curve_points(points: Iterable[Any]) -> tuple[Any, ...]:
    """Omit only extreme thresholds that produce no positive predictions."""
    return tuple(point for point in points if point.true_positive > 0)


def plot_legend(include_task1: bool) -> list[Line2D]:
    """Return the legend artists used in the original curve figures."""
    handles = [
        Line2D([], [], color=BLUE, label="MCC"),
        Line2D([], [], color=GREEN, label="Precision"),
        Line2D([], [], color=ORANGE, label="Recall"),
        Line2D([], [], color=RED, linestyle="--", label="Submitted threshold"),
        Line2D([], [], color=BLUE, linestyle="--", label="Selected MCC optimum"),
    ]
    if include_task1:
        handles.append(
            Line2D([], [], color=GRAY, linestyle="--", label="Task 1 MCC")
        )
    return handles


def paper_plot_legend(include_task1: bool) -> list[Line2D]:
    """Return the compact paper-grid legend, optionally including Task 1 MCC."""
    handles = [
        Line2D([], [], color=BLUE, label="MCC"),
        Line2D([], [], color=GREEN, label="Precision"),
        Line2D([], [], color=ORANGE, label="Recall"),
        Line2D([], [], color=RED, linestyle="--", label="Submitted threshold"),
        Line2D([], [], color=BLUE, linestyle="--", label="Per-language optimum"),
    ]
    if include_task1:
        handles.append(
            Line2D([], [], color=GRAY, linestyle=":", label="Task 1->3 MCC")
        )
    return handles


def select_results_by_id(
    results: Iterable[AnalysisResult], track: str, entry_ids: list[str]
) -> list[AnalysisResult]:
    """Select requested systems in order, rejecting missing or duplicated identities."""
    by_id = {
        str(result.row["entry_id"]): result
        for result in results
        if result.row["track"] == track
    }
    unknown = [entry_id for entry_id in entry_ids if entry_id not in by_id]
    if unknown:
        raise DataError(
            f"Unknown {track} paper-figure entry ID(s): {', '.join(unknown)}"
        )
    return [by_id[entry_id] for entry_id in entry_ids]


def short_pair_label(language_pair: str) -> str:
    """Return the compact paper-facing label for a language pair."""
    source, target = language_pair.split("---", maxsplit=1)
    return f"{source.split('_', maxsplit=1)[0]}-{target.split('_', maxsplit=1)[0]}"


def write_paper_grid_plot(
    path: Path,
    results: list[AnalysisResult],
    language_pairs: list[str],
    regime_labels: list[str],
    task1_mcc: dict[str, dict[str, float]] | None = None,
    show_legend: bool = True,
) -> None:
    """Write a compact language-by-system threshold comparison grid."""
    if not results or not language_pairs or len(results) != len(regime_labels):
        raise DataError("Paper grid requires systems, pairs, and one regime per system")
    missing = [
        (str(result.row["entry_id"]), pair)
        for result in results
        for pair in language_pairs
        if result.language(pair) is None
    ]
    if missing:
        details = ", ".join(f"{entry_id}:{pair}" for entry_id, pair in missing)
        raise DataError(f"Paper-figure language coverage is missing: {details}")

    source_maxima = [
        float(language.row["source_score_max"])
        for result in results
        for pair in language_pairs
        if (language := result.language(pair)) is not None
    ]
    hundred_point_scale = all(value > 1 for value in source_maxima)
    if not hundred_point_scale and any(value > 1 for value in source_maxima):
        raise DataError("Paper grid cannot mix 0-1 and 0-100 score scales")
    score_max = 100.0 if hundred_point_scale else 1.0
    references = (
        [
            task1_mcc[str(result.row["entry_id"])][pair]
            for result in results
            for pair in language_pairs
            if pair in task1_mcc.get(str(result.row["entry_id"]), {})
        ]
        if task1_mcc is not None
        else []
    )
    minimum_reference = min(references, default=0.0)
    y_min = (
        math.floor((minimum_reference - 0.01) / 0.05) * 0.05
        if minimum_reference < 0
        else 0.0
    )

    row_count = len(language_pairs)
    column_count = len(results)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.2, 1.2 * row_count + 0.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    if show_legend:
        figure.legend(
            handles=paper_plot_legend(task1_mcc is not None),
            loc="upper center",
            bbox_to_anchor=(0.53, 0.995),
            frameon=False,
            ncol=3,
            fontsize=7,
            handlelength=2.1,
            columnspacing=1.7,
            labelspacing=0.6,
        )

    for column, result in enumerate(results):
        label = str(result.row["submission_name"]).replace(" - Baseline", "")
        axes[0][column].set_title(label, fontsize=8, pad=5)

    for row_index, language_pair in enumerate(language_pairs):
        for column_index, result in enumerate(results):
            axis = axes[row_index][column_index]
            language = result.language(language_pair)
            assert language is not None
            language_row = language.row
            curve_points = plottable_curve_points(language.plot_points)
            x_values = [
                threshold_x(point.threshold, 1.0, 0.0, score_max)
                for point in curve_points
            ]
            for values, color in (
                ([point.mcc for point in curve_points], BLUE),
                ([point.precision for point in curve_points], GREEN),
                ([point.recall for point in curve_points], ORANGE),
            ):
                axis.plot(x_values, values, color=color, linewidth=1.15)

            if task1_mcc is not None:
                reference = task1_mcc.get(str(result.row["entry_id"]), {}).get(
                    language_pair
                )
                if reference is not None:
                    axis.axhline(
                        reference,
                        color=GRAY,
                        linestyle=":",
                        linewidth=1.0,
                        zorder=1,
                    )

            declared_threshold = float(result.row["declared_threshold"])
            selected_threshold = float(language_row["language_optimal_threshold"])
            markers = (
                (declared_threshold, float(language_row["declared_mcc"]), RED),
                (selected_threshold, float(language_row["language_optimal_mcc"]), BLUE),
            )
            for threshold, mcc, color in markers:
                x_value = threshold_x(threshold, 1.0, 0.0, score_max)
                axis.axvline(
                    x_value,
                    color=color,
                    linestyle="--",
                    linewidth=0.9,
                    zorder=1,
                )
                axis.scatter(
                    [x_value],
                    [mcc],
                    color=color,
                    edgecolors="white",
                    linewidths=0.35,
                    s=17,
                    zorder=3,
                )

            delta = float(language_row["language_optimal_mcc_delta_vs_declared"])
            axis.text(
                0.025,
                0.1 if y_min < 0 else 0.035,
                f"t: {declared_threshold:.1f} -> "
                f"{selected_threshold:.1f}; "
                f"dMCC {delta:+.3f}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.2,
                },
            )
            axis.set_xlim(0, score_max)
            axis.set_ylim(y_min, 1)
            axis.set_xticks((0, score_max / 2, score_max))
            axis.set_yticks((0, 0.5, 1))
            axis.grid(axis="y", color="#e6e6e6", linewidth=0.55)
            axis.tick_params(length=2, labelsize=7)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)

        axes[row_index][0].set_ylabel(
            f"{short_pair_label(language_pair)}\nmetric", fontsize=7.5, labelpad=7
        )

    for axis in axes[-1]:
        axis.set_xlabel(
            "Task 2 score (0-100)" if hundred_point_scale else "Task 2 score (0-1)",
            fontsize=7.5,
        )

    figure.subplots_adjust(
        left=0.105,
        right=0.99,
        top=0.78 if show_legend else 0.88,
        bottom=0.095,
        hspace=0.25,
        wspace=0.17,
    )
    save_pdf(figure, path)


def write_curve_plot(
    path: Path,
    title: str,
    language_pair: str | None,
    results: list[AnalysisResult],
    task1_mcc: dict[str, dict[str, float]] | None = None,
    show_header: bool = True,
) -> None:
    """Plot MCC/precision/recall with submitted/optimal threshold markers."""
    columns = 3
    row_count = math.ceil(len(results) / columns)
    height = 0.9 + 2.85 * row_count if show_header else 0.35 + 2.85 * row_count
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(13.2, height),
        squeeze=False,
    )

    if show_header:
        figure.suptitle(title, x=0.065, y=0.985, ha="left", fontsize=14)
        figure.legend(
            handles=plot_legend(task1_mcc is not None),
            loc="upper left",
            bbox_to_anchor=(0.06, 0.958),
            frameon=False,
            ncol=6,
            fontsize=8,
        )

    for axis, result in zip(axes.flat, results):
        row = result.row
        if language_pair is None:
            curve_points = result.plot_points
            source_score_min = float(row["source_score_min"])
            source_score_max = float(row["source_score_max"])
            declared_mcc = float(row["declared_pooled_mcc"])
            selected_threshold = float(row["optimal_threshold"])
            selected_mcc = float(row["optimal_pooled_mcc"])
        else:
            language = result.language(language_pair)
            if language is None:
                axis.set_visible(False)
                continue
            curve_points = language.plot_points
            language_row = language.row
            source_score_min = float(language_row["source_score_min"])
            source_score_max = float(language_row["source_score_max"])
            declared_mcc = float(language_row["declared_mcc"])
            selected_threshold = float(language_row["language_optimal_threshold"])
            selected_mcc = float(language_row["language_optimal_mcc"])

        curve_points = plottable_curve_points(curve_points)

        facet_scale = str(row["entry_id"]) in {"869543", "869546"}
        score_divisor = (
            1.0
            if facet_scale or source_score_max <= 1
            else 100.0
        )
        score_min = source_score_min if facet_scale else 0.0
        score_max = source_score_max if facet_scale else 1.0

        x_values = [
            threshold_x(point.threshold, score_divisor, score_min, score_max)
            for point in curve_points
        ]
        axis.plot(x_values, [point.mcc for point in curve_points], color=BLUE)
        axis.plot(
            x_values,
            [point.precision for point in curve_points],
            color=GREEN,
        )
        axis.plot(
            x_values,
            [point.recall for point in curve_points],
            color=ORANGE,
        )

        for threshold, mcc, color in (
            (float(row["declared_threshold"]), declared_mcc, RED),
            (selected_threshold, selected_mcc, BLUE),
        ):
            x_value = threshold_x(threshold, score_divisor, score_min, score_max)
            axis.axvline(x_value, color=color, linestyle="--", linewidth=1)
            axis.scatter([x_value], [mcc], color=color, s=20, zorder=3)

        if task1_mcc is not None and language_pair is not None:
            reference = task1_mcc.get(str(row["entry_id"]), {}).get(language_pair)
            if reference is not None:
                axis.axhline(reference, color=GRAY, linestyle="--", linewidth=0.9)

        label = f"{row['entry_id']}  {row['submission_name']}"
        axis.set_title(
            label if len(label) <= 48 else label[:45] + "...",
            loc="left",
            fontsize=9,
        )
        axis.set_xlim(score_min, score_max)
        axis.set_ylim(0, 1)
        axis.set_xticks(
            [score_min, score_max],
            [fmt_threshold(score_min), fmt_threshold(score_max)],
        )
        axis.set_yticks([0, 1], [fmt_mcc(0), fmt_mcc(1)])
        axis.set_xlabel(
            "raw submitted Task 2 score"
            if facet_scale
            else "normalized Task 2 score",
            fontsize=7,
        )
        axis.tick_params(length=0, labelsize=7)

    for axis in list(axes.flat)[len(results) :]:
        axis.set_visible(False)

    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.89 if show_header else 0.94,
        bottom=0.075,
        hspace=0.48,
        wspace=0.28,
    )
    save_pdf(figure, path)


def write_curve_plot_set(
    path: Path,
    title: str,
    language_pair: str | None,
    results: list[AnalysisResult],
    task1_mcc: dict[str, dict[str, float]] | None = None,
) -> list[Path]:
    """Write a full curve grid and its original top/bottom panel splits."""
    write_curve_plot(path, title, language_pair, results, task1_mcc)
    written = [path]
    if len(results) > 9:
        top = path.with_name(f"{path.stem}__top.pdf")
        bottom = path.with_name(f"{path.stem}__bottom.pdf")
        write_curve_plot(top, title, language_pair, results[:9], task1_mcc)
        write_curve_plot(
            bottom,
            title,
            language_pair,
            results[9:],
            task1_mcc,
            show_header=False,
        )
        written.extend((top, bottom))
    return written


def write_pooled_plots(
    output_dir: Path,
    results: list[AnalysisResult],
) -> list[Path]:
    """Write pooled threshold curves for participant and baseline tracks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for track, label in (
        ("participant", "Participant"),
        ("baseline", "Organizer baseline"),
    ):
        selected = [result for result in results if result.row["track"] == track]
        if not selected:
            continue
        path = output_dir / f"{track}_pooled_curves.pdf"
        written.extend(
            write_curve_plot_set(
                path,
                f"{label} pooled Task 2 threshold curves",
                None,
                selected,
            )
        )
    return written


def write_language_plots(
    output_dir: Path,
    results: list[AnalysisResult],
    language_pairs: list[str],
    task1_mcc: dict[str, dict[str, float]],
) -> list[Path]:
    """Write per-language curves and matching Task 1 baseline reference lines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for language_pair in language_pairs:
        slug = plot_slug(language_pair)
        for track, label in (
            ("participant", "Participant"),
            ("baseline", "Organizer baseline"),
        ):
            selected = [
                result
                for result in results
                if result.row["track"] == track
                and result.language(language_pair) is not None
            ]
            if not selected:
                continue
            path = output_dir / f"{track}_curves__{slug}.pdf"
            title = f"{label} {language_pair} Task 2 threshold curves"
            reference = task1_mcc if track == "baseline" else None
            written.extend(
                write_curve_plot_set(
                    path,
                    title,
                    language_pair,
                    selected,
                    reference,
                )
            )
    return written
