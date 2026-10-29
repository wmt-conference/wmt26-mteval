"""Shared single-column publication layout for Task 1 reference effects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ReferenceEffectRow:
    """One direction, reference description, cESA label, and ΔF1 in points."""

    direction: str
    reference: str
    cesa: str
    delta: float


def short_direction(direction: str) -> str:
    """Return the project's three-letter direction label from a script-tagged pair.

    Chinese targets retain the project's CN/TW distinction; Chinese sources
    use zho. Hyphenated labels keep the three-column paper layout compact.
    """

    source, target = direction.split("-")
    chinese_targets = {
        "zho_Hans": "zho_CN", "zho_CN": "zho_CN",
        "zho_Hant": "zho_TW", "zho_Hant_TW": "zho_TW", "zho_TW": "zho_TW",
    }
    return f"{source.split('_')[0]}-{chinese_targets.get(target, target.split('_')[0])}"


def plot_reference_effects(
    path: Path, rows: Sequence[ReferenceEffectRow], plt: Any, *, definition: str,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> tuple[Path, ...]:
    """Save the requested 3.5-inch dot-plot formats from point-scale effect rows.

    All geometry and typography live here so metadata length cannot move the
    grid boundaries. Axis ranges adapt to the data; effect units stay the same.
    The connecting lines indicate distance from zero, not uncertainty.
    """

    if not rows:
        return ()
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    height = 0.35 * len(rows) + 0.72
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": True, "savefig.bbox": None,
    }):
        figure = plt.figure(figsize=(3.5, height))
        bottom, top = 0.43 / height, 1 - 0.29 / height
        directions = figure.add_axes((0.025, bottom, 0.18, top - bottom))
        metadata = figure.add_axes((0.22, bottom, 0.38, top - bottom))
        axis = figure.add_axes((0.62, bottom, 0.35, top - bottom))
        for panel in (directions, metadata, axis):
            panel.set_ylim(len(rows) - 0.5, -0.5)
            panel.set_yticks([])
            for spine in panel.spines.values():
                spine.set_visible(False)
        for panel in (directions, metadata):
            panel.set_xlim(0, 1)
            panel.set_xticks([])
        for index, row in enumerate(rows):
            directions.text(0, index, short_direction(row.direction),
                          fontsize=9, fontweight="bold", va="center")
            _draw_reference_badges(metadata, index - 0.18, row.reference)
            metadata.text(0.5, index + 0.22, row.cesa, fontsize=7.5,
                          color="#555555", va="center", ha="center")
            color = "#2878A0" if row.delta >= 0 else "#B64B3E"
            axis.hlines(index, 0, row.delta, color=color, linewidth=2, zorder=3)
            axis.plot(row.delta, index, "o", color=color, markersize=4, zorder=4)
            axis.annotate(f"{row.delta:+.2f}", (row.delta, index),
                          xytext=(0, 7), textcoords="offset points",
                          ha="center", va="bottom", fontsize=8.5,
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
        bound = max(0.1, max(abs(row.delta) for row in rows)) * 1.45
        axis.set_xlim(-bound, bound)
        ticks = MaxNLocator(nbins=4, symmetric=True).tick_values(-bound, bound)
        axis.set_xticks([tick for tick in ticks if -bound < tick < bound])
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axis.tick_params(axis="x", labelsize=8, length=2.5, color="#888888")
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#dddddd", linewidth=0.5)
        axis.axvline(0, color="#777777", linewidth=0.8, zorder=2)
        axis.set_xlabel("Mean ΔF1", fontsize=9, labelpad=4)
        figure.text(0.025, 1 - 0.12 / height, definition, fontsize=8,
                    va="center", color="#333333")
        paths = tuple(path.with_suffix(f".{suffix}") for suffix in formats)
        for output in paths:
            figure.savefig(output, dpi=300)
        plt.close(figure)
    return paths


def _draw_reference_badges(axis: Any, position: float, reference: str) -> None:
    """Center a reference label or A → B pair with consistent type colours.

    Inputs are the metadata axis, row position, and display label. Adds compact
    rounded badges to the axis; an arrow between reference types stays neutral.
    """

    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    colors = {"Native": "#e8eef3", "Post-edited": "#f2ead8", "Pseudo": "#ece7f2"}
    labels = []
    for index, name in enumerate(reference.split(" → ")):
        if index:
            labels.append(TextArea("→", textprops={"fontsize": 8, "color": "#555555"}))
        labels.append(TextArea(name, textprops={
            "fontsize": 8, "color": "#222222",
            "bbox": {"boxstyle": "round,pad=0.13", "facecolor": colors.get(name, "#eeeeee"),
                     "edgecolor": "none"},
        }))
    packed = HPacker(children=labels, align="center", pad=0, sep=3)
    axis.add_artist(AnnotationBbox(packed, (0.5, position), frameon=False,
                                  box_alignment=(0.5, 0.5), pad=0))
