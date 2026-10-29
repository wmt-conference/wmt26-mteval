"""Paper figure for common source support under the two SPA variants."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .paper_labels import TEX_DIRECTION_LABELS
PLOT_LABELS = {key: value.replace("$\\rightarrow$", "→").replace("$", "").replace("{\\\'a}", "á") for key, value in TEX_DIRECTION_LABELS.items()}

def plot_support(path: Path, rows: list[dict]) -> None:
    labels = [PLOT_LABELS[row["direction"]] for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), sharey=True)
    fig.subplots_adjust(left=.245, right=.985, bottom=.18, top=.77, wspace=.045)
    panels = (("Median", "screen_spa_items_median", "segment_spa_items_median"),
              ("Mean", "screen_spa_items_mean", "segment_spa_items_mean"))
    for axis, (title, screen_key, source_key) in zip(axes, panels, strict=True):
        screen = [row[screen_key] for row in rows]
        source = [row[source_key] for row in rows]
        maximum_support = max(*screen, *source)
        label_offset = maximum_support * .012
        axis.barh(y - .18, screen, .34, label="Screen-based SPA", color="#8b9aab")
        axis.barh(y + .18, source, .34, label="Source-based SPA", color="#2a6fbb")
        for position, value in zip(y - .18, screen, strict=True):
            axis.text(value + label_offset, position, f"{value:.0f}", va="center", fontsize=8)
        for position, value in zip(y + .18, source, strict=True):
            axis.text(value + label_offset, position, f"{value:.0f}", va="center", fontsize=8)
        axis.set_xlim(0, maximum_support * 1.105)
        axis.set_title(f"{title} pair support", weight="bold", pad=9)
        axis.set_xlabel("Common source items per system pair")
        axis.grid(axis="x", color="#d9e0e7", linewidth=.7)
        axis.set_axisbelow(True)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.64, .905),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Source-based SPA retains much broader support", fontsize=16,
                 weight="bold", y=.975)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
