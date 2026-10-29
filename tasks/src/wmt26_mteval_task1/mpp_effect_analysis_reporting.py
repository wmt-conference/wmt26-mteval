"""Render the paper reference-use and thinking effect artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re
from typing import Any

from wmt26_mteval_task1.constants import PRIMARY_REFERENCE_TYPE_BY_DIRECTION
from wmt26_mteval_task1.markdown_reporting import markdown_table, write_markdown_companion
from wmt26_mteval_task1.mpp_effect_analysis import MppEffectAnalysisResult, REFERENCE_EFFECT
from wmt26_mteval_task1.reference_effect_plotting import ReferenceEffectRow, plot_reference_effects


_CESA_SCORES_BY_DIRECTION = {
    "ces_Latn-deu_Latn": 80.7,
    "ces_Latn-vie_Latn": 87.4,
    "eng_Latn-arz_Arab": 83.4,
    "eng_Latn-ekk_Latn": 81.0,
    "eng_Latn-ind_Latn": 75.1,
    "eng_Latn-isl_Latn": 87.5,
    "eng_Latn-sme_Latn": 81.2,
    "zho_Hans-jpn_Jpan": 79.6,
}

def _combined_effect_latex() -> str:
    """Return the approved combined reference/thinking F1 publication table."""

    return r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lrrr}

\toprule

\textbf{Auto-evaluator} & \multicolumn{1}{c}{Ref-free} & \multicolumn{1}{c}{Ref-based} & \multicolumn{1}{c}{$\Delta$F1} \\
\midrule
Gemma 4\textsuperscript{\#} & $31.72$ & $32.58$ & $+0.86$ \\
Qwen 3.6\textsuperscript{\#} & $29.77$ & $29.38$ & $-0.39$ \\
Gemma 4 & $32.02$ & $32.08$ & $+0.06$ \\
Qwen 3.6 & $24.62$ & $23.46$ & $-1.16$ \\
xCOMET XL & $13.77$ & $16.74$ & $+2.97$ \\
xCOMET XXL & $13.93$ & $17.85$ & $+3.92$ \\

\midrule

& \multicolumn{1}{c}{Non-thinking} & \multicolumn{1}{c}{Thinking} & \multicolumn{1}{c}{$\Delta$F1} \\
\midrule
Gemma 4 & $32.08$ & $32.58$ & $+0.49$ \\
Qwen 3.6 & $23.46$ & $29.38$ & $+5.91$ \\
Gemma 4\textsuperscript{*} & $32.02$ & $31.72$ & $-0.31$ \\
Qwen 3.6\textsuperscript{*} & $24.62$ & $29.77$ & $+5.15$ \\

\bottomrule
\end{tabular}
\caption{Effect of reference use and thinking in task 1. \textsuperscript{\#} marks thinking auto-evaluators. \textsuperscript{*} marks reference-free auto-evaluators.}
\label{tab:task1-reference-thinking-effect}
\end{table}
"""


def _combined_behavior_latex() -> str:
    """Return the approved combined reference/thinking behavior table."""

    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{
\begin{tabular}{lrrrrrrrrr}

\textbf{Auto-evaluator} & \multicolumn{3}{c}{\textbf{Errors per prediction}} & \multicolumn{3}{c}{\textbf{Mean span length}} & \multicolumn{3}{c}{\textbf{Mean severity weight}} \\

\midrule
& Ref-free & Ref-based & $\Delta$ & Ref-free & Ref-based & $\Delta$ & Ref-free & Ref-based & $\Delta$ \\
\midrule
Gemma 4\textsuperscript{\#} & $2.40$ & $3.13$ & $+0.73$ & $21.20$ & $18.03$ & $-3.16$ & $0.57$ & $0.52$ & $-0.05$ \\
Qwen 3.6\textsuperscript{\#} & $3.11$ & $4.10$ & $+0.99$ & $21.43$ & $20.14$ & $-1.29$ & $0.55$ & $0.50$ & $-0.04$ \\
Gemma 4 & $2.61$ & $3.65$ & $+1.04$ & $21.69$ & $20.32$ & $-1.36$ & $0.61$ & $0.53$ & $-0.07$ \\
Qwen 3.6 & $3.93$ & $5.23$ & $+1.29$ & $27.43$ & $26.49$ & $-0.94$ & $0.67$ & $0.64$ & $-0.03$ \\

\midrule

 & Non-thinking & Thinking & $\Delta$ & Non-thinking & Thinking & $\Delta$ & Non-thinking & Thinking & $\Delta$ \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}

Gemma 4 & $3.71$ & $3.18$ & $-0.53$ & $19.69$ & $17.50$ & $-2.19$ & $0.53$ & $0.51$ & $-0.02$ \\
Qwen 3.6 & $5.36$ & $4.15$ & $-1.21$ & $25.84$ & $19.88$ & $-5.95$ & $0.63$ & $0.50$ & $-0.13$ \\
Gemma 4\textsuperscript{*} & $2.67$ & $2.44$ & $-0.24$ & $21.25$ & $20.27$ & $-0.98$ & $0.60$ & $0.57$ & $-0.04$ \\
Qwen 3.6\textsuperscript{*} & $3.97$ & $3.15$ & $-0.82$ & $27.34$ & $20.95$ & $-6.39$ & $0.67$ & $0.54$ & $-0.13$ \\

\bottomrule
\end{tabular}
}
\caption{Effect of reference use and thinking on predicted errors in task 1. \textsuperscript{\#} marks thinking auto-evaluators. \textsuperscript{*} marks reference-free auto-evaluators. Errors per prediction include omissions; mean span length is computed over non-omission spans; and mean severity weight includes omissions.}
\label{tab:task1-reference-thinking-effect-spans}
\end{table*}
"""


def _static_table_rows(latex: str) -> list[tuple[str, ...]]:
    """Extract the approved numeric TeX rows for a matching Markdown table."""

    rows = []
    for line in latex.splitlines():
        line = line.strip()
        if not line.startswith(("Gemma 4", "Qwen 3.6", "xCOMET")) or not line.endswith(r"\\"):
            continue
        cells = []
        for raw in line[:-2].split("&"):
            value = raw.strip().replace(r"\textsuperscript{\#}", " [thinking]")
            value = value.replace(r"\textsuperscript{*}", " [reference-free]")
            value = re.sub(r"^\$(.*)\$$", r"\1", value)
            cells.append(value)
        rows.append(tuple(cells))
    return rows


def _effect_markdown(latex: str) -> str:
    """Render the reference-use and thinking F1 sections from approved TeX rows."""

    rows = _static_table_rows(latex)
    if len(rows) != 10 or any(len(row) != 4 for row in rows):
        raise ValueError("reference/thinking TeX rows do not match the paper table")
    return (
        "## Reference use\n\n"
        + markdown_table(
            ("Auto-evaluator", "Ref-free F1", "Ref-based F1", "ΔF1"),
            rows[:6],
        )
        + "\n\n## Thinking\n\n"
        + markdown_table(
            ("Auto-evaluator", "Non-thinking F1", "Thinking F1", "ΔF1"),
            rows[6:],
        )
        + "\n\n[thinking] marks thinking evaluators; [reference-free] marks reference-free evaluators."
    )


def _behavior_markdown(latex: str) -> str:
    """Render the three prediction-behavior measures from approved TeX rows."""

    rows = _static_table_rows(latex)
    if len(rows) != 8 or any(len(row) != 10 for row in rows):
        raise ValueError("prediction-behavior TeX rows do not match the paper table")
    measures = ("Errors per prediction", "Mean span length", "Mean severity weight")

    def headers(conditions: tuple[str, str]) -> tuple[str, ...]:
        """Expand the paper's grouped measure headers into flat Markdown columns."""

        return ("Auto-evaluator",) + tuple(
            f"{measure}: {condition}"
            for measure in measures
            for condition in (*conditions, "Δ")
        )

    return (
        "## Reference use\n\n"
        + markdown_table(headers(("Ref-free", "Ref-based")), rows[:4])
        + "\n\n## Thinking\n\n"
        + markdown_table(headers(("Non-thinking", "Thinking")), rows[4:])
        + "\n\nErrors per prediction include omissions. Mean span length excludes omissions; "
        "mean severity weight includes them. [thinking] and [reference-free] "
        "identify the marked evaluator conditions."
    )


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


def _mean_direction_f1_effects(
    result: MppEffectAnalysisResult,
    effect: str,
    *,
    exclude_xcomet: bool = False,
) -> list[tuple[str, float]]:
    """Average one effect's F1 deltas across eligible matched model pairs."""

    output = []
    for direction in result.directions:
        values = [
            row.delta_values[2]
            for row in result.direction_effects
            if row.effect == effect
            and row.language_pair == direction
            and (not exclude_xcomet or not row.pair_id.startswith("xcomet_"))
        ]
        if values:
            output.append((direction, sum(values) / len(values)))
    return output


def plot_column_reference_effect(
    output: Path, rows: Iterable[tuple[str, float]], plt: Any,
    *, formats: tuple[str, ...] = ("pdf", "png"),
) -> tuple[Path, ...]:
    """Render non-xCOMET direction means (unit-scale F1) as column figures."""

    return plot_reference_effects(output, [
        ReferenceEffectRow(direction, PRIMARY_REFERENCE_TYPE_BY_DIRECTION[direction],
                           f"cESA {_CESA_SCORES_BY_DIRECTION[direction]:.1f}", 100 * value)
        for direction, value in rows
    ], plt, definition="ΔF1 = with reference − without reference", formats=formats)


def write_paper_mpp_effect_reports(
    output_dir: Path, result: MppEffectAnalysisResult
) -> tuple[Path, Path, Path]:
    """Write the two approved effect tables and the eight-direction PDF figure.

    The result supplies matched baseline effects; the returned paths are the
    paper's F1 table, behavior table, and reference-use PDF respectively.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    effect_path = output_dir / "task1-reference-thinking-effect.tex"
    behavior_path = output_dir / "task1-reference-thinking-effect-spans.tex"
    effect_latex = _combined_effect_latex()
    behavior_latex = _combined_behavior_latex()
    effect_path.write_text(effect_latex, encoding="utf-8")
    behavior_path.write_text(behavior_latex, encoding="utf-8")
    write_markdown_companion(
        effect_path, "Task 1: reference use and thinking effects",
        _effect_markdown(effect_latex),
    )
    write_markdown_companion(
        behavior_path, "Task 1: prediction behavior by condition",
        _behavior_markdown(behavior_latex),
    )
    direction_effects = _mean_direction_f1_effects(
        result, REFERENCE_EFFECT, exclude_xcomet=True
    )
    figure_paths = plot_column_reference_effect(
        output_dir / "task1-llm-reference-effect",
        direction_effects,
        _matplotlib(),
        formats=("pdf",),
    )
    if len(figure_paths) != 1:
        raise ValueError("reference-use paper figure has no supported directions")
    write_markdown_companion(
        figure_paths[0], "Task 1: reference effect by priority direction",
        markdown_table(
            ("Translation direction", "Reference type", "cESA", "Mean ΔF1"),
            (
                (direction, PRIMARY_REFERENCE_TYPE_BY_DIRECTION[direction],
                 f"{_CESA_SCORES_BY_DIRECTION[direction]:.1f}", f"{100 * value:+.2f}")
                for direction, value in direction_effects
            ),
        ) + "\n\nΔF1 = with reference − without reference; Gemma 4 and Qwen 3.6 only.",
    )
    return effect_path, behavior_path, figure_paths[0]
