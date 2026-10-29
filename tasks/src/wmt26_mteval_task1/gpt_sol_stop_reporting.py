"""Render the paper GPT Sol shared-stop comparison table."""

from pathlib import Path

from typing import Any

from wmt26_mteval_task1.gpt_sol_stop import (
    MEDIUM,
    XHIGH,
    GptSolStopResult,
)

from wmt26_mteval_task1.markdown_reporting import markdown_table, write_markdown_companion
from wmt26_mteval_task1.reporting import _latex_escape, _latex_metric

_LATEX_EVALUATOR_NAMES = {
    MEDIUM: "GPT-5.6 Sol - medium thinking",
    XHIGH: "GPT-5.6 Sol - xhigh thinking",
}

_BEHAVIOR_FIELDS = (
    "mean_errors_per_assessment",
    "mean_text_span_length_characters",
    "mean_severity_weight",
)

def _latex_behavior_value(value: float | None, *, signed: bool = False) -> str:
    """Format one natural-scale prediction statistic for a LaTeX table."""

    if value is None:
        return "--"
    specification = "+.2f" if signed else ".2f"
    return f"${format(float(value), specification)}$"

def _gpt_sol_stop_latex(result: GptSolStopResult) -> str:
    """Render pooled MPP scores and prediction behavior as a publication table."""

    scores = {row["evaluator"]: row for row in result.scores}
    statistics = {row["evaluator"]: row for row in result.prediction_statistics}
    rows: list[tuple[str, dict[str, Any], dict[str, float | None], bool]] = []
    for evaluator in (MEDIUM, XHIGH):
        rows.append(
            (
                _latex_escape(_LATEX_EVALUATOR_NAMES[evaluator]),
                scores[evaluator],
                statistics[evaluator],
                False,
            )
        )
    rows.append(
        (
            r"$\Delta$ (xhigh $-$ medium)",
            scores["xhigh_minus_medium"],
            {
                field: (
                    float(statistics[XHIGH][field])
                    - float(statistics[MEDIUM][field])
                    if statistics[XHIGH][field] is not None
                    and statistics[MEDIUM][field] is not None
                    else None
                )
                for field in _BEHAVIOR_FIELDS
            },
            True,
        )
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Auto-evaluator & \multicolumn{1}{c}{P (pp)} & \multicolumn{1}{c}{R (pp)} "
        r"& \multicolumn{1}{c}{F1 (pp)} & \multicolumn{1}{c}{Errors per prediction} "
        r"& \multicolumn{1}{c}{Mean span length} "
        r"& \multicolumn{1}{c}{Mean severity weight} \\",
        r"\midrule",
    ]
    for name, score, behavior, signed in rows:
        lines.append(
            f"{name} & {_latex_metric(float(score['precision']), signed=signed)} & "
            f"{_latex_metric(float(score['recall']), signed=signed)} & "
            f"{_latex_metric(float(score['f1']), signed=signed)} & "
            f"{_latex_behavior_value(behavior[_BEHAVIOR_FIELDS[0]], signed=signed)} & "
            f"{_latex_behavior_value(behavior[_BEHAVIOR_FIELDS[1]], signed=signed)} & "
            f"{_latex_behavior_value(behavior[_BEHAVIOR_FIELDS[2]], signed=signed)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{GPT-5.6 Sol thinking-effort comparison on identical "
            r"shared-stop task 1 pairs. MPP scores are pooled across language "
            r"directions. Errors per prediction include omissions; mean span "
            r"length is computed over non-omission spans; and mean severity "
            r"weight includes omissions. Precision, recall, and F1 are percentage "
            r"points. Deltas are xhigh minus medium.}",
            r"\label{tab:task1-gpt-sol-stop}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def _gpt_sol_stop_markdown(result: GptSolStopResult) -> str:
    """Render pooled shared-stop MPP and behavior values as a Markdown table."""

    scores = {row["evaluator"]: row for row in result.scores}
    statistics = {row["evaluator"]: row for row in result.prediction_statistics}

    def behavior(value: float | None, *, signed: bool = False) -> str:
        """Format one behavior measure in its natural units."""

        return "—" if value is None else f"{value:+.2f}" if signed else f"{value:.2f}"

    rows = []
    for evaluator in (MEDIUM, XHIGH):
        score = scores[evaluator]
        stats = statistics[evaluator]
        rows.append((
            _LATEX_EVALUATOR_NAMES[evaluator],
            *(f"{100 * float(score[field]):.2f}" for field in ("precision", "recall", "f1")),
            *(behavior(stats[field]) for field in _BEHAVIOR_FIELDS),
        ))
    delta = scores["xhigh_minus_medium"]
    rows.append((
        "Δ (xhigh − medium)",
        *(f"{100 * float(delta[field]):+.2f}" for field in ("precision", "recall", "f1")),
        *(
            behavior(
                float(statistics[XHIGH][field]) - float(statistics[MEDIUM][field])
                if statistics[XHIGH][field] is not None
                and statistics[MEDIUM][field] is not None else None,
                signed=True,
            )
            for field in _BEHAVIOR_FIELDS
        ),
    ))
    return markdown_table(
        ("Auto-evaluator", "P", "R", "F1",
         "Errors per prediction", "Mean span length", "Mean severity weight"),
        rows,
    ) + (
        "\n\nPrecision, recall, and F1 are measured over evaluation items for which "
        "both models completed reasoning. Errors per prediction "
        "include omissions; mean span length excludes omissions; mean severity "
        "weight includes omissions."
    )

def write_paper_gpt_sol_stop_report(
    output_dir: Path, result: GptSolStopResult
) -> Path:
    """Write the medium-versus-xhigh shared-stop comparison as one paper TeX file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gpt_sol_stop.tex"
    path.write_text(_gpt_sol_stop_latex(result), encoding="utf-8")
    write_markdown_companion(
        path, "Task 1: GPT-5.6 Sol medium versus xhigh thinking",
        _gpt_sol_stop_markdown(result),
    )
    return path
