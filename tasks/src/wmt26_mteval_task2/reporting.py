"""Machine-readable results and matching human-readable paper tables."""

from __future__ import annotations

import csv
from pathlib import Path

from .conversion_labels import CONVERSION_FINE_TUNED
from .paper_labels import FINE_TUNED_IDS, TEX_MODELS, TEX_NAMES, TEX_DIRECTION_LABELS


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write ordered dictionaries as UTF-8 CSV, including an empty-file case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def tex_escape(value: str) -> str:
    """Escape plain labels without interpreting user-provided TeX commands."""
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                    "$": r"\$", "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}"}
    return "".join(replacements.get(character, character) for character in value)


def ranking_rows(ids, observed, metadata, *, comparison=None) -> list[dict]:
    """Format exact full-precision components and optional two-tail clusters.

    Observed axes are evaluator/direction/component. Avg Corr uses the same
    NumPy reduction order as the frozen significance runs.
    """
    averages = observed.mean(axis=(1, 2))
    order = sorted(range(len(ids)), key=lambda i: (-averages[i], ids[i]))
    if comparison is not None and order != comparison[0]:
        raise ValueError("Ranking and statistical-test orders differ")
    rows = []
    for rank, index in enumerate(order, 1):
        entry = metadata[ids[index]]
        row = dict(rank=rank, submission_id=ids[index], submission_name=entry.submission_name,
                   entry_type=entry.entry_type, uses_references=entry.uses_references,
                   based_on_models=entry.based_on_models,
                   screen_pdp=float(observed[index, :, 0].mean()),
                   source_spa=float(observed[index, :, 1].mean()),
                   average_correlation=float(averages[index]))
        if comparison is not None:
            row.update(one_sided_cluster=int(comparison[5][rank - 1]),
                       two_sided_cluster=int(comparison[4][rank - 1]))
        rows.append(row)
    return rows


def ranking_table(rows: list[dict], *, direction=None, conversion=False, draws=1000) -> str:
    """Render a floating booktabs table with the paper's labels and symbols."""
    clusters = "two_sided_cluster" in rows[0]
    columns = "rrllrrr" if clusters else "rllrrr"
    header = "Rank & " + ("Cluster & " if clusters else "")
    header += "Auto-evaluator & Underlying model & PDP & SPA & Avg Corr " + r"\\"
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{columns}}}", r"\toprule", header, r"\midrule"]
    for row in rows:
        identity = row["submission_id"]
        name = row["submission_name"] if conversion else TEX_NAMES[identity]
        model = row["based_on_models"] if conversion else TEX_MODELS[identity]
        symbols = "" if row["uses_references"] else r"\textsuperscript{*}"
        if identity in (CONVERSION_FINE_TUNED if conversion else FINE_TUNED_IDS):
            symbols += r"\textsuperscript{\textdagger}"
        prefix = r"\rowcolor{black!15} " if row["entry_type"] == "baseline" else ""
        cluster = f"${row['two_sided_cluster']}$ & " if clusters else ""
        lines.append(prefix + f"${row['rank']}$ & {cluster}{tex_escape(name)}{symbols} & {model} & "
                     f"${row['screen_pdp']:.3f}$ & ${row['source_spa']:.3f}$ & "
                     f"${row['average_correlation']:.3f}$ " + r"\\")
    scope = "the eight priority translation directions" if direction is None else TEX_DIRECTION_LABELS[direction]
    caption = ("Task~1-to-Task~2 conversion ranking" if conversion else "Task~2 ranking") + f" on {scope}. "
    caption += ("Only evaluators with complete predictions on every retained human-supported translation in this scope are included. "
                "Avg Corr equally averages screen-based PDP and source-based SPA; aggregate components give each direction equal weight. "
                r"\textsuperscript{*} marks reference-free evaluators; \textsuperscript{\textdagger} marks fine-tuned evaluators. "
                "Light-grey rows are organizer baselines. ")
    if not conversion:
        caption += ("E1: Gemini 3.5 Flash, Gemma 4 12B, Gemma 4 31B. "
                    "E2: GPT-5.4, Gemini 3.5 Flash, GPT-5.6 Luna, COMET-22. "
                    "E3: Gemini 2.5 Flash, DeepSeek V4 Flash, Qwen Plus. ")
    else:
        caption += "Dedup labels distinguish the supplied span snapshots. "
    if clusters:
        caption += (f"Statistical significance uses two-sided source-level permutation tests with {draws:,} draws "
                    r"and unadjusted $p\leq0.05$, followed by greedy WMT clustering "
                    r"\citep{deutsch-etal-2021-statistical,freitag-etal-2024-llms,lavie-etal-2025-findings}. ")
    label = ("tab:task2-task1-conversion-ranking" if conversion else "tab:task2-priority-results") if direction is None else (
        "tab:wmt26-task2-" + direction.replace("_", "-"))
    lines += [r"\bottomrule", r"\end{tabular}", "\\caption{" + caption.strip() + "}",
              "\\label{" + label + "}", r"\end{table*}", ""]
    return "\n".join(lines)


def ranking_markdown(rows: list[dict]) -> str:
    """Return a compact view of the ranking, with both tails when computed."""
    clusters = "two_sided_cluster" in rows[0]
    fields = ["rank", "submission_name", "screen_pdp", "source_spa", "average_correlation"]
    if clusters:
        fields[1:1] = ["one_sided_cluster", "two_sided_cluster"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[k]:.6f}" if isinstance(row[k], float)
                                          else str(row[k]).replace("|", "\\|") for k in fields) + " |")
    return "\n".join(lines) + "\n"


def write_ranking(stem: Path, rows: list[dict], **options) -> str:
    """Write just paired TeX/Markdown artifacts; return the rendered TeX."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    tex = ranking_table(rows, **options)
    stem.with_suffix(".tex").write_text(tex, encoding="utf-8")
    stem.with_suffix(".md").write_text(ranking_markdown(rows), encoding="utf-8")
    return tex
