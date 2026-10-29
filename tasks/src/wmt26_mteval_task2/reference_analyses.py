"""The three reference analyses reported in the Task 2 paper section."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

import numpy as np

from .metadata import DIRECT_ENTRIES
from .metrics import prepare_spa_reference, score_measure
from .paper_labels import PRIMARY_DIRECTIONS
from .pipeline import direct_entries, load_inputs
from .preprocessing import load_screens
from .reference_quality import analyze_reference_quality, load_reference_quality_index
from .reporting import tex_escape, write_csv
from .significance_progress import atomic_json
from .significance_storage import numerical_environment


# Each tuple changes reference use only, holding model and reasoning fixed.
REFERENCE_PAIRS = (
    ("Gemma 4", "gemma_task2", "gemma_ref_task2", True),
    ("Qwen 3.6", "qwen_task2", "qwen_ref_task2", True),
    ("Gemma 4", "gemma4_no_thinking_task2", "gemma_ref_no_thinking_task2", False),
    ("Qwen 3.6", "qwen36_no_thinking_task2", "qwen_ref_no_thinking_task2", False),
    ("xCOMET XL", "xcomet_xl_no_ref_task2", "xcomet_xl_ref_task2", False),
    ("xCOMET XXL", "xcomet_xxl_no_ref_task2", "xcomet_xxl_ref_task2", False),
)


def comparison_table(stem, rows, left_label, right_label, caption, label, *, digits=3):
    """Write one compact four-column TeX table and its Markdown view.

    Rows contain name, baseline, reasoning, left, and right. Differences always
    subtract the left condition from the right; rows are not statistical tests.
    """
    metric = "score" if digits == 1 else "Avg Corr"
    lines = [r"\begin{center}", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             f"Auto-evaluator & {left_label} & {right_label} & $\\Delta$ {metric} " + r"\\", r"\midrule"]
    md = [caption, "", f"| Auto-evaluator | {left_label} | {right_label} | Delta {metric} |",
          "| --- | ---: | ---: | ---: |"]
    for row in rows:
        name = tex_escape(row["name"]) + (r"\textsuperscript{\#}" if row.get("reasoning") else "")
        prefix = r"\rowcolor{black!15} " if row["baseline"] else ""
        left, right = row["left"], row["right"]
        lines.append(prefix + f"{name} & ${left:.{digits}f}$ & ${right:.{digits}f}$ & "
                     f"${right-left:+.{digits}f}$ " + r"\\")
        mdname = row["name"] + (" (reasoning)" if row.get("reasoning") else "")
        md.append(f"| {mdname} | {left:.{digits}f} | {right:.{digits}f} | {right-left:+.{digits}f} |")
    lines += [r"\bottomrule", r"\end{tabular}", r"\captionof{table}{" + caption + "}",
              r"\label{" + label + "}", r"\end{center}", ""]
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".tex").write_text("\n".join(lines), encoding="utf-8")
    stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def matched_references(inputs):
    """Return matched performance and score-scale summaries on fixed support.

    Auxiliary paper analyses retain their original float32 inner-SPA arithmetic;
    official rankings instead use the audited float64 significance engine.
    xCOMET scores stay on their submitted [0, 100] scale for summaries.
    """
    required = {identity for _, free, based, _ in REFERENCE_PAIRS for identity in (free, based)}
    if any(not required <= set(inputs.predictions[d]) for d in PRIMARY_DIRECTIONS):
        raise ValueError("All six matched reference pairs need complete primary coverage")
    direction_rows = []
    for direction in PRIMARY_DIRECTIONS:
        evidence = inputs.evidence[direction]
        reference = prepare_spa_reference(evidence, 1000, 4)
        keys = sorted((item, system) for item, values in evidence.segment_scores.items() for system in values)
        for name, free, based, reasoning in REFERENCE_PAIRS:
            values, means = [], []
            for identity in (free, based):
                predictions = inputs.predictions[direction][identity]
                pdp, _ = score_measure(evidence, predictions, "group-by-screen-pdp")
                spa, _ = score_measure(evidence, predictions, "segment-based-spa", spa_reference=reference)
                values.append((pdp + spa) / 2)
                means.append(float(np.array([predictions[item][system] for item, system in keys]).mean()))
            direction_rows.append(dict(direction=direction, name=name, reasoning=reasoning,
                free_id=free, based_id=based, support=len(keys), left=values[0], right=values[1],
                free_score=means[0], based_score=means[1]))
    performance, behavior = [], []
    for name, free, _, reasoning in REFERENCE_PAIRS:
        rows = [r for r in direction_rows if r["free_id"] == free]
        performance.append(dict(name=name, baseline=True, reasoning=reasoning,
                                left=fmean(r["left"] for r in rows), right=fmean(r["right"] for r in rows)))
        behavior.append(dict(name=name, baseline=True, reasoning=reasoning,
                             left=fmean(r["free_score"] for r in rows), right=fmean(r["based_score"] for r in rows)))
    return performance, behavior, direction_rows


def main(argv=None):
    """Reproduce four reference tables, with optional full-precision diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, default=Path("test-set-files/mteval-test26.jsonl"))
    parser.add_argument("--ground-truth", type=Path, default=Path("human_annotations/ground-truth.jsonl"))
    parser.add_argument("--input-dir", type=Path, action="append", dest="input_dirs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/references"))
    parser.add_argument("--audit-details", action="store_true")
    args = parser.parse_args(argv)
    roots = args.input_dirs or [Path("submissions/task2"), Path("baselines/task2")]
    inputs = load_inputs(args.test_set, args.ground_truth, roots, scope="primary")
    performance, behavior, direction_rows = matched_references(inputs)
    out = args.output_dir
    common = ("Values equally average the eight priority directions. "
              r"\textsuperscript{\#} marks reasoning evaluators. Light-grey rows are baselines.")
    comparison_table(out / "task2-reference-use-effects", performance, "Ref-free", "Ref-based",
        "Effect of reference use in strictly matched Task~2 baselines. Avg Corr averages screen-based PDP and source-based SPA. "
        + common, "tab:task2-reference-use-effects")
    comparison_table(out / "task2-reference-score-behavior", behavior, "Ref-free", "Ref-based",
        "Mean submitted scores on translations with human assessments, excluding the designated reference. "
        "xCOMET scores are reported on their submitted [0,100] scale. These are score-scale shifts, not performance differences. "
        + common, "tab:task2-reference-score-behavior", digits=1)

    index = load_reference_quality_index(args.test_set)
    entries = [entry for entry in direct_entries(roots) if DIRECT_ENTRIES[entry.submission_id].uses_references]
    quality = analyze_reference_quality(entries, DIRECT_ENTRIES, index, load_screens(args.ground_truth))
    comparisons = {}
    for comparison, stem, label, left, right in (
        ("main_vs_pseudo", "task2-main-reference-vs-pseudoref", "tab:task2-main-reference-pseudoref", "Main human", "PseudoRef"),
        ("from_scratch_vs_postedit", "task2-from-scratch-vs-postedit", "tab:task2-from-scratch-postedit", "From scratch", "Post-edited"),
    ):
        selected = [row for row in quality.direction_scores if row.comparison == comparison and row.direction in PRIMARY_DIRECTIONS]
        directions = sorted({row.direction for row in selected})
        rows = []
        for identity in sorted({row.submission_id for row in selected}):
            values = [row for row in selected if row.submission_id == identity]
            if {row.direction for row in values} != set(directions):
                raise ValueError("Incomplete common direction set in reference comparison")
            name = values[0].submission_name.replace(" Reference-based - ", " RB ").replace(
                " Reference-based", " RB").replace(" - Baseline", "").replace(" Baseline", "")
            rows.append(dict(name=name, baseline=values[0].entry_type == "baseline", reasoning=False,
                             left=fmean(row.left_task2 for row in values), right=fmean(row.right_task2 for row in values)))
        rows.sort(key=lambda row: (-row["left"], row["name"]))
        if not rows:
            raise ValueError(f"No eligible matched reference-quality data: {comparison}")
        caption = (f"Task~2 reference comparison over {len(directions)} matched priority directions, equally weighted. "
                   "Avg Corr averages screen-based PDP and source-based SPA. Both conditions use identical sources, "
                   "MT translations, and human evidence; all from-scratch, post-edited, and pseudo-reference candidates are removed. "
                   "Light-grey rows are baselines.")
        if comparison == "main_vs_pseudo":
            caption += " The main human reference is from scratch for Czech to German, English to Estonian and Icelandic, and post-edited for Chinese to Japanese."
        comparison_table(out / stem, rows, left, right, caption, label)
        comparisons[comparison] = dict(directions=directions, evaluators=len(rows))
    atomic_json(out / "run.json", dict(input_sha256=inputs.hashes, numerical_environment=numerical_environment(),
        spa_permutations=1000, spa_seed=4, significance=False, comparisons=comparisons,
        reference_quality_issues=quality.issues,
        matched_support={d: sum(len(v) for v in inputs.evidence[d].segment_scores.values()) for d in PRIMARY_DIRECTIONS}))
    if args.audit_details:
        write_csv(out / "audit" / "matched_directions.csv", direction_rows)
        write_csv(out / "audit" / "quality_directions.csv", [asdict(row) for row in quality.direction_scores])
        write_csv(out / "audit" / "quality_eligibility.csv", [asdict(row) for row in quality.eligibility])
