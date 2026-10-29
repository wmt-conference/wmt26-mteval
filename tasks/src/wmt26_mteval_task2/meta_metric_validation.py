"""Reproduce the appendix's screen/source correlations and SPA support figure."""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path
from statistics import fmean, median

from .constants import MEASURE_NAMES
from .domain_coverage import file_sha256, load_domain_index
from .io import load_predictions_for_directions
from .metadata import DIRECT_ENTRIES
from .metrics import pearson_correlation, prepare_spa_reference, score_measure
from .paper_labels import PRIMARY_DIRECTIONS, TEX_DIRECTION_LABELS
from .pipeline import direct_entries
from .preprocessing import aggregate_human_evidence, load_screens
from .reporting import write_csv
from .significance_progress import atomic_json
from .significance_storage import numerical_environment
from .support_plot import plot_support


def pair_support(evidence):
    """Describe per-MT-pair overlap, retaining the historical paper denominators.

    Screen summary statistics cover pairs with at least one co-screen item.
    Source summary statistics cover all possible pairs, including zero overlap.
    Report their counts so the reader can see the different populations.
    """
    screen_counts = Counter((a, b) for _, a, b in evidence.co_screen_scores)
    source_counts = [sum(a in values and b in values for values in evidence.segment_scores.values())
                     for a, b in itertools.combinations(evidence.systems, 2)]
    return dict(direction=evidence.direction, system_pairs=len(source_counts),
        screen_pairs=len(screen_counts), screen_zero_pairs=len(source_counts)-len(screen_counts),
        source_zero_pairs=sum(n == 0 for n in source_counts),
        screen_spa_items_median=median(screen_counts.values()),
        screen_spa_items_mean=fmean(screen_counts.values()),
        segment_spa_items_median=median(source_counts), segment_spa_items_mean=fmean(source_counts))


def strict_rank_correlations(scores, metadata):
    """Compare deterministic strict ranks, including the original tie breakers.

    This is not average-rank Spearman on tied scores: the paper diagnostic used
    strict ranks, with ties ordered by submission name and then stable ID.
    """
    identities = sorted(scores)
    ranks = {}
    for measure in MEASURE_NAMES:
        ordered = sorted(identities, key=lambda i: (-scores[i][measure], metadata[i].submission_name.casefold(), i))
        ranks[measure] = {identity: rank for rank, identity in enumerate(ordered, 1)}
    result = {}
    for family, left, right in (("pdp", "group-by-screen-pdp", "group-by-segment-pdp"),
                                ("spa", "screen-based-spa", "segment-based-spa")):
        result[family + "_spearman"] = pearson_correlation(
            [ranks[left][i] for i in identities], [ranks[right][i] for i in identities])
    return result


def main(argv=None):
    """Write one correlation table, one support PDF, Markdown views and a manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, default=Path("test-set-files/mteval-test26.jsonl"))
    parser.add_argument("--ground-truth", type=Path, default=Path("human_annotations/ground-truth.jsonl"))
    parser.add_argument("--input-dir", type=Path, action="append", dest="input_dirs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/meta_metric_validation"))
    parser.add_argument("--exclude-placeholder", action="store_true",
                        help="Sensitivity check only: remove the constant GPT xhigh diagnostic entry")
    parser.add_argument("--audit-details", action="store_true")
    args = parser.parse_args(argv)
    roots = args.input_dirs or [Path("submissions/task2"), Path("baselines/task2")]
    entries = direct_entries(roots, include_placeholder=not args.exclude_placeholder)
    index = load_domain_index(args.test_set, PRIMARY_DIRECTIONS)
    screens = load_screens(args.ground_truth)
    evidence = {d: aggregate_human_evidence(index.test_sets[d], screens, d,
        excluded_systems=(index.designated_references[d],)) for d in PRIMARY_DIRECTIONS}
    references = {d: prepare_spa_reference(ev, 1000, 4) for d, ev in evidence.items()}
    scores = {d: {} for d in PRIMARY_DIRECTIONS}
    hashes = {"test_set": file_sha256(args.test_set), "ground_truth": file_sha256(args.ground_truth)}
    for position, entry in enumerate(entries, 1):
        print(f"Scoring grouping variants {position}/{len(entries)}: {entry.submission_id}", flush=True)
        hashes[entry.submission_id] = file_sha256(entry.scores_path)
        outcomes = load_predictions_for_directions(entry.scores_path, index.test_sets)
        for direction, outcome in outcomes.items():
            if outcome.status != "valid":
                continue
            scores[direction][entry.submission_id] = {measure: score_measure(
                evidence[direction], outcome.predictions, measure, spa_reference=references[direction])[0]
                for measure in MEASURE_NAMES}
    rows = [dict(direction=d, entries=len(scores[d]), **strict_rank_correlations(scores[d], DIRECT_ENTRIES))
            for d in PRIMARY_DIRECTIONS]
    support = [pair_support(evidence[d]) for d in PRIMARY_DIRECTIONS]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrr}", r"\toprule",
             r"Direction & PDP $\rho$ & SPA $\rho$ \\", r"\midrule"]
    for row in rows:
        lines.append(f"{TEX_DIRECTION_LABELS[row['direction']]} & {row['pdp_spearman']:.3f} & "
                     f"{row['spa_spearman']:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Correlations between strict evaluator rankings under screen and source grouping on the eight priority directions. Task~1 conversions are excluded.}",
        r"\label{tab:task2-screen-source-correlation}", r"\end{table}", ""]
    (out / "task2-meta-metric-correlations.tex").write_text("\n".join(lines), encoding="utf-8")
    md = ["| Direction | Evaluators | PDP rho | SPA rho |", "| --- | ---: | ---: | ---: |"]
    md += [f"| {r['direction']} | {r['entries']} | {r['pdp_spearman']:.6f} | {r['spa_spearman']:.6f} |" for r in rows]
    (out / "task2-meta-metric-correlations.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    plot_support(out / "task2-screen-vs-source-spa-support.pdf", support)
    fields = list(support[0])
    md = ["Screen averages cover positive co-screen support; source averages cover all MT pairs.", "",
          "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    md += ["| " + " | ".join(str(row[f]) for f in fields) + " |" for row in support]
    (out / "task2-screen-vs-source-spa-support.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    atomic_json(out / "run.json", dict(input_sha256=hashes, numerical_environment=numerical_environment(),
        spa_permutations=1000, spa_seed=4, strict_ranks=True,
        legacy_constant_placeholder_included=not args.exclude_placeholder,
        eligible_ids={d: sorted(values) for d, values in scores.items()}))
    if args.audit_details:
        write_csv(out / "audit" / "correlations.csv", rows)
        write_csv(out / "audit" / "support.csv", support)
