"""Release-path orchestration for the original post-hoc Task 2 threshold analysis."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .core import (AnalysisResult, analyze_system, load_official_metrics,
                   verify_declared_result, CV_FOLDS, NEAR_OPTIMAL_TOLERANCE)
from .plotting import (select_language_pairs, select_results_by_id, write_language_plots,
    write_paper_gain_plot, write_paper_grid_plot, write_pooled_plots, write_summary_plots)
from .reporting import write_result_tables
from .presentation import write_baseline_conversion_deltas
from ..io import DataError, sha256_file
from ..pipeline import load_context, conversion_decisions, write_run_manifest

PRIORITY_LANGUAGE_PAIRS = (
    "ces_Latn---deu_Latn",
    "ces_Latn---vie_Latn",
    "eng_Latn---arz_Arab",
    "eng_Latn---ekk_Latn",
    "eng_Latn---ind_Latn",
    "eng_Latn---isl_Latn",
    "eng_Latn---sme_Latn",
    "zho_Hans---jpn_Jpan",
)


PAPER_LANGUAGE_PAIRS = (
    "eng_Latn---arz_Arab",
    "eng_Latn---ekk_Latn",
    "zho_Hans---jpn_Jpan",
)


PAPER_EXCLUDED_PARTICIPANT_IDS = (
    "870257",
    "870554",
    "871298",
    "875949",
    "876231",
)


PAPER_SYSTEM_GROUPS = (
    (
        "participants",
        "participant",
        ("867652", "872328", "876271"),
        ("low P / high R", "mid P / mid R", "high P / low R"),
    ),
    (
        "baselines",
        "baseline",
        ("gemma4-reasoning", "gpt-5.6-sol-medium", "xcomet-xxl-no-reference"),
        ("lower P / high R", "mid P / mid R", "higher P / low R"),
    ),
)


def task1_mcc_by_system(
    rows_by_system: dict[str, list[dict[str, str]]],
) -> dict[str, dict[str, float]]:
    """Index per-language Task 1 MCC by the matching organizer baseline identity."""
    return {
        entry_key: {
            row["language_pair"]: float(row["mcc"])
            for row in rows
        }
        for entry_key, rows in rows_by_system.items()
    }


def run_analysis(args: argparse.Namespace) -> tuple[
    list[AnalysisResult],
    dict[str, dict[str, float]],
    dict[str, object],
]:
    """Load prepared inputs and run the original complete-priority threshold experiment."""
    inventory, human_reference, _ = load_context(args.data_dir)
    audit = args.data_dir / "input_manifest.json"
    _, participant_decisions, participant_names = conversion_decisions(args.data_dir, 2)
    _, baseline_decisions, baseline_names = conversion_decisions(args.data_dir, 2, baseline=True)
    official_participant = load_official_metrics(
        args.participant_metrics, "submission_id"
    )
    official_baseline = load_official_metrics(args.baseline_metrics, "entry_key")
    task1_rows = load_official_metrics(args.task1_baseline_metrics, "entry_key")

    results = []
    excluded_priority_coverage = []
    required_pairs = frozenset(PRIORITY_LANGUAGE_PAIRS)
    for track, decisions, names, task2_dir, official in (
        (
            "participant",
            participant_decisions,
            participant_names,
            args.task2_submissions_dir,
            official_participant,
        ),
        (
            "baseline",
            baseline_decisions,
            baseline_names,
            args.baseline_extracted_dir,
            official_baseline,
        ),
    ):
        for decision in decisions:
            if decision.status != "converted":
                continue
            official_rows = official.get(str(decision.submission_id))
            if not official_rows:
                raise DataError(f"Missing official metrics for {decision.submission_id}")
            covered_pairs = {row["language_pair"] for row in official_rows}
            missing_pairs = sorted(required_pairs - covered_pairs)
            if missing_pairs:
                print(
                    f"Skipping {track} {decision.submission_id}: missing priority pairs "
                    f"{', '.join(missing_pairs)}"
                )
                excluded_priority_coverage.append(
                    {
                        "track": track,
                        "entry_id": str(decision.submission_id),
                        "covered_priority_pairs": len(required_pairs & covered_pairs),
                        "missing_pairs": missing_pairs,
                    }
                )
                continue
            print(f"Analyzing {track} {decision.submission_id}")
            result = analyze_system(
                track,
                decision,
                names[decision.submission_id],
                task2_dir,
                inventory,
                human_reference,
                required_pairs,
            )
            priority_rows = [
                row for row in official_rows if row["language_pair"] in required_pairs
            ]
            verify_declared_result(result, priority_rows)
            results.append(result)

    context = {
        "audit": audit,
        "participant_decisions": participant_decisions,
        "baseline_decisions": baseline_decisions,
        "included_entry_ids": {
            track: [
                str(result.row["entry_id"])
                for result in results
                if result.row["track"] == track
            ]
            for track in ("participant", "baseline")
        },
        "excluded_priority_coverage": excluded_priority_coverage,
    }
    return results, task1_mcc_by_system(task1_rows), context


def write_paper_plots(
    output_dir: Path,
    results: list[AnalysisResult],
    task1_mcc: dict[str, dict[str, float]],
) -> list[Path]:
    """Render the original paper-selected grids and macro threshold-gain figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_results = [
        result
        for result in results
        if not (
            result.row["track"] == "participant"
            and str(result.row["entry_id"]) in PAPER_EXCLUDED_PARTICIPANT_IDS
        )
    ]
    written = []
    for label, track, entry_ids, regime_labels in PAPER_SYSTEM_GROUPS:
        selected = select_results_by_id(paper_results, track, list(entry_ids))
        path = output_dir / f"paper_threshold_{label}.pdf"
        write_paper_grid_plot(
            path,
            selected,
            list(PAPER_LANGUAGE_PAIRS),
            list(regime_labels),
            task1_mcc if track == "baseline" else None,
            track == "baseline",
        )
        written.append(path)
    gain_path = output_dir / "paper_macro_threshold_gains.pdf"
    write_paper_gain_plot(gain_path, [result.row for result in paper_results])
    written.append(gain_path)
    return written



def main(argv: list[str] | None = None) -> int:
    """Run the original analysis/plot sequence against the prepared release layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/task3"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/task3"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-language-pair", action="append", dest="plot_language_pairs")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or args.results_dir / "threshold"
    args.task2_submissions_dir = args.data_dir / "submissions/task2"
    args.baseline_extracted_dir = args.data_dir / "baselines"
    args.participant_metrics = args.results_dir / "task2_converted/metrics_by_language.csv"
    args.baseline_metrics = args.results_dir / "task2_baselines_converted/metrics_by_language.csv"
    args.task1_baseline_metrics = args.results_dir / "task1_baselines_converted/metrics_by_language.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, task1_mcc, context = run_analysis(args)
    write_result_tables(args.output_dir, results)
    rows = [result.row for result in results]
    write_summary_plots(args.output_dir, rows)
    plotted_pairs = select_language_pairs(results, args.plot_language_pairs)
    plot_files = write_pooled_plots(args.output_dir / "plots", results)
    plot_files.extend(write_language_plots(args.output_dir / "plots", results, plotted_pairs, task1_mcc))
    plot_files.extend(write_paper_plots(args.output_dir / "paper_figures", results, task1_mcc))
    templates = Path(__file__).resolve().parent / "templates"
    for template in templates.rglob("*.tex"):
        destination = args.output_dir / template.relative_to(templates)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if "@@DELTA_" not in template.read_text():
            shutil.copyfile(template, destination)
    write_baseline_conversion_deltas(args.output_dir, args.task1_baseline_metrics,
                                     PRIORITY_LANGUAGE_PAIRS)
    metadata = {
        "schema_version": 3,
        "status": "post-hoc sensitivity analysis; not an official evaluation result",
        "objective": "unweighted macro-average MCC over the priority eight language pairs",
        "priority_language_pairs": list(PRIORITY_LANGUAGE_PAIRS),
        "per_language_validation": "none; same-data optimistic upper bounds",
        "pooled_objective": "pooled unit-weight MCC",
        "positive_class": "error-free", "optimized_rule": "score >= threshold",
        "near_optimal_absolute_mcc_tolerance": NEAR_OPTIMAL_TOLERANCE,
        "cross_validation": {"folds": CV_FOLDS, "group": "complete item_id",
                             "assignment": "first 64 bits of SHA-256(item_id), modulo folds"},
        "included_entry_ids": context["included_entry_ids"],
        "excluded_incomplete_priority_coverage": context["excluded_priority_coverage"],
        "paper_excluded_participant_ids": list(PAPER_EXCLUDED_PARTICIPANT_IDS),
        "plotted_pairs": plotted_pairs,
        "files": {str(p.relative_to(args.output_dir)): sha256_file(p) for p in plot_files},
        "inputs": {str(p.relative_to(args.results_dir)): sha256_file(p) for p in
                   (args.participant_metrics, args.baseline_metrics, args.task1_baseline_metrics)},
        "input_manifest_sha256": sha256_file(args.data_dir / "input_manifest.json"),
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_run_manifest(args.results_dir, args.data_dir, role="scoring and threshold analysis")
    print(f"Wrote {len(results)} analyses and {len(plotted_pairs)} language-pair plot sets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
