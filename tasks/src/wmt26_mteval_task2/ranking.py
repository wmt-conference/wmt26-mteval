"""Point estimates and optional resumable significance for the paper rankings."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .conversion_labels import CONVERSION_DIRECTIONS
from .paper_labels import PRIMARY_DIRECTIONS
from .pipeline import aggregate_ids, load_inputs
from .reporting import ranking_rows, ranking_table, ranking_markdown, write_csv, write_ranking
from .significance import aggregate_exchanges, observed_components, prepare_direction, source_key
from .significance_progress import atomic_json
from .significance_reporting import compare_scope
from .significance_runner import run_permutations
from .significance_storage import canonical_hash, exclusive_lock, numerical_environment
from .domain_coverage import file_sha256


def subset_direction(direction, indices):
    """Select evaluators without recomputing any of their prepared statistics."""
    return replace(direction, pdp_numerators=direction.pdp_numerators[indices],
                   pdp_norms=direction.pdp_norms[indices],
                   prediction_means=direction.prediction_means[indices],
                   prediction_sds=direction.prediction_sds[indices],
                   widths=tuple(replace(w, differences=w.differences[indices],
                                        metric_counts=w.metric_counts[indices]) for w in direction.widths))


def prepare_inputs(inputs, *, conversion=False):
    """Return sorted evaluator identities and immutable statistics per direction.

    The primary direct runs retain their original sufficient-sum arithmetic.
    Both orientations are explicitly expanded for conversion and nonprimary
    runs, matching the production runs that produced those paper tables.
    """
    identities, prepared = {}, {}
    for direction in inputs.directions:
        ids = sorted(inputs.predictions[direction])
        if len(ids) < 2:
            raise ValueError(f"At least two complete evaluators required in {direction}")
        lookup = inputs.source_lookup if direction in PRIMARY_DIRECTIONS else {
            key: i for i, key in enumerate(sorted({source_key(item)
                 for item in inputs.evidence[direction].segment_scores}))}
        print(f"Preparing {direction}: {len(ids)} evaluators", flush=True)
        prepared[direction] = prepare_direction(
            inputs.evidence[direction], [inputs.predictions[direction][i] for i in ids], lookup,
            ordered_pdp_differences=conversion or direction not in PRIMARY_DIRECTIONS)
        identities[direction] = ids
    return identities, prepared


def execute(args) -> None:
    """Compute requested rankings; incomplete permutation runs retain checkpoints."""
    conversion = args.track == "conversion"
    roots = args.input_dirs or [Path("submissions/task2"), Path("baselines/task2")]
    inputs = load_inputs(args.test_set, args.ground_truth, roots, track=args.track, scope=args.scope,
                        task1_submissions=args.task1_submissions_dir, task1_dedup=args.task1_dedup_dir,
                        task1_original=args.task1_original_dir)
    ids_by_direction, prepared = prepare_inputs(inputs, conversion=conversion)
    ids = aggregate_ids(inputs)
    directions = CONVERSION_DIRECTIONS if conversion else PRIMARY_DIRECTIONS
    aggregate = tuple(subset_direction(prepared[d], [ids_by_direction[d].index(i) for i in ids])
                      for d in directions) if ids else ()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(track=args.track, scope=args.scope, directions=inputs.directions,
                      input_sha256=inputs.hashes, designated_reference_exclusions=inputs.exclusions,
                      eligible_ids=ids_by_direction, aggregate_ids=ids,
                      spa_permutations=1000, spa_seed=4, permutations=args.permutations,
                      seed=args.seed, chunk_size=args.chunk_size,
                      numerical_environment=numerical_environment(),
                      code_sha256={p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    fingerprint = canonical_hash(provenance)
    provenance["input_fingerprint"] = fingerprint
    # Numerical checkpoints independently reject changed provenance or inputs.
    provenance["human_support"] = [asdict(inputs.evidence[d].statistics) for d in inputs.directions]
    provenance["excluded_entries"] = [row for row in inputs.eligibility if row["status"] != "valid"]
    manifest_path = out / "run.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if previous and previous.get("input_fingerprint") != fingerprint:
        checkpoint = out / "checkpoints"
        if checkpoint.exists() and any(checkpoint.rglob("permutations.sqlite3")):
            raise ValueError("Inputs or code differ from saved significance draws; select a new output directory")
    atomic_json(manifest_path, provenance)
    if args.audit_details:
        write_csv(out / "audit" / "eligibility.csv", inputs.eligibility)
    draws = args.permutations

    def permutations(values, identities, sources, destination, pairs=None):
        """Run deterministic chunks and stop this invocation after a partial run."""
        arrays, status = run_permutations(
            values, identities, sources, destination / "permutations.sqlite3", fingerprint,
            permutations=draws, seed=args.seed, workers=args.workers, chunk_size=args.chunk_size,
            pairs=pairs, stop_after_chunks=args.stop_after_chunks, start_method=args.start_method,
            progress=args.progress, progress_path=destination / "progress.json")
        if not status["complete"]:
            print(f"Partial run saved at {destination}. Repeat the same command to resume.", flush=True)
            raise SystemExit(130 if status["interrupted"] else 0)
        return arrays

    aggregate_samples = {}
    if aggregate:
        observed = observed_components(aggregate)
        comparison = None
        if draws:
            aggregate_samples = permutations(aggregate, ids, len(inputs.source_lookup), out / "checkpoints" / "aggregate")
            comparison = compare_scope(ids, observed.mean(axis=1),
                {pair: aggregate_exchanges(value) for pair, value in aggregate_samples.items()},
                averages=observed.mean(axis=(1, 2)))
            if args.audit_details:
                write_csv(out / "audit" / "aggregate_pairwise_tests.csv", comparison[1])
        rows = ranking_rows(ids, observed, inputs.metadata, comparison=comparison)
        write_ranking(out / "priority_ranking", rows, conversion=conversion, draws=draws)
        if args.audit_details:
            write_csv(out / "audit" / "aggregate_ranking.csv", rows)
            write_csv(out / "audit" / "observed_components.csv", [dict(submission_id=identity, direction=d,
                  screen_pdp=float(observed[i, j, 0]), source_spa=float(observed[i, j, 1]))
                  for i, identity in enumerate(ids) for j, d in enumerate(directions)])
    if conversion:
        return  # The conversion track has one aggregate table in the paper.
    tables, markdown = [], []
    for direction in inputs.directions:
        local_ids = ids_by_direction[direction]
        current = prepared[direction]
        observed = observed_components((current,))
        destination = out / "checkpoints" / direction
        comparison = None
        if draws:
            pairs = list(itertools.combinations(range(len(local_ids)), 2))
            samples = {}
            if direction in PRIMARY_DIRECTIONS and aggregate_samples:
                column = directions.index(direction)
                positions = {identity: i for i, identity in enumerate(local_ids)}
                samples = {(positions[ids[a]], positions[ids[b]]): value[:, :, column, :]
                           for (a, b), value in aggregate_samples.items()}
            extra_pairs = [pair for pair in pairs if pair not in samples]
            if extra_pairs:
                sources = len(inputs.source_lookup) if direction in PRIMARY_DIRECTIONS else len(current.sources)
                extra = permutations((current,), local_ids, sources, destination, extra_pairs)
                samples.update({pair: value[:, :, 0, :] for pair, value in extra.items()})
            comparison = compare_scope(local_ids, observed[:, 0, :], samples)
            if args.audit_details:
                write_csv(out / "audit" / direction / "pairwise_tests.csv", comparison[1])
        rows = ranking_rows(local_ids, observed, inputs.metadata, comparison=comparison)
        tables.append(ranking_table(rows, direction=direction, draws=draws))
        markdown.append(f"## {direction}\n\n" + ranking_markdown(rows))
        if args.audit_details:
            write_csv(out / "audit" / direction / "ranking.csv", rows)
    (out / "per_direction_rankings.tex").write_text("\n".join(tables), encoding="utf-8")
    (out / "per_direction_rankings.md").write_text("\n".join(markdown), encoding="utf-8")
    if draws:
        provenance["significance"] = dict(
            draws=draws, one_sided_and_two_sided=True, alpha=0.05,
            multiple_comparison_correction=None, completed=True,
            checkpoint="checkpoints/",
        )
    atomic_json(manifest_path, provenance)


def main(argv=None) -> None:
    """Parse paths and runtime controls for the direct or conversion ranking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, default=Path("test-set-files/mteval-test26.jsonl"))
    parser.add_argument("--ground-truth", type=Path, default=Path("human_annotations/ground-truth.jsonl"))
    parser.add_argument("--input-dir", action="append", type=Path, dest="input_dirs")
    parser.add_argument("--track", choices=("direct", "conversion"), default="direct")
    parser.add_argument("--task1-submissions-dir", type=Path, default=Path("submissions/task1"))
    parser.add_argument("--task1-dedup-dir", type=Path, default=Path("baselines/task1_deduplicated"))
    parser.add_argument("--task1-original-dir", type=Path, default=Path("baselines/task1_prededup_and_enhy_fix"))
    parser.add_argument("--scope", choices=("all", "primary", "nonprimary"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--permutations", type=int, default=0,
                        help="0 for point estimates; use 1000 for the paper's significance tests")
    parser.add_argument("--seed", type=int, default=98001)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--start-method", choices=("fork", "spawn"), default="spawn")
    parser.add_argument("--progress", choices=("auto", "plain", "off"), default="auto")
    parser.add_argument("--stop-after-chunks", type=int, help="Diagnostic interruption/resumption check")
    parser.add_argument("--audit-details", action="store_true", help="Also write detailed CSV diagnostics")
    args = parser.parse_args(argv)
    if args.permutations < 0 or args.seed < 0 or min(args.workers, args.chunk_size) < 1:
        parser.error("Invalid permutation, seed, worker, or chunk count")
    if args.track == "conversion" and args.scope == "nonprimary":
        parser.error("Conversion results cover only priority directions")
    if args.output_dir is None:
        suffix = "significance" if args.permutations else "scores"
        args.output_dir = Path("outputs/task2") / suffix / f"{args.track}_{args.scope}"
    with exclusive_lock(args.output_dir / ".run.lock"):
        execute(args)
