"""Original CLEAN_CODE permutation kernels with prepared-release input adapters.

Statistical kernels are transplanted from significance-test/CLEAN_CODE/src/scoring.py;
only data loading, CLI configuration and output binding are release-specific.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io import DataError, iter_jsonl, load_json
from .model_cards import load_answers, resolved_base_model
from .pipeline import load_context, write_run_manifest
from .reporting import write_csv

@dataclass(frozen=True)
class _Item:
    """Store typed item state shared by the original analysis routines."""
    pair: str
    source_cluster: tuple[str, str, str, str, str]


@dataclass
class _System:
    """Store typed system state shared by the original analysis routines."""
    metric_key: str
    organization: str
    model: str
    covered_pairs: set[str]
    predictions: dict[tuple[str, str], bool]


@dataclass(frozen=True)
class ScoredSystem:
    """Store typed scored system state shared by the original analysis routines."""
    metric_key: str
    organization: str
    model: str
    rank: int
    cluster: int
    mcc: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class _Scores:
    """Store typed scores state shared by the original analysis routines."""
    mcc: float
    precision: float
    recall: float
    f1: float


@dataclass
class _Dataset:
    """Store typed dataset state shared by the original analysis routines."""
    items: dict[str, _Item]
    labels: dict[tuple[str, str], tuple[bool, ...]]
    systems: list[_System]
    all_pairs: tuple[str, ...]


def _item_parts(item_id: str) -> tuple[str, str, tuple[str, str, str, str, str]]:
    """Split an item ID into set, language pair and shared source-segment cluster."""
    parts = item_id.split("_###_")
    if len(parts) != 6:
        raise DataError(f"Invalid item ID: {item_id}")
    set_id, source, target, domain, document, segment = parts
    return (
        set_id,
        f"{source}---{target}",
        (set_id, source, domain, document, segment),
    )


def _is_clean(task1_prediction: dict[str, Any]) -> bool:
    """Return whether Task 1 predicts no error, omission or instruction fault."""
    return (
        not task1_prediction["errors"]
        and task1_prediction["omission"] is None
        and task1_prediction.get("instruction_fault") is None
    )


def _predictions(
    rows,
    field: str,
    convert: Callable[[Any], bool],
    human_keys: set[tuple[str, str]],
) -> tuple[set[str], dict[tuple[str, str], bool]]:
    """Extract official predictions and coverage restricted to human-assessed keys."""
    covered_pairs = set()
    predictions = {}
    for row in rows:
        item_id = row["item_id"]
        set_id, pair, _ = _item_parts(item_id)
        if set_id != "official":
            continue
        covered_pairs.add(pair)
        for system, value in row[field].items():
            key = item_id, system
            if key in human_keys:
                predictions[key] = convert(value)
    return covered_pairs, predictions


def _mcc(counts: np.ndarray) -> np.ndarray:
    """Vectorize the original zero-denominator MCC calculation over confusion arrays."""
    values = np.asarray(counts, dtype=np.float64)
    tp, fp, fn, tn = (values[..., index] for index in range(4))
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )


def _scores(counts: np.ndarray) -> _Scores:
    """Compute the original scalar MCC/precision/recall/F1 from confusion counts."""
    tp, fp, fn, _ = (int(counts[index]) for index in range(4))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return _Scores(float(_mcc(counts)), precision, recall, f1)


def _pair_counts(
    systems: list[_System],
    pairs: tuple[str, ...],
    dataset: _Dataset,
) -> tuple[list[np.ndarray], list[list[str]]]:
    """Construct per-system, per-item confusion arrays for each selected language pair."""
    keys_by_pair = {
        pair: sorted(
            (key for key in dataset.labels if dataset.items[key[0]].pair == pair)
        )
        for pair in pairs
    }
    all_counts = []
    item_ids_by_pair = []
    for pair in pairs:
        keys = keys_by_pair[pair]
        item_ids = sorted({item_id for item_id, _ in keys})
        item_index = {item_id: index for index, item_id in enumerate(item_ids)}
        counts = np.zeros((len(systems), len(item_ids), 4), dtype=np.int64)
        for item_id, translation_system in keys:
            labels = dataset.labels[(item_id, translation_system)]
            positives = sum(labels)
            negatives = len(labels) - positives
            for metric_index, system in enumerate(systems):
                prediction = system.predictions[(item_id, translation_system)]
                if prediction:
                    counts[metric_index, item_index[item_id], :2] += positives, negatives
                else:
                    counts[metric_index, item_index[item_id], 2:] += positives, negatives
        all_counts.append(counts)
        item_ids_by_pair.append(item_ids)
    return all_counts, item_ids_by_pair


def _permutation_p_values(
    counts_by_pair: list[np.ndarray],
    item_ids_by_pair: list[list[str]],
    items: dict[str, _Item],
    permutations: int,
    seed: int,
) -> np.ndarray:
    """Return unadjusted two-sided p-values for every system pair.

    Each permutation exchanges the two systems' labels by source segment. The
    same exchange is used for occurrences shared across target languages, and
    the p-value uses the absolute macro-MCC difference with add-one correction.
    """
    clusters = sorted(
        {
            items[item_id].source_cluster
            for item_ids in item_ids_by_pair
            for item_id in item_ids
        }
    )
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    indexes_by_pair = [
        np.asarray([cluster_index[items[item_id].source_cluster] for item_id in item_ids])
        for item_ids in item_ids_by_pair
    ]
    totals = [counts.sum(axis=1) for counts in counts_by_pair]
    observed = np.mean([_mcc(total) for total in totals], axis=0)
    observed_difference = observed[:, None] - observed[None, :]
    extreme = np.zeros_like(observed_difference, dtype=np.int64)
    rng = np.random.default_rng(seed)
    metric_count = len(observed)

    completed = 0
    chunk_size = 1_000
    while completed < permutations:
        current = min(chunk_size, permutations - completed)
        swaps = rng.integers(0, 2, size=(current, len(clusters)), dtype=np.int8)
        differences = np.zeros((current, metric_count, metric_count))
        for counts, totals_for_pair, indexes in zip(
            counts_by_pair, totals, indexes_by_pair
        ):
            flattened = counts.transpose(1, 0, 2).reshape(len(indexes), -1)
            flattened = flattened.astype(np.float64, copy=False)
            selected = (swaps[:, indexes] @ flattened).reshape(
                current, metric_count, 4
            )
            shift = selected[:, None, :, :] - selected[:, :, None, :]
            left = totals_for_pair[None, :, None, :] + shift
            right = totals_for_pair[None, None, :, :] - shift
            differences += (_mcc(left) - _mcc(right)) / len(counts_by_pair)
        extreme += np.count_nonzero(
            np.abs(differences) >= np.abs(observed_difference)[None, :, :] - 1e-15,
            axis=0,
        )
        completed += current
    return (extreme + 1) / (permutations + 1)


def _seed(base_seed: int, pairs: tuple[str, ...]) -> int:
    """Derive the original deterministic RNG seed from the base seed and ordered pair scope."""
    digest = hashlib.sha256(f"{base_seed}:{','.join(pairs)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _ranked_results(
    systems: list[_System],
    scores: dict[str, _Scores],
    p_values: dict[tuple[str, str], float],
    alpha: float,
) -> list[ScoredSystem]:
    """Apply competition ranking and WMT's greedy significance clustering.

    A system starts a new cluster when it differs significantly from any
    system already admitted to the current cluster.
    """
    systems = sorted(
        systems, key=lambda system: (-scores[system.metric_key].mcc, system.metric_key)
    )
    ranks = []
    previous_mcc = None
    rank = 0
    for position, system in enumerate(systems, start=1):
        if scores[system.metric_key].mcc != previous_mcc:
            rank = position
            previous_mcc = scores[system.metric_key].mcc
        ranks.append(rank)

    clusters = []
    cluster = 1
    cluster_start = 0
    for candidate_index, candidate in enumerate(systems):
        significant = any(
            p_values[tuple(sorted((earlier.metric_key, candidate.metric_key)))] < alpha
            for earlier in systems[cluster_start:candidate_index]
        )
        if significant:
            cluster += 1
            cluster_start = candidate_index
        clusters.append(cluster)

    results = []
    for system, rank, cluster in zip(systems, ranks, clusters):
        score = scores[system.metric_key]
        results.append(
            ScoredSystem(
                system.metric_key,
                system.organization,
                system.model,
                rank,
                cluster,
                score.mcc,
                score.precision,
                score.recall,
                score.f1,
            )
        )
    return results


def _analyze(
    dataset: _Dataset,
    pairs: tuple[str, ...],
    analysis: dict[str, Any],
) -> tuple[list[_System], dict[str, _Scores], dict[tuple[str, str], float]]:
    """Use an unweighted mean of the independently computed per-pair scores."""
    systems = [system for system in dataset.systems if set(pairs) <= system.covered_pairs]
    counts_by_pair, item_ids_by_pair = _pair_counts(systems, pairs, dataset)

    per_pair_scores = [
        [_scores(counts.sum(axis=0)) for counts in pair_counts]
        for pair_counts in counts_by_pair
    ]
    scores = {
        system.metric_key: _Scores(
            *(
                float(
                    np.mean(
                        [
                            getattr(pair_scores[index], field)
                            for pair_scores in per_pair_scores
                        ]
                    )
                )
                for field in ("mcc", "precision", "recall", "f1")
            )
        )
        for index, system in enumerate(systems)
    }
    matrix = _permutation_p_values(
        counts_by_pair,
        item_ids_by_pair,
        dataset.items,
        analysis["permutations"],
        _seed(analysis["random_seed"], pairs),
    )
    p_values = {
        tuple(sorted((left.metric_key, right.metric_key))): float(matrix[i, j])
        for i, left in enumerate(systems)
        for j, right in enumerate(systems)
        if i < j
    }
    return systems, scores, p_values



def load_dataset(root: Path, wanted: set[str]) -> _Dataset:
    """Adapt validated release files to the original permutation dataset objects."""
    inventory, reference, manifest = load_context(root)
    items = {item_id: _Item(item.parsed_id.language_pair, _item_parts(item_id)[2])
             for item_id, item in inventory.items.items()}
    labels = {key: tuple(j.error_free for j in reference.judgments_for(key))
              for key in reference.labels}
    overrides = load_json(Path(__file__).with_name("base_model_overrides.json"))
    systems = []
    for family, prefix, field, convert in (
        ("submissions", "task3", "task3_pred", lambda value: value),
        ("task1_baselines_converted", "task1-baseline", "task1_pred", _is_clean),
        ("task1_converted", "task1", "task1_pred", _is_clean),
        ("task2_converted", "task2", "task2_pred", None),
    ):
        for entry in manifest["families"][family]:
            metric_key = f"{prefix}:{entry['entry_id']}"
            if metric_key not in wanted:
                continue
            answers = load_answers(root / entry["model_card"])
            rule = convert
            if prefix == "task1":
                from .task1_conversion import inspect_task1_decision
                decision = inspect_task1_decision(int(entry["entry_id"]), root / "submissions/task1")
                if decision.status != "converted":
                    raise DataError(f"{metric_key}: Task 1 conversion no longer permitted")
            if field == "task2_pred":
                # Use the very same validated declared rule as official scoring.
                from .task2_conversion import inspect_conversion_decision
                decision = inspect_conversion_decision(int(entry["entry_id"]), root / "submissions/task2")
                if decision.status != "converted":
                    raise DataError(f"{metric_key}: conversion no longer permitted")
                rule = (lambda score, d=decision: score >= d.threshold if d.inclusive else score > d.threshold)
            covered, predictions = _predictions(iter_jsonl_rows(root / entry["predictions"]),
                                                 field, rule, set(labels))
            override_track = "task3_direct" if prefix == "task3" else "task2_participant"
            model = resolved_base_model(answers, int(entry["entry_id"]) if entry["entry_id"].isdigit() else 0,
                                        {int(k): v for k, v in overrides[override_track].items()})
            systems.append(_System(metric_key, answers["Organization name"].strip(), model,
                                   covered, predictions))
    if {system.metric_key for system in systems} != wanted:
        raise DataError("The frozen permutation system catalog was not fully loaded")
    return _Dataset(items, labels, systems, tuple(sorted(inventory.item_ids_by_pair)))


def iter_jsonl_rows(path: Path):
    """Adapt the strict release JSONL reader to the original row-only iterator API."""
    for _, row in iter_jsonl(path):
        yield row


def main(argv: list[str] | None = None) -> int:
    """Recompute, rather than import, the three paper significance cluster sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/task3"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task3"))
    parser.add_argument("--permutations", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    with Path(__file__).with_name("significance_config.toml").open("rb") as handle:
        analysis = tomllib.load(handle)["analysis"]
    if args.permutations is not None:
        analysis["permutations"] = args.permutations
    if args.seed is not None:
        analysis["random_seed"] = args.seed
    if analysis["permutations"] < 1:
        parser.error("--permutations must be positive")
    with Path(__file__).with_name("significance_systems.csv").open() as handle:
        catalog = {row["metric_key"]: row for row in csv.DictReader(handle)}
    dataset = load_dataset(args.data_dir, set(catalog))
    rows = []
    matrices = {}
    for scope, pairs in (("priority", tuple(analysis["priority_language_pairs"])),
                         ("all_pairs", dataset.all_pairs)):
        print(f"Permuting {scope}: {analysis['permutations']} replicates", flush=True)
        systems, scores, p_values = _analyze(dataset, pairs, analysis)
        matrices[scope] = {"pairs": list(pairs), "seed": _seed(analysis["random_seed"], pairs),
                           "p_values": [{"left": a, "right": b, "p_value": v}
                                        for (a, b), v in sorted(p_values.items())]}
        groups = [("table3", systems)] if scope == "priority" else [("table4", systems)]
        if scope == "priority":
            groups.insert(0, ("table1", [s for s in systems if s.metric_key.startswith(("task3:", "task1-baseline:"))]))
        for table, selected in groups:
            ranked = _ranked_results(selected, scores, p_values, analysis["significance_alpha"])
            for result in ranked:
                rows.append({"table": table, "metric_key": result.metric_key,
                             "short_name": catalog[result.metric_key]["short_name"],
                             "rank": result.rank, "cluster": result.cluster, "mcc": f"{result.mcc:.4f}"})
    output = args.output_dir / "latex"
    write_csv(output / "significance_clusters.csv", rows, list(rows[0]))
    (output / "significance_analysis.json").write_text(json.dumps({
        "analysis": analysis, "resampling_unit": "source segment shared across target languages",
        "test": "two-sided absolute macro-MCC difference, add-one correction, no multiplicity adjustment",
        "clustering": "original WMT greedy clustering", "scopes": matrices,
    }, indent=2, sort_keys=True) + "\n")
    write_run_manifest(args.output_dir, args.data_dir, role="scoring, threshold and permutation analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
