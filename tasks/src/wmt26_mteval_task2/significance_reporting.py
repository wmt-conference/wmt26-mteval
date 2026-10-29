"""Test both tails and form WMT greedy clusters from completed draws."""
from __future__ import annotations
import numpy as np
from .significance import assign_clusters, one_sided_pvalue, two_sided_pvalue

def compare_scope(ids, components, samples, averages=None):
    """Test and cluster one scope, independently ordering its observed Avg Corr.

    ``components`` has evaluator/component axes; each pair's ``samples`` has
    draw/hybrid/component axes. Local rankings must never inherit the aggregate
    ordering: even the higher evaluator in a one-sided test can change.
    """
    components = np.asarray(components)
    if components.shape != (len(ids), 2) or not np.isfinite(components).all():
        raise ValueError("Invalid observed components")
    pairs = {(a, b) for a in range(len(ids)) for b in range(a+1, len(ids))}
    if not pairs or set(samples) != pairs or len(set(ids)) != len(ids):
        raise ValueError("Complete pairwise results and unique evaluator IDs required")
    averages = components.mean(axis=1) if averages is None else np.asarray(averages)
    if averages.shape != (len(ids),) or not np.allclose(averages, components.mean(axis=1), rtol=0, atol=2e-15):
        raise ValueError("Avg Corr differs from the mean of its components")
    order = sorted(range(len(ids)), key=lambda i: (-averages[i], ids[i]))
    positions = {i: j for j, i in enumerate(order)}
    matrix_two = np.ones((len(ids), len(ids)))
    matrix_one = np.ones_like(matrix_two)
    comparisons, draw_counts = [], set()
    for a, b in sorted(pairs):
        values = np.asarray(samples[a, b])
        if values.ndim != 3 or values.shape[1:] != (2, 2) or not np.isfinite(values).all():
            raise ValueError("Invalid complete null statistics")
        draw_counts.add(len(values))
        delta = float(averages[a] - averages[b])
        differences = values[:, 0, :].mean(axis=1) - values[:, 1, :].mean(axis=1)
        p_two, count_two = two_sided_pvalue(differences, delta)
        orientation = 1 if positions[a] < positions[b] else -1
        p_one, count_one = one_sided_pvalue(orientation*differences, orientation*delta)
        reverse_one, _ = one_sided_pvalue(-orientation*differences, -orientation*delta)
        higher, lower = (a, b) if orientation == 1 else (b, a)
        matrix_two[positions[a], positions[b]] = matrix_two[positions[b], positions[a]] = p_two
        matrix_one[positions[higher], positions[lower]] = p_one
        matrix_one[positions[lower], positions[higher]] = reverse_one
        comparisons.append(dict(evaluator_a=ids[a], evaluator_b=ids[b],
            observed_higher=ids[higher], observed_lower=ids[lower], observed_avg_difference=delta,
            p_two_sided=p_two, p_one_sided_higher=p_one, p_one_sided_lower=reverse_one,
            extreme_two_sided=count_two, extreme_one_sided=count_one, draws=len(values)))
    if len(draw_counts) != 1:
        raise ValueError("All pairs must have the same draw count")
    return order, comparisons, matrix_two, matrix_one, assign_clusters(matrix_two), assign_clusters(matrix_one)
