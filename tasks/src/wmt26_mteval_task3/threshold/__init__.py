"""Reusable Task 2 threshold-sensitivity analysis."""

from .core import (
    AnalysisResult,
    CurvePoint,
    LanguageAnalysis,
    Observation,
    analyze_system,
    downsample_curve,
    load_official_metrics,
    mcc_from_counts,
    optimize_per_language,
    optimize_threshold,
    pooled_metrics_from_official_rows,
    verify_declared_result,
)

__all__ = [
    "AnalysisResult",
    "CurvePoint",
    "LanguageAnalysis",
    "Observation",
    "analyze_system",
    "downsample_curve",
    "load_official_metrics",
    "mcc_from_counts",
    "optimize_per_language",
    "optimize_threshold",
    "pooled_metrics_from_official_rows",
    "verify_declared_result",
]
