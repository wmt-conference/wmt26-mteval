"""Render the two publication-facing Task 1 ranking tables."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from wmt26_mteval_task1.constants import PRIMARY_LANGUAGE_PAIRS
from wmt26_mteval_task1.markdown_reporting import markdown_table, write_markdown_companion
from wmt26_mteval_task1.models import BatchScoreResult
from wmt26_mteval_task1.reporting_policy import reported_fine_tuned
from wmt26_mteval_task1.rankings import ranking_scopes
from wmt26_mteval_task1.significance import SignificanceResult


def normalize_label(value: str | None) -> str:
    """Normalize human-entered metadata for grouping, not display."""

    if value is None:
        return ""
    return " ".join(value.split()).casefold()


def _clean_display(value: str | None) -> str:
    """Collapse display whitespace while preserving user-facing capitalization."""

    return " ".join(value.split()) if value else ""


def _yes_no_text(value: bool | None) -> str:
    """Format a normalized model-card boolean for a human-facing table."""

    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"

_DESCRIPTION_MODEL_PATTERNS = (
    re.compile(r"\bCohere Command A\+(?!\w)", re.IGNORECASE),
    re.compile(r"\bGemini(?:[- ]\d+(?:\.\d+)?)?(?:[- ]Flash)?\b", re.IGNORECASE),
    re.compile(r"\bDeepSeek(?:[- ]\w+(?:\.\w+)*)*\b", re.IGNORECASE),
    re.compile(r"\bQwen Plus\b", re.IGNORECASE),
    re.compile(
        r"\bQwen\d+(?:\.\d+)?(?:[- ]\d+B(?:-A\d+B)?)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bGemma[- ]\d+(?:[- ]\d+B)?(?:-it)?\b", re.IGNORECASE),
    re.compile(r"\bxCOMET(?:[- ](?:XL|XXL))?\b", re.IGNORECASE),
)

_RELATED_SUBMISSION_PATTERN = re.compile(
    r"\bsame procedure for submission\s+(\d+)\b",
    re.IGNORECASE,
)

_MODEL_CONFIGURATION_SUFFIX = re.compile(
    r"\s*\(\s*reasoning[_ ]effort\s*=\s*[^)]+\)\s*$",
    re.IGNORECASE,
)


def _canonical_model_component(value: str) -> str:
    """Return one model identity in a consistent publication-facing style."""

    cleaned = _clean_display(value)
    model_only = _MODEL_CONFIGURATION_SUFFIX.sub("", cleaned)

    match = re.fullmatch(
        r"gemini[-_\s]+(\d+(?:\.\d+)?)[-_\s]+flash",
        model_only,
        re.IGNORECASE,
    )
    if match:
        return f"Gemini {match.group(1)} Flash"

    match = re.fullmatch(
        r"deepseek[-_\s]+(v?\d+(?:\.\d+)?)[-_\s]+flash",
        model_only,
        re.IGNORECASE,
    )
    if match:
        return f"DeepSeek {match.group(1).upper()} Flash"

    if re.fullmatch(r"qwen[-_\s]+plus", model_only, re.IGNORECASE):
        return "Qwen Plus"

    match = re.fullmatch(
        r"gpt[-_\s]*(\d+(?:\.\d+)*)"
        r"(?:[-_\s]+(mini|terra|sol|luna))?"
        r"(?:[-_](\d{4}-\d{2}-\d{2}))?",
        model_only,
        re.IGNORECASE,
    )
    if match:
        version, variant, snapshot = match.groups()
        canonical = f"GPT-{version}"
        if variant:
            canonical += f" {variant.title()}"
        if snapshot:
            canonical += f" ({snapshot})"
        return canonical

    match = re.fullmatch(
        r"qwen\s*(\d+(?:\.\d+)?)(?:[-_\s]+(\d+b(?:[-_]a\d+b)?))?",
        model_only,
        re.IGNORECASE,
    )
    if match:
        version, size = match.groups()
        return f"Qwen{version}" + (f"-{size.upper()}" if size else "")

    match = re.fullmatch(
        r"(?:google/)?gemma[-_\s]*(\d+(?:\.\d+)?)(?:[-_\s]+(\d+b))?"
        r"(?:[-_\s]+(it))?",
        model_only,
        re.IGNORECASE,
    )
    if match:
        version, size, instruction_tuned = match.groups()
        canonical = f"Gemma {version}" + (f" {size.upper()}" if size else "")
        if instruction_tuned:
            canonical += " IT"
        return canonical

    match = re.fullmatch(r"xcomet[-_\s]*(xl|xxl)", model_only, re.IGNORECASE)
    if match:
        return f"xCOMET-{match.group(1).upper()}"

    if re.fullmatch(r"cohere[-_\s]+command[-_\s]+a\+", model_only, re.IGNORECASE):
        return "Cohere Command A+"
    return model_only


def _canonical_model_names(value: str) -> str:
    """Canonicalize and deduplicate a semicolon-delimited model-name list."""

    canonical_names: list[str] = []
    seen: set[str] = set()
    for component in value.split(";"):
        canonical = _canonical_model_component(component)
        normalized = normalize_label(canonical)
        if canonical and normalized not in seen:
            seen.add(normalized)
            canonical_names.append(canonical)
    return "; ".join(canonical_names)


def _linked_model_name(system_link: str | None) -> str:
    """Extract an explicit Hugging Face repository ID from a system link."""

    if not system_link:
        return ""
    try:
        parsed = urlsplit(system_link)
    except ValueError:
        return ""
    if (parsed.hostname or "").casefold() not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        return ""
    components = [component for component in parsed.path.split("/") if component]
    return "/".join(components[:2]) if len(components) >= 2 else ""


def _description_model_names(description: str | None) -> str:
    """Return explicit model-family mentions found in a system description."""

    if not description:
        return ""
    matches: list[str] = []
    seen: set[str] = set()
    for pattern in _DESCRIPTION_MODEL_PATTERNS:
        for match in pattern.finditer(description):
            value = _canonical_model_component(match.group(0))
            normalized = value.casefold()
            if normalized not in seen:
                seen.add(normalized)
                matches.append(value)
    return "; ".join(matches)


def _underlying_model(metadata: Any) -> str:
    """Choose the clearest model identity explicitly supported by a model card."""

    model_names = (
        _clean_display(metadata.closed_source_model_name)
        or _linked_model_name(metadata.system_link)
        or _description_model_names(metadata.submission_description)
    )
    return _canonical_model_names(model_names) or "Not specified"


def _underlying_models(scores: Sequence[Any]) -> dict[tuple[str, str], str]:
    """Resolve direct and explicit cross-submission model identities for a batch."""

    models = {
        _descriptor_entry_key(score.descriptor): _underlying_model(score.model_card)
        for score in scores
    }
    submission_models = {
        score.descriptor.submission_id: models[_descriptor_entry_key(score.descriptor)]
        for score in scores
        if not score.descriptor.is_baseline
    }
    for score in scores:
        key = _descriptor_entry_key(score.descriptor)
        if (
            models[key] != "Not specified"
            or not score.model_card.submission_description
        ):
            continue
        match = _RELATED_SUBMISSION_PATTERN.search(
            score.model_card.submission_description
        )
        if match is None:
            continue
        referenced_id = match.group(1)
        referenced_model = submission_models.get(referenced_id)
        if referenced_model and referenced_model != "Not specified":
            models[key] = referenced_model
    return models


def _other_model_card_info(metadata: Any) -> str:
    """Summarize useful structured model-card details not shown elsewhere."""

    details = []
    if metadata.model_size:
        details.append(f"Size: {_clean_display(metadata.model_size)}")

    open_weight = metadata.evaluator_characteristics.get("open_weight")
    if open_weight is True:
        details.append("Open weights")
    elif open_weight is False:
        details.append("Closed weights")
    else:
        details.append("Weight availability unknown")

    open_source = metadata.evaluator_characteristics.get("open_source")
    if open_source is True:
        details.append("Open-source system")
    elif open_source is False:
        details.append("Closed-source system")
    else:
        details.append("System source availability unknown")

    if metadata.api_provider:
        details.append(f"API: {_clean_display(metadata.api_provider)}")
    return "; ".join(details)


def _model_card_report_fields(
    descriptor: Any,
    metadata: Any,
    *,
    underlying_model: str | None = None,
) -> dict[str, str]:
    """Build the five model-card columns shared by aggregate report formats."""

    characteristics = metadata.evaluator_characteristics
    return {
        "underlying_model": underlying_model or _underlying_model(metadata),
        "reference_based": _yes_no_text(characteristics.get("reference_use")),
        "fine_tuned": _yes_no_text(
            reported_fine_tuned(descriptor, characteristics.get("supervised"))
        ),
        "is_ensemble": _yes_no_text(characteristics.get("ensemble")),
        "other": _other_model_card_info(metadata),
    }


def _score_entry_fields(score: Any) -> dict[str, Any]:
    """Return stable identity fields for one scored entry report row."""

    descriptor = score.descriptor
    return {
        "entry_type": "baseline" if descriptor.is_baseline else "submission",
        "entry": descriptor.report_label,
        "submission_id": descriptor.submission_id,
        "baseline_name": descriptor.baseline_name if descriptor.is_baseline else "",
    }


def _descriptor_entry_key(descriptor: Any) -> tuple[str, str]:
    """Return a stable mixed submission/baseline key for report lookups."""

    if descriptor.is_baseline:
        return "baseline", descriptor.baseline_name
    return "submission", descriptor.submission_id


def _entry_display_names(scores: Sequence[Any]) -> dict[tuple[str, str], str]:
    """Return public display names, rejecting ambiguous submission names."""

    grouped: dict[str, list[Any]] = defaultdict(list)
    base_names: dict[tuple[str, str], str] = {}
    for score in scores:
        key = _descriptor_entry_key(score.descriptor)
        base = _clean_display(score.model_card.submission_name)
        base = base or score.descriptor.report_label
        if score.descriptor.is_baseline:
            base = re.sub(r"\bNo Reasoning\b", "No Thinking", base, flags=re.IGNORECASE)
            base = re.sub(r"\bReasoning\b", "Thinking", base, flags=re.IGNORECASE)
        base_names[key] = base
        grouped[normalize_label(base)].append(score)

    display_names = dict(base_names)
    for collisions in grouped.values():
        if len(collisions) < 2:
            continue
        reference_modes = {
            score.model_card.evaluator_characteristics.get("reference_use")
            for score in collisions
        }
        if (
            all(score.descriptor.is_baseline for score in collisions)
            and reference_modes == {False, True}
            and len(collisions) == 2
        ):
            # Paired organizer baselines intentionally share a concise model
            # name; the reference-use field (and LaTeX star) distinguishes them.
            continue
        name = base_names[_descriptor_entry_key(collisions[0].descriptor)]
        raise ValueError(f"Submission name {name!r} is ambiguous across entries")
    return display_names


def _aggregate_score_row(
    score: Any,
    *,
    priority_only: bool,
    underlying_model: str,
    directions: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Build one aggregate score row, or return none when no direction qualifies."""

    direction_scores = tuple(
        direction
        for direction in score.direction_scores
        if (direction.language_pair in directions if directions is not None
            else not priority_only or direction.language_pair in PRIMARY_LANGUAGE_PAIRS)
    )
    if not direction_scores:
        return None

    return {
        **_score_entry_fields(score),
        "organization_name": score.model_card.organization_name or "",
        "submission_name": score.model_card.submission_name or "",
        **_model_card_report_fields(
            score.descriptor,
            score.model_card,
            underlying_model=underlying_model,
        ),
        "scored_direction_count": len(direction_scores),
        "scored_directions": ";".join(
            sorted(direction.language_pair for direction in direction_scores)
        ),
        "macro_precision": sum(direction.precision for direction in direction_scores)
        / len(direction_scores),
        "macro_recall": sum(direction.recall for direction in direction_scores)
        / len(direction_scores),
        "macro_f1": sum(direction.f1 for direction in direction_scores)
        / len(direction_scores),
    }


def _ranked_aggregate_rows(
    result: BatchScoreResult,
    *,
    priority_only: bool,
    underlying_models: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """Return coverage-gated aggregate rows sorted and assigned ranks."""

    scope_name = "priority" if priority_only else "all_directions"
    scope = next(scope for scope in ranking_scopes(result) if scope.name == scope_name)
    rows = []
    for rank, score in enumerate(scope.entries, 1):
        row = _aggregate_score_row(
            score, priority_only=priority_only, directions=scope.directions,
            underlying_model=underlying_models[_descriptor_entry_key(score.descriptor)],
        )
        if row is not None:
            row["rank"] = rank
            rows.append(row)
    return rows


def _per_direction_ranking_rows(result: BatchScoreResult) -> list[dict[str, Any]]:
    """Return the shared independently ordered per-direction rankings."""
    rows = []
    for scope in ranking_scopes(result):
        if not scope.name.startswith("direction:"):
            continue
        language_pair = scope.directions[0]
        for rank, score in enumerate(scope.entries, 1):
            direction = next(
                row for row in score.direction_scores if row.language_pair == language_pair
            )
            rows.append({
                **_score_entry_fields(score), "rank": rank,
                "organization_name": score.model_card.organization_name or "",
                "submission_name": score.model_card.submission_name or "",
                "language_pair": language_pair,
                "precision": direction.precision, "recall": direction.recall,
                "f1": direction.f1, "aligned_assessments": direction.aligned_assessments,
            })
    return rows


def _latex_escape(value: Any) -> str:
    """Escape one plain-text value for use in a LaTeX table cell."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _latex_metric(value: float, *, signed: bool = False) -> str:
    """Format one unit-scale metric as a two-decimal percentage in math mode."""

    specification = "+.2f" if signed else ".2f"
    return f"${format(100 * float(value), specification)}$"


def _latex_publication_submission_name(descriptor: Any, display_name: str) -> str:
    """Return the concise publication name for one scored entry."""

    name = _clean_display(display_name)
    name = re.sub(r"\bNo Reasoning\b", "No Thinking", name, flags=re.IGNORECASE)
    name = re.sub(r"\bReasoning\b", "Thinking", name, flags=re.IGNORECASE)
    if not descriptor.is_baseline:
        return name

    name = re.sub(
        r"\s+Reference-(?:based|free)(?=\s|$)",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"\s*-\s*Baseline$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+Baseline$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*-\s*No Thinking$", "", name, flags=re.IGNORECASE)
    name = re.sub(
        r"\s*-\s*(Medium|Xhigh) Thinking$",
        lambda match: f" - {match.group(1).lower()} thinking",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(
        r"\s*-\s*Thinking$",
        " - thinking",
        name,
        flags=re.IGNORECASE,
    )
    return _clean_display(name)


def _latex_submission_cell(score: Any, display_name: str) -> tuple[str, bool, bool]:
    """Render a publication name and return its star/dagger legend flags."""

    name = _latex_escape(
        _latex_publication_submission_name(score.descriptor, display_name)
    )
    characteristics = score.model_card.evaluator_characteristics
    reference_free = characteristics.get("reference_use") is False
    fine_tuned = (
        reported_fine_tuned(score.descriptor, characteristics.get("supervised"))
        is True
    )
    if reference_free:
        name += r"\textsuperscript{*}"
    if fine_tuned:
        name += r"\textsuperscript{\textdagger}"
    return name, reference_free, fine_tuned


def _latex_publication_underlying_model(underlying_model: str) -> str:
    """Return a compact LaTeX-only model identity for publication tables."""

    components = []
    for component in underlying_model.split(";"):
        cleaned = _clean_display(component)
        cleaned = re.sub(r"^Qwen/", "", cleaned, flags=re.IGNORECASE)
        match = re.fullmatch(
            r"(?:Unbabel/)?XCOMET[- ](XL|XXL)", cleaned, flags=re.IGNORECASE
        )
        if match:
            cleaned = f"xCOMET-{match.group(1).upper()}"
        if cleaned.casefold() == "cohere command a+":
            cleaned = "Command A+"
        components.append(cleaned)
    return "; ".join(component for component in components if component)


def _latex_style_notes(
    *, has_reference_free: bool, has_fine_tuned: bool, has_baseline: bool
) -> list[str]:
    """Return the shared publication legend sentences for a LaTeX caption."""

    notes: list[str] = []
    if has_reference_free:
        notes.append(r"\textsuperscript{*} marks reference-free submissions")
    if has_fine_tuned:
        notes.append(r"\textsuperscript{\textdagger} marks fine-tuned systems")
    if has_baseline:
        notes.append("Light-grey rows indicate organizer baselines")
    return notes


def _ensemble_components(metadata: Any, underlying_model: str) -> str:
    """Return distinct model identities available for an ensemble caption note."""

    candidates: list[str] = []
    if underlying_model != "Not specified":
        candidates.extend(underlying_model.split(";"))
    candidates.extend(
        _description_model_names(metadata.submission_description).split(";")
    )
    components: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _canonical_model_component(candidate)
        normalized = cleaned.casefold()
        if cleaned and normalized not in seen:
            seen.add(normalized)
            components.append(cleaned)
    components = [
        component
        for component in components
        if not (
            len(component.split()) == 1
            and any(
                other.casefold().startswith(component.casefold() + " ")
                for other in components
                if other != component
            )
        )
    ]
    return "; ".join(components) or "not specified in the model card"


def _latex_significance_note(manifest: Mapping[str, Any]) -> str:
    """Render the aggregate-caption methods sentence from the test's provenance."""
    alpha = float(manifest["alpha"])
    alternative = manifest.get("alternative", "one-sided")
    draws = int(manifest["permutations"])
    return (
        r"Similar to previous years \cite{freitag-etal-2023-results, freitag-etal-2024-llms, lavie-etal-2025-findings}, "
        r"statistical significance uses the PERM-BOTH hypothesis test introduced by "
        r"\citet{10.1162/tacl_a_00417}, with clusters computed greedily "
        rf"($p\leq{alpha:g}$, unadjusted, {alternative}, {draws} draws)"
    )


def _latex_ranking_table(
    rows: Sequence[Mapping[str, Any]],
    display_names: Mapping[tuple[str, str], str],
    scores_by_entry: Mapping[tuple[str, str], Any],
    underlying_models: Mapping[tuple[str, str], str],
    *,
    caption: str,
    label: str,
    precision_field: str,
    recall_field: str,
    f1_field: str,
    caption_is_latex: bool = False,
    include_significance_note: bool = False,
    significance: SignificanceResult | None = None,
) -> str:
    """Render a LaTeX ranking, optionally citing significance for aggregate tables."""

    body_lines: list[str] = []
    ensemble_notes: list[tuple[int, str, str]] = []
    has_reference_free = False
    has_fine_tuned = False
    has_baseline = False
    if not rows:
        body_lines.append(r"\multicolumn{7}{c}{No eligible entries.} \\")
    for row in rows:
        key = (
            str(row["entry_type"]),
            str(row["baseline_name"] or row["submission_id"]),
        )
        score = scores_by_entry[key]
        metadata = score.model_card
        name, reference_free, fine_tuned = _latex_submission_cell(
            score, display_names[key]
        )
        has_reference_free = has_reference_free or reference_free
        has_fine_tuned = has_fine_tuned or fine_tuned

        underlying_model = underlying_models[key]
        if metadata.evaluator_characteristics.get("ensemble") is True:
            ensemble_number = len(ensemble_notes) + 1
            ensemble_notes.append(
                (
                    ensemble_number,
                    display_names[key],
                    _ensemble_components(metadata, underlying_model),
                )
            )
            model_cell = rf"Ensemble\textsuperscript{{E{ensemble_number}}}"
        else:
            model_cell = _latex_escape(
                _latex_publication_underlying_model(underlying_model)
            )

        prefix = ""
        if row["entry_type"] == "baseline":
            prefix = r"\rowcolor{black!15} "
            has_baseline = True
        body_lines.append(
            "{}${}$ & {} & {} & {} & {} & {} & {} \\\\".format(
                prefix,
                row["rank"],
                (f"${row['significance_cluster']}$" if row.get("significance_cluster", "") != "" else ""),
                name,
                model_cell,
                _latex_metric(float(row[precision_field])),
                _latex_metric(float(row[recall_field])),
                _latex_metric(float(row[f1_field])),
            )
        )

    caption_notes = _latex_style_notes(
        has_reference_free=has_reference_free,
        has_fine_tuned=has_fine_tuned,
        has_baseline=has_baseline,
    )
    if ensemble_notes:
        ensemble_summary = "; ".join(
            "E{} [{}]".format(
                number,
                _latex_escape(components),
            )
            for number, _, components in ensemble_notes
        )
        caption_notes.append(
            "The ensemble components listed in the model card are "
            + ensemble_summary
        )
    if include_significance_note and significance is not None:
        caption_notes.append(_latex_significance_note(significance.manifest))

    full_caption = (caption if caption_is_latex else _latex_escape(caption)).rstrip(".")
    if caption_notes:
        full_caption += ". " + ". ".join(caption_notes)
    full_caption += "."

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rrllrrr}",
        r"\toprule",
        r"\multicolumn{1}{c}{Rank} & \multicolumn{1}{c}{Cluster} & Auto-evaluator & Underlying model & "
        r"\multicolumn{1}{c}{P} & \multicolumn{1}{c}{R} & "
        r"\multicolumn{1}{c}{F1} \\",
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{full_caption}}}",
        rf"\label{{{_latex_escape(label)}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)

_DIRECTION_PUBLICATION_NAMES = {
    "ces_Latn-deu_Latn": ("Czech", "German", "ces", "deu"),
    "ces_Latn-ukr_Cyrl": ("Czech", "Ukrainian", "ces", "ukr"),
    "ces_Latn-vie_Latn": ("Czech", "Vietnamese", "ces", "vie"),
    "eng_Latn-arz_Arab": ("English", "Egyptian Arabic", "eng", "arz"),
    "eng_Latn-bel_Cyrl": ("English", "Belarusian", "eng", "bel"),
    "eng_Latn-ces_Latn": ("English", "Czech", "eng", "ces"),
    "eng_Latn-deu_Latn": ("English", "German", "eng", "deu"),
    "eng_Latn-ekk_Latn": ("English", "Estonian", "eng", "ekk"),
    "eng_Latn-hye_Armn": ("English", "Armenian", "eng", "hye"),
    "eng_Latn-ind_Latn": ("English", "Indonesian", "eng", "ind"),
    "eng_Latn-isl_Latn": ("English", "Icelandic", "eng", "isl"),
    "eng_Latn-kaz_Cyrl": ("English", "Kazakh", "eng", "kaz"),
    "eng_Latn-lij_Latn": ("English", "Ligurian", "eng", "lij"),
    "eng_Latn-lld_Latn": ("English", "Ladin", "eng", "lld"),
    "eng_Latn-rus_Cyrl": ("English", "Russian", "eng", "rus"),
    "eng_Latn-sme_Latn": ("English", "Northern Sami", "eng", "sme"),
    "eng_Latn-tha_Thai": ("English", "Thai", "eng", "tha"),
    "eng_Latn-ukr_Cyrl": ("English", "Ukrainian", "eng", "ukr"),
    "eng_Latn-zho_Hans": ("English", "Chinese (Simplified)", "eng", "zho-Hans"),
    "eng_Latn-zho_Hant_TW": ("English", "Chinese (Traditional, Taiwan)", "eng", "zho-Hant"),
    "zho_Hans-jpn_Jpan": ("Chinese (Simplified)", "Japanese", "zho", "jpn"),
}


def _direction_publication_caption(language_pair: str) -> str:
    """Return a long-form direction caption for a per-direction ranking table."""

    source, target, _, _ = _DIRECTION_PUBLICATION_NAMES[language_pair]
    return f"Task 1 ranking for the {source}$\\rightarrow${target} translation direction"


def _direction_publication_label(language_pair: str) -> str:
    """Return the concise stable LaTeX label for one direction's ranking table."""

    _, _, source, target = _DIRECTION_PUBLICATION_NAMES[language_pair]
    return f"tab:task1-{source}-{target}-results"


def _markdown_ranking_table(
    rows: Sequence[Mapping[str, Any]],
    display_names: Mapping[tuple[str, str], str],
    scores_by_entry: Mapping[tuple[str, str], Any],
    underlying_models: Mapping[tuple[str, str], str],
    *,
    precision_field: str,
    recall_field: str,
    f1_field: str,
) -> str:
    """Render the same ordered ranking values as a readable Markdown table."""

    output = []
    for row in rows:
        key = (str(row["entry_type"]), str(row["baseline_name"] or row["submission_id"]))
        score = scores_by_entry[key]
        characteristics = score.model_card.evaluator_characteristics
        reference_use = characteristics.get("reference_use")
        fine_tuned = reported_fine_tuned(
            score.descriptor, characteristics.get("supervised")
        )
        output.append((
            row["rank"],
            row.get("significance_cluster", ""),
            _latex_publication_submission_name(score.descriptor, display_names[key]),
            _latex_publication_underlying_model(underlying_models[key]),
            f"{100 * float(row[precision_field]):.2f}",
            f"{100 * float(row[recall_field]):.2f}",
            f"{100 * float(row[f1_field]):.2f}",
            "Baseline" if score.descriptor.is_baseline else "Submission",
            "Yes" if reference_use is True else "No" if reference_use is False else "Unknown",
            "Yes" if fine_tuned is True else "No" if fine_tuned is False else "Unknown",
        ))
    return markdown_table(
        ("Rank", "Cluster", "Auto-evaluator", "Underlying model", "P",
         "R", "F1", "Entry type", "Uses reference", "Fine-tuned"),
        output,
    )


def write_paper_score_reports(
    output_dir: Path,
    result: BatchScoreResult,
    significance: SignificanceResult | None,
) -> tuple[Path, Path]:
    """Write the priority and 21-direction paper tables from scored entries.

    Inputs are a complete Task 1 scoring result and optional paired significance
    tests. The returned paths are the two publication-facing TeX files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    underlying_models = _underlying_models(result.scores)
    priority_rows = _ranked_aggregate_rows(
        result, priority_only=True, underlying_models=underlying_models
    )
    direction_rows = _per_direction_ranking_rows(result)
    for name, rows in (("priority", priority_rows), ("direction", direction_rows)):
        for row in rows:
            scope = name if name == "priority" else "direction:" + str(row["language_pair"])
            key = (str(row["entry_type"]), str(row["baseline_name"] or row["submission_id"]))
            row["significance_cluster"] = (
                significance.clusters[scope][key] if significance else ""
            )
    display_names = _entry_display_names(result.scores)
    scores_by_entry = {
        _descriptor_entry_key(score.descriptor): score for score in result.scores
    }
    priority_path = output_dir / "priority_ranking.tex"
    priority_path.write_text(
        _latex_ranking_table(
            priority_rows,
            display_names,
            scores_by_entry,
            underlying_models,
            caption="Task 1 ranking for the set of eight priority translation directions",
            label="tab:task1-priority-results",
            precision_field="macro_precision",
            recall_field="macro_recall",
            f1_field="macro_f1",
            include_significance_note=True,
            significance=significance,
        ) + "\n",
        encoding="utf-8",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in direction_rows:
        grouped[str(row["language_pair"])].append(row)
    tables = [
        _latex_ranking_table(
            grouped[direction],
            display_names,
            scores_by_entry,
            underlying_models,
            caption=_direction_publication_caption(direction),
            label=_direction_publication_label(direction),
            precision_field="precision",
            recall_field="recall",
            f1_field="f1",
            caption_is_latex=True,
        )
        for direction in sorted(grouped)
    ]
    direction_path = output_dir / "per_direction_rankings.tex"
    direction_path.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    write_markdown_companion(
        priority_path,
        "Task 1 ranking: eight priority directions",
        _markdown_ranking_table(
            priority_rows, display_names, scores_by_entry, underlying_models,
            precision_field="macro_precision", recall_field="macro_recall",
            f1_field="macro_f1",
        ) + "\n\nBlank clusters mean significance was skipped.",
    )
    direction_sections = []
    for direction in sorted(grouped):
        source, target, _, _ = _DIRECTION_PUBLICATION_NAMES[direction]
        direction_sections.append(
            f"## {source} → {target} ({direction})\n\n"
            + _markdown_ranking_table(
                grouped[direction], display_names, scores_by_entry,
                underlying_models, precision_field="precision",
                recall_field="recall", f1_field="f1",
            )
        )
    write_markdown_companion(
        direction_path,
        "Task 1 rankings by translation direction",
        "\n\n".join(direction_sections)
        + "\n\nBlank clusters mean significance was skipped.",
    )
    return priority_path, direction_path
