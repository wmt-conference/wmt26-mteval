"""Render the Task 3 result CSVs under ``results/`` as LaTeX tables.

Reads the per-language metric dumps produced by ``run_task3_analysis.py`` and
the four conversion analyses, then writes paper-ready tables into
``results/latex/``:

* ``table1_official_ranking.tex`` -- official ranking over the eight priority
  language pairs (direct Task 3 submissions + Task 1 baselines only).
* ``table2_all_pairs.tex`` -- every system covering all language pairs, across
  all tracks, with the source track marked.
* ``per_language/<pair>.tex`` -- one merged breakdown table per language pair.

MCC is the primary metric; precision, recall and F1 use error-free as the
positive class.  Averages are macro averages over language pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path.cwd()
RESULTS_DIR = REPO_ROOT / "outputs" / "task3"
OUTPUT_DIR = RESULTS_DIR / "latex"

# (directory, short track label, long track label, is_baseline)
#
# ``task2_baselines_converted`` is deliberately excluded from every table: the
# Task 2 -> Task 3 conversion of the organizer baselines has known issues, so
# those numbers are not comparable to the rest.  Re-add the entry below to
# bring them back.
SOURCES = (
    ("submissions", "T3", "Task 3 submission", False),
    ("task1_converted", "T1", "Task 1 conversion", False),
    ("task2_converted", "T2", "Task 2 conversion", False),
    ("task1_baselines_converted", "T1-B", "Task 1 baseline", True),
)

OFFICIAL_SOURCES = ("submissions", "task1_baselines_converted")

# Organizations dropped from every table and every CSV side-output, for holds
# that are not yet reflected in results/*/metrics_by_language.csv.  Empty at
# present: withdrawn teams are now filtered upstream.
EXCLUDED_ORGANIZATIONS: tuple[str, ...] = ()

# The eight priority language pairs of the WMT26 Task 3 shared task.
PRIORITY_PAIRS = (
    "ces_Latn---deu_Latn",
    "ces_Latn---vie_Latn",
    "eng_Latn---arz_Arab",
    "eng_Latn---ekk_Latn",
    "eng_Latn---ind_Latn",
    "eng_Latn---isl_Latn",
    "eng_Latn---sme_Latn",
    "zho_Hans---jpn_Jpan",
)

# Model cards come only from the prepared release inputs.
MODEL_CARD_DIR = Path("data/task3/submissions/task3")
DESCRIPTION_QUESTION = "Submission description (a short paragraph)"

# Draft, not part of the numbered tables: the six pairs PTT-MULTITAN-v2 entered,
# which happen to miss the priority set entirely.  Set to () to drop the table.
DRAFT_PAIRS = (
    "eng_Latn---bel_Cyrl",
    "eng_Latn---deu_Latn",
    "eng_Latn---rus_Cyrl",
    "eng_Latn---ukr_Cyrl",
    "eng_Latn---zho_Hans",
    "eng_Latn---zho_Hant_TW",
)

# Teams that entered under more than one organization string.  The systems are
# described together under the mapped-to name, and the differing original string
# is named on the system it belongs to.
ORGANIZATION_MERGES = {
    "Phrase a.s. + CUNI": "Phrase a.s.",
}

# Copy-edits applied to the quoted descriptions before LaTeX escaping, as
# (submission id or None for all, before, after).  Keep this list short and
# auditable: everything not listed here is reproduced verbatim from the
# participant's model card.
DESCRIPTION_EDITS = (
    # Shouted in the original; set as running text.
    ("876247", "CHARACTER-LEVEL MAJORITY VOTE", "character-level majority vote"),
    # PTT-MULTITAN-v2 ends with a bare list of pair codes; recast as a sentence
    # in the short pair names used throughout the paper.
    (
        None,
        "Language pairs: eng_Latn-bel_Cyrl, eng_Latn-deu_Latn, "
        "eng_Latn-rus_Cyrl, eng_Latn-ukr_Cyrl, eng_Latn-zho_Hans, "
        "eng_Latn-zho_Hant_TW",
        "The system covers six language pairs: eng-bel, eng-deu, eng-rus, "
        "eng-ukr, eng-zho_CN and eng-zho_TW.",
    ),
)

# Typesetting applied after escaping, so the replacements may contain LaTeX.
DESCRIPTION_FORMATTING = (
    ("[0,100]", "$[0,100]$"),
    ("(Wmt26SpanTaskV2Directive)", "(\\texttt{Wmt26SpanTaskV2Directive})"),
)

# Language-pair codes as used in the paper's language-pair table.  Chinese
# targets carry a region suffix (zho_CN, zho_TW); every other side is the bare
# three-letter code with the script tag dropped.
PAIR_CODE = {
    "ces_Latn---deu_Latn": "ces-deu",
    "ces_Latn---ukr_Cyrl": "ces-ukr",
    "ces_Latn---vie_Latn": "ces-vie",
    "eng_Latn---arz_Arab": "eng-arz",
    "eng_Latn---bel_Cyrl": "eng-bel",
    "eng_Latn---ces_Latn": "eng-ces",
    "eng_Latn---deu_Latn": "eng-deu",
    "eng_Latn---ekk_Latn": "eng-ekk",
    "eng_Latn---hye_Armn": "eng-hye",
    "eng_Latn---ind_Latn": "eng-ind",
    "eng_Latn---isl_Latn": "eng-isl",
    "eng_Latn---kaz_Cyrl": "eng-kaz",
    "eng_Latn---lij_Latn": "eng-lij",
    "eng_Latn---lld_Latn": "eng-lld",
    "eng_Latn---rus_Cyrl": "eng-rus",
    "eng_Latn---sme_Latn": "eng-sme",
    "eng_Latn---tha_Thai": "eng-tha",
    "eng_Latn---ukr_Cyrl": "eng-ukr",
    "eng_Latn---zho_Hans": "eng-zho_CN",
    "eng_Latn---zho_Hant_TW": "eng-zho_TW",
    "zho_Hans---jpn_Jpan": "zho-jpn",
}


def short_pair(pair: str) -> str:
    """The paper's code for a language pair, e.g. ``eng-zho_CN``."""
    return PAIR_CODE[pair]


def pair_slug(pair: str) -> str:
    """Filename- and label-safe form of the code, e.g. ``eng-zho-CN``."""
    return short_pair(pair).replace("_", "-")


def escape(text: str) -> str:
    """Escape LaTeX specials.  Accented characters pass through as UTF-8."""
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
    return "".join(replacements.get(char, char) for char in text)


def breakable(text: str) -> str:
    """Allow line breaks inside long model identifiers."""
    return text.replace("/", "/\\allowbreak{}").replace("; ", ";\\allowbreak{} ")


def fmt(value: Any, places: int = 4) -> str:
    """Compute fmt from the supplied analysis state and return its representation."""
    if value is None or value == "":
        return "--"
    return f"{float(value):.{places}f}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_reference_flags() -> dict[tuple[str, str], str]:
    """Read ``uses_reference`` per system from each track's conversion info.

    Task 3 is a reference-free task, so a conversion whose source system used
    the reference translations does not belong in these tables.  A blank answer
    is left in: it means unanswered, not confirmed reference-based.
    """
    flags = {}
    for directory, *_ in SOURCES:
        path = RESULTS_DIR / directory / "conversion_info.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                flags[(directory, row["display_name"])] = row.get("uses_reference", "")
    return flags


def load_rows() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str, str], dict[str, float]]]:
    """Return (systems keyed by (track, name), metrics keyed by (track, name, pair))."""
    systems: dict[tuple[str, str], dict[str, Any]] = {}
    metrics: dict[tuple[str, str, str], dict[str, float]] = {}
    reference_flags = load_reference_flags()
    dropped_for_reference: set[tuple[str, str]] = set()

    for directory, track, track_long, is_baseline in SOURCES:
        path = RESULTS_DIR / directory / "metrics_by_language.csv"
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("organization_name") in EXCLUDED_ORGANIZATIONS:
                    continue
                if reference_flags.get((directory, row["display_name"])) == "True":
                    dropped_for_reference.add((track, row["display_name"]))
                    continue
                name = row["display_name"]
                key = (track, name)
                entry_id = row.get("submission_id") or row.get("entry_key", "")
                systems.setdefault(
                    key,
                    {
                        "track": track,
                        "track_long": track_long,
                        "directory": directory,
                        "is_baseline": is_baseline,
                        "display_name": name,
                        "organization_name": row.get("organization_name", ""),
                        "submission_name": row.get("submission_name", ""),
                        "base_model": row.get("base_model", ""),
                        "entry_id": entry_id,
                        "threshold": row.get("threshold", ""),
                        "pairs": set(),
                    },
                )
                pair = row["language_pair"]
                metric_key = (track, name, pair)
                if metric_key in metrics:
                    raise SystemExit(
                        f"duplicate metric row for {metric_key} in {path}"
                    )
                systems[key]["pairs"].add(pair)
                metrics[metric_key] = {
                    field: float(row[field])
                    for field in ("mcc", "precision", "recall", "f1")
                }
    for track, name in sorted(dropped_for_reference):
        print(f"excluded (uses reference translations): {track} {name}")
    return systems, metrics


def metric_key(system: dict[str, Any]) -> str:
    """The key the significance pipeline uses, e.g. ``task3:870955``."""
    return f"{TRACK_TO_METRIC_PREFIX[system['track']]}:{system['entry_id']}"


def load_clusters(table: str, rows: list[dict[str, Any]]) -> dict[str, int] | None:
    """Significance clusters for ``table``, or None when unavailable or stale."""
    if not CLUSTER_FILE.exists():
        return None
    recorded = {}
    with CLUSTER_FILE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["table"] == table:
                recorded[row["metric_key"]] = (int(row["cluster"]), float(row["mcc"]))
    if not recorded:
        return None

    clusters = {}
    stale = []
    for row in rows:
        key = metric_key(row["system"])
        if key not in recorded:
            stale.append(f"{key} missing from {CLUSTER_FILE.name}")
            continue
        cluster, mcc = recorded[key]
        if abs(mcc - row["mcc"]) > 5e-5:
            stale.append(f"{key} MCC {row['mcc']:.4f} != recorded {mcc:.4f}")
        clusters[key] = cluster
    if stale or len(recorded) != len(rows):
        print(f"warning: {CLUSTER_FILE.name} is stale for {table}; omitting the "
              "cluster column. Re-run the significance pipeline.")
        for message in stale[:5]:
            print(f"    {message}")
        return None
    return clusters


def load_summary() -> dict[str, dict[str, Any]]:
    """Compute load summary from the supplied analysis state and return its representation."""
    path = RESULTS_DIR / "submissions" / "human_reference_summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["language_pair"]: row for row in csv.DictReader(handle)}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def macro_average(
    system: dict[str, Any],
    metrics: dict[tuple[str, str, str], dict[str, float]],
    pairs: Iterable[str],
) -> dict[str, float]:
    """Compute macro average from the supplied analysis state and return its representation."""
    pairs = list(pairs)
    out = {}
    for field in ("mcc", "precision", "recall", "f1"):
        values = [
            metrics[(system["track"], system["display_name"], pair)][field]
            for pair in pairs
        ]
        out[field] = sum(values) / len(values)
    return out


def assign_ranks(rows: list[dict[str, Any]], field: str = "mcc") -> list[dict[str, Any]]:
    """Sort by ``field`` descending and assign competition ranks (ties share a rank)."""
    rows.sort(key=lambda row: (-row[field], row["sort_key"]))
    previous_value = None
    previous_rank = 0
    for position, row in enumerate(rows, start=1):
        if previous_value is None or row[field] != previous_value:
            previous_rank = position
            previous_value = row[field]
        row["rank"] = previous_rank
    return rows


# --------------------------------------------------------------------------
# Rendering helpers -- styled to match the WMT26 Task 1 ranking table
# --------------------------------------------------------------------------

PREAMBLE_NOTE = (
    "% Generated by run_latex_tables.py -- do not edit by hand.\n"
    "% Requires: \\usepackage{booktabs}, \\usepackage[table]{xcolor},\n"
    "% \\usepackage{amsmath} and \\usepackage{graphicx}\n"
)

BASELINE_ROW_COLOR = "\\rowcolor{black!15} "

# Organization tints.  The first four match the WMT26 Task 1 ranking table so
# that a team keeps the same colour across the two papers; the rest are drawn
# from PALETTE in alphabetical order of organization name.  A tint is only
# applied when the organization has more than one row in that table.
PINNED_ORG_COLORS = {
    "Algebras": "FCE0FF",
    "Charles University": "FFEAE0",
    "Soongsil University": "E0F0FF",
    "AIST": "F9FFE0",
}
PALETTE = (
    "E0FFEA", "FFE0E8", "EDE0FF", "E0FFFF", "FFF6E0",
    "E8F5E0", "FFE8F5", "E0E8FF", "FFF0E8", "F0E8FF",
)

BASELINE_ORG = "WMT Automated MT Evaluation Organizing Committee"

# Task 1 spellings for the organizer baselines, so the two rankings line up.
SUBMISSION_NAME_MAP = {
    "Gemini-3.6-Flash": "Gemini 3.6 Flash",
    "Gemma 4 - No Reasoning Baseline": "Gemma 4 No Reasoning",
    "Gemma 4 - Reasoning Baseline": "Gemma 4 Reasoning",
    "GPT-5.6 Sol - Medium Reasoning": "GPT-5.6 Sol Medium Reasoning",
    "Qwen 3.6 - No Reasoning Baseline": "Qwen 3.6 No Reasoning",
    "Qwen 3.6 - Reasoning Baseline": "Qwen 3.6 Reasoning",
    "xCOMET XL Reference-free - Baseline": "xCOMET XL",
    "xCOMET XXL Reference-free - Baseline": "xCOMET XXL",
    "Wukongopterus, a variant of Pterodactyl": "Wukongopterus",
    "Austriadraco, a variant of Pterodactyl": "Austriadraco",
    "Austriadactylus, a variant of Pterodactyl": "Austriadactylus",
}

# Task 1 spellings for the underlying models.  Vendor prefixes are dropped, as
# in the Task 1 ranking table.
MODEL_NAME_MAP = {
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "google/gemma-4-31B-it": "Gemma 4 31B IT",
    "google/gemma-4-31b-it": "Gemma 4 31B IT",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "GPT-4.1-mini-2025-04-14": "GPT-4.1 Mini (2025-04-14)",
    "Qwen/Qwen3.6-35B-A3B": "Qwen3.6-35B-A3B",
    "Qwen/Qwen3.6-27B": "Qwen3.6-27B",
    "Tower-Babel/Babel-83B-Chat": "Babel-83B-Chat",
    "Unbabel/XCOMET-XL": "xCOMET-XL",
    "Unbabel/XCOMET-XXL": "xCOMET-XXL",
    # Ensemble components, which the Task 1 table does not itself name.  These
    # follow its conventions (drop the vendor prefix, space out the version) and
    # the spelling the participants use in their own descriptions.
    "GPT-5.6-terra": "GPT-5.6 Terra",
    "GPT-5.6-luna": "GPT-5.6 Luna",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/gemini-3.1-pro-global": "Gemini 3.1 Pro",
    "mistral/mistral-large-3": "Mistral Large 3",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "qwen-plus": "Qwen Plus",
}

# Systems whose model card left the base model blank, but whose model the Task 1
# ranking table names.  Applied only when base_model is empty, keyed by
# submission name.
MODEL_BY_SUBMISSION = {
    "Cohere CAT+": "Command A+",
    "Qwen-based Omission": "Qwen3-32B",
    "Rule-based Omission": "Qwen3-32B",
    "cuni-cat-v4": "Gemma 4 31B",
    "cuni-v14": "Gemma 4 31B",
}

UNKNOWN_MODEL = "---"

# Significance clusters come from the separate permutation-test pipeline in
# ../significance-test/CLEAN_CODE (100k permutations, greedy WMT clustering).
# They are read from this file rather than recomputed; the recorded MCC is
# checked against the current data so a stale file cannot pass unnoticed.
CLUSTER_FILE = OUTPUT_DIR / "significance_clusters.csv"
SIGNIFICANCE_PERMUTATIONS = 100_000
SIGNIFICANCE_ALPHA = 0.05
CLUSTER_CAPTION_REF = (
    " Cluster is defined as in Table~\\ref{tab:wmt26-task3-priority}."
)
CLUSTER_CAPTION = (
    " Cluster groups systems that are not significantly different, by a paired "
    f"permutation test over source segments ({SIGNIFICANCE_PERMUTATIONS:,} "
    f"permutations, two-sided, $\\alpha = {SIGNIFICANCE_ALPHA}$) on the "
    "macro-averaged MCC, with WMT's greedy clustering."
)
TRACK_TO_METRIC_PREFIX = {
    "T3": "task3", "T1": "task1", "T2": "task2", "T1-B": "task1-baseline",
}

# \scalebox factor applied to the per-language tabulars (table 5 only).
PER_LANGUAGE_SCALE = 0.8


def pretty_submission(name: str) -> str:
    """Compute pretty submission from the supplied analysis state and return its representation."""
    return SUBMISSION_NAME_MAP.get(name, name)


def pretty_model(name: str) -> str:
    """Compute pretty model from the supplied analysis state and return its representation."""
    return MODEL_NAME_MAP.get(name, name)


def is_ensemble(base_model: str) -> bool:
    """Compute is ensemble from the supplied analysis state and return its representation."""
    return ";" in base_model or ", " in base_model


def ensemble_components(base_model: str) -> str:
    """Compute ensemble components from the supplied analysis state and return its representation."""
    parts = [
        pretty_model(part.strip())
        for part in base_model.replace(",", ";").split(";")
        if part.strip()
    ]
    return "; ".join(parts)


def assign_org_colors(systems: dict[tuple[str, str], dict[str, Any]]) -> dict[str, str]:
    """Compute assign org colors from the supplied analysis state and return its representation."""
    colors = dict(PINNED_ORG_COLORS)
    remaining = sorted(
        {
            system["organization_name"]
            for system in systems.values()
            if system["organization_name"] not in colors
            and system["organization_name"] != BASELINE_ORG
        }
    )
    for organization, color in zip(remaining, PALETTE):
        colors[organization] = color
    return colors


class TableStyle:
    """Per-table bookkeeping: row tints, name disambiguation, ensemble notes."""

    def __init__(self, rows: list[dict[str, Any]], org_colors: dict[str, str]) -> None:
        """Compute init from the supplied analysis state and return its representation."""
        self.org_colors = org_colors
        counts: Counter[str] = Counter()
        name_counts: Counter[tuple[str, str]] = Counter()
        for row in rows:
            system = row["system"]
            counts[system["organization_name"]] += 1
            name_counts[(pretty_submission(system["submission_name"]), system["track"])] += 1
        # Tint an organization only when it has several rows in this table.
        self.tinted = {
            organization
            for organization, count in counts.items()
            if count > 1 and organization != BASELINE_ORG
        }
        self.ambiguous = {key for key, count in name_counts.items() if count > 1}
        self.ensembles: list[tuple[str, str]] = []

    def row_color(self, system: dict[str, Any]) -> str:
        """Compute row color from the supplied analysis state and return its representation."""
        if system["is_baseline"]:
            return BASELINE_ROW_COLOR
        organization = system["organization_name"]
        if organization in self.tinted and organization in self.org_colors:
            return f"\\rowcolor[HTML]{{{self.org_colors[organization]}}} "
        return ""

    def submission_cell(self, system: dict[str, Any]) -> str:
        """Compute submission cell from the supplied analysis state and return its representation."""
        name = pretty_submission(system["submission_name"])
        if (name, system["track"]) in self.ambiguous and system["entry_id"]:
            name = f"{name} [{system['entry_id']}]"
        return escape(name)

    def model_cell(self, system: dict[str, Any]) -> str:
        """Compute model cell from the supplied analysis state and return its representation."""
        base_model = system["base_model"].strip()
        if not base_model:
            fallback = MODEL_BY_SUBMISSION.get(system["submission_name"])
            return escape(fallback) if fallback else UNKNOWN_MODEL
        if not is_ensemble(base_model):
            return escape(pretty_model(base_model))
        label = pretty_submission(system["submission_name"])
        components = ensemble_components(base_model)
        for index, (existing_label, existing) in enumerate(self.ensembles, start=1):
            if (existing_label, existing) == (label, components):
                return f"Ensemble\\textsuperscript{{E{index}}}"
        self.ensembles.append((label, components))
        return f"Ensemble\\textsuperscript{{E{len(self.ensembles)}}}"

    def ensemble_note(self) -> str:
        """Compute ensemble note from the supplied analysis state and return its representation."""
        if not self.ensembles:
            return ""
        parts = [
            f"\\textsuperscript{{E{index}}} {escape(label)}: {escape(components)}"
            for index, (label, components) in enumerate(self.ensembles, start=1)
        ]
        return " Ensemble components named in the model cards are " + "; ".join(parts) + "."


def num(value: Any) -> str:
    """Metrics are tabulated on a 0--100 scale, two decimals, no per-cent sign."""
    return "---" if value is None or value == "" else f"${float(value) * 100:.2f}$"


GENERATED_NOTE = "% Generated by run_latex_tables.py -- do not edit by hand.\n"


def write(path: Path, lines: list[str], note: str = PREAMBLE_NOTE) -> None:
    """Compute write from the supplied analysis state and return its representation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def centered(label: str) -> str:
    """Compute centered from the supplied analysis state and return its representation."""
    return f"\\multicolumn{{1}}{{c}}{{{label}}}"


# --------------------------------------------------------------------------
# Table 1 / Table 2: macro-average rankings
# --------------------------------------------------------------------------

# P/R/F1 are proportions, so (%) is right for them; MCC is a correlation
# coefficient, so its header states the scaling instead of calling it a per cent.
METRIC_HEADERS = (
    "$\\text{MCC}_{\\times 100}$", "P(\\%)", "R(\\%)", "F1(\\%)",
)
METRIC_FIELDS = ("mcc", "precision", "recall", "f1")


def macro_table(
    rows: list[dict[str, Any]],
    org_colors: dict[str, str],
    caption: str,
    label: str,
    show_track: bool,
    placement: str = "htbp",
    scale: float | None = None,
    clusters: dict[str, int] | None = None,
) -> list[str]:
    """Compute macro table from the supplied analysis state and return its representation."""
    style = TableStyle(rows, org_colors)
    colspec = ("rr" if clusters else "r") + "ll"
    colspec += ("l" if show_track else "") + "r" * len(METRIC_FIELDS)
    header = [centered("Rank")]
    if clusters:
        header.append(centered("Cluster"))
    header += ["Submission", "Underlying model"]
    if show_track:
        header.append("Track")
    header += [centered(name) for name in METRIC_HEADERS]

    body = []
    for row in rows:
        system = row["system"]
        cells = [f"${row['rank']}$"]
        if clusters:
            cells.append(f"${clusters[metric_key(system)]}$")
        cells += [style.submission_cell(system), style.model_cell(system)]
        if show_track:
            cells.append(system["track"])
        cells += [num(row[field]) for field in METRIC_FIELDS]
        body.append(style.row_color(system) + " & ".join(cells) + " \\\\")

    tabular = [
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    if scale is not None:
        tabular = [f"\\scalebox{{{scale}}}{{%", *tabular, "}"]
    return [
        f"\\begin{{table*}}[{placement}]",
        "\\centering",
        *tabular,
        f"\\caption{{{caption}{style.ensemble_note()}}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ]


def per_pair_mcc_table(
    rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
    pairs: tuple[str, ...],
    metrics: dict[tuple[str, str, str], dict[str, float]],
    org_colors: dict[str, str],
    caption: str,
    label: str,
    show_track: bool = False,
) -> list[str]:
    """Compute per pair mcc table from the supplied analysis state and return its representation."""
    style = TableStyle(rows + extra_rows, org_colors)
    colspec = "rll" + ("c" if show_track else "") + "r" * len(pairs) + "r"
    header = [centered("Rank"), "Submission", "Underlying model"]
    if show_track:
        header.append(centered("Track"))
    header += [centered(escape(short_pair(pair))) for pair in pairs]
    header.append(centered("Avg."))

    def cell(system: dict[str, Any], pair: str) -> float | None:
        """Compute cell from the supplied analysis state and return its representation."""
        key = (system["track"], system["display_name"], pair)
        return metrics[key]["mcc"] if key in metrics else None

    # Compare against every row shown, so a partial-coverage system that leads a
    # column is not passed over when marking the best value.
    best = {}
    for pair in pairs:
        values = [
            value
            for row in rows + extra_rows
            if (value := cell(row["system"], pair)) is not None
        ]
        best[pair] = max(values) if values else None

    def body_line(row: dict[str, Any], ranked: bool) -> str:
        """Compute body line from the supplied analysis state and return its representation."""
        system = row["system"]
        cells = [
            f"${row['rank']}$" if ranked else "",
            style.submission_cell(system),
            style.model_cell(system),
        ]
        if show_track:
            cells.append(system["track"])
        for pair in pairs:
            value = cell(system, pair)
            if value is None:
                cells.append(UNKNOWN_MODEL)
            elif value == best[pair]:
                cells.append(f"$\\mathbf{{{value * 100:.2f}}}$")
            else:
                cells.append(num(value))
        cells.append(num(row["mcc"]) if ranked else UNKNOWN_MODEL)
        return style.row_color(system) + " & ".join(cells) + " \\\\"

    body = [body_line(row, ranked=True) for row in rows]
    if extra_rows:
        body.append("\\midrule")
        body += [body_line(row, ranked=False) for row in extra_rows]

    return [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}{style.ensemble_note()}}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ]


def build_omitted(
    systems: dict[tuple[str, str], dict[str, Any]],
    pairs: tuple[str, ...],
    directories: Iterable[str],
) -> list[dict[str, Any]]:
    """Systems from ``directories`` covering some, but not all, of ``pairs``.

    Systems with no overlap at all are left out: every cell would be empty, which
    reads as a system that scored nothing rather than one that competed in other
    directions.  The caption note from ``omitted_note`` still accounts for them.
    """
    wanted = set(directories)
    required = set(pairs)
    rows = []
    for system in systems.values():
        if system["directory"] not in wanted or required <= system["pairs"]:
            continue
        if not system["pairs"] & required:
            continue
        rows.append(
            {
                "system": system,
                "covered": len(system["pairs"] & required),
                "sort_key": (system["display_name"], system["track"]),
            }
        )
    rows.sort(key=lambda row: (-row["covered"], row["sort_key"]))
    return rows


def build_ranking(
    systems: dict[tuple[str, str], dict[str, Any]],
    metrics: dict[tuple[str, str, str], dict[str, float]],
    pairs: tuple[str, ...],
    directories: Iterable[str],
) -> list[dict[str, Any]]:
    """Systems from ``directories`` covering every pair in ``pairs``, macro-averaged."""
    wanted = set(directories)
    required = set(pairs)
    rows = []
    for system in systems.values():
        if system["directory"] not in wanted:
            continue
        if not required <= system["pairs"]:
            continue
        averages = macro_average(system, metrics, pairs)
        rows.append(
            {
                "system": system,
                "sort_key": (system["display_name"], system["track"]),
                **averages,
            }
        )
    return assign_ranks(rows)


def omitted_note(
    systems: dict[tuple[str, str], dict[str, Any]],
    pairs: tuple[str, ...],
    directories: Iterable[str],
    show_track: bool,
    name_limit: int = 6,
) -> str:
    """Name the systems a ranking table drops for not covering every pair.

    Without this the omission is invisible: a system can be absent from all
    three rankings and still appear in the per-language tables.
    """
    wanted = set(directories)
    required = set(pairs)
    groups: dict[str, dict[str, set]] = {}
    for system in systems.values():
        if system["directory"] not in wanted or required <= system["pairs"]:
            continue
        group = groups.setdefault(
            pretty_submission(system["submission_name"]),
            {"counts": set(), "tracks": set()},
        )
        group["counts"].add(len(system["pairs"] & required))
        group["tracks"].add(system["track"])
    if not groups:
        return ""

    total = len(required)
    if len(groups) > name_limit:
        return (
            f" {len(groups)} further systems did not cover all {total} language "
            "pairs and are omitted; their results appear in the per-language "
            "tables."
        )

    described = []
    for name, group in sorted(groups.items()):
        counts = sorted(group["counts"])
        if counts == [0]:
            extent = f"no {'priority ' if total < 21 else ''}pairs"
        elif len(counts) == 1:
            extent = f"{counts[0]} of {total}"
        else:
            extent = f"{counts[0]}--{counts[-1]} of {total}"
        if show_track:
            extent += "; " + ", ".join(sorted(group["tracks"]))
        described.append(f"{escape(name)} ({extent})")
    return (
        " Omitted for not covering every pair: "
        + join_names(described)
        + ". Their results appear in the per-language tables."
    )


def write_ranking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Compute write ranking csv from the supplied analysis state and return its representation."""
    fields = [
        "rank", "track", "track_long", "display_name", "organization_name",
        "submission_name", "base_model", "entry_id", "n_pairs",
        "mcc", "precision", "recall", "f1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            system = row["system"]
            writer.writerow(
                {
                    "rank": row["rank"],
                    "track": system["track"],
                    "track_long": system["track_long"],
                    "display_name": system["display_name"],
                    "organization_name": system["organization_name"],
                    "submission_name": system["submission_name"],
                    "base_model": system["base_model"],
                    "entry_id": system["entry_id"],
                    "n_pairs": row.get("n_pairs", ""),
                    "mcc": f"{row['mcc']:.6f}",
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "f1": f"{row['f1']:.6f}",
                }
            )
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Table 3: per-language breakdowns
# --------------------------------------------------------------------------

TRACK_LEGEND = (
    "Track: T3 is a direct Task 3 submission, T1 and T2 are conversions of a "
    "Task 1 or Task 2 submission, and T1-B is a converted organizer Task 1 "
    "baseline. Light-grey rows are organizer baselines. Conversions whose "
    "source system used the reference translations are excluded, Task 3 being "
    "a reference-free task."
)


TRACK_LEGEND_SHORT = (
    "Track: T3 is a direct Task 3 submission, T1 and T2 are conversions of a "
    "Task 1 or Task 2 submission, and T1-B is a converted organizer Task 1 "
    "baseline. Conversions whose source system used the reference translations "
    "are excluded, Task 3 being a reference-free task."
)


def per_language_table(
    pair: str,
    systems: dict[tuple[str, str], dict[str, Any]],
    metrics: dict[tuple[str, str, str], dict[str, float]],
    summary: dict[str, Any],
    org_colors: dict[str, str],
) -> list[str]:
    """Compute per language table from the supplied analysis state and return its representation."""
    rows = []
    for system in systems.values():
        key = (system["track"], system["display_name"], pair)
        if key not in metrics:
            continue
        rows.append(
            {
                "system": system,
                "sort_key": (system["display_name"], system["track"]),
                **metrics[key],
            }
        )
    assign_ranks(rows)

    coverage = (
        f"{int(summary['annotated_items']):,}/{int(summary['inventory_items']):,} items "
        f"({float(summary['item_coverage']):.1%}) and "
        f"{int(summary['annotated_item_systems']):,}/"
        f"{int(summary['inventory_item_systems']):,} item--system pairs "
        f"({float(summary['item_system_coverage']):.1%}) are annotated, giving "
        f"{int(summary['human_assessments']):,} human assessments "
        f"({int(summary['error_free']):,} error-free, {int(summary['error']):,} error)."
    ).replace("%", r"\%")
    caption = (
        f"WMT26 Task 3 results for {escape(short_pair(pair))} "
        f"(\\texttt{{{escape(pair)}}}). {coverage} {TRACK_LEGEND}"
    )
    # ~40 systems per pair: full text width, \footnotesize and [p] placement keep
    # each breakdown on a single page.
    lines = macro_table(
        rows,
        org_colors,
        caption,
        f"tab:wmt26-task3-{pair_slug(pair)}",
        show_track=True,
        placement="p",
        scale=PER_LANGUAGE_SCALE,
    )
    lines.insert(2, "\\footnotesize")
    return lines


# --------------------------------------------------------------------------
# System descriptions, quoted from the Task 3 model cards
# --------------------------------------------------------------------------


def load_descriptions() -> dict[str, str]:
    """Map submission id -> description, for every model card on disk."""
    if not MODEL_CARD_DIR.is_dir():
        return {}
    descriptions = {}
    for path in sorted(MODEL_CARD_DIR.glob("*/model_card.json")):
        submission_id = path.parent.name.split("_")[0]
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in card:
            if entry.get("question") == DESCRIPTION_QUESTION:
                text = (entry.get("answer") or "").strip()
                if text:
                    descriptions[submission_id] = text
                break
    return descriptions


def edited(submission_id: str, text: str) -> str:
    """Compute edited from the supplied analysis state and return its representation."""
    for target_id, before, after in DESCRIPTION_EDITS:
        if target_id in (None, submission_id):
            text = text.replace(before, after)
    return text


def quote(text: str) -> str:
    """Escape a model-card excerpt, then apply the typesetting rules."""
    text = escape(text)
    for before, after in DESCRIPTION_FORMATTING:
        text = text.replace(before, after)
    return text


def description_entries(
    systems: dict[tuple[str, str], dict[str, Any]],
    descriptions: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group the direct Task 3 submissions in the tables by organization."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    missing = []
    for system in systems.values():
        if system["track"] != "T3" or system["is_baseline"]:
            continue
        text = descriptions.get(system["entry_id"])
        if not text:
            missing.append(system)
            continue
        organization = system["organization_name"]
        team = ORGANIZATION_MERGES.get(organization, organization)
        grouped.setdefault(team, []).append(
            {
                "name": pretty_submission(system["submission_name"]),
                "id": system["entry_id"],
                "organization_name": organization,
                "text": edited(system["entry_id"], text),
            }
        )

    entries = []
    for organization, members in sorted(grouped.items()):
        members.sort(key=lambda member: (member["name"], member["id"]))
        entries.append({"organization_name": organization, "systems": members})
    missing.sort(key=lambda system: system["display_name"])
    return entries, missing


NUMBER_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten",
}
ORDINAL_WORDS = {0: "first", 1: "second", 2: "third", 3: "fourth", 4: "fifth"}


def join_names(names: list[str]) -> str:
    """Compute join names from the supplied analysis state and return its representation."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def sentences(text: str) -> list[str]:
    """Compute sentences from the supplied analysis state and return its representation."""
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def shared_run(texts: list[str]) -> tuple[str, list[str], str]:
    """Split the runs of sentences common to the start and end of every text.

    Teams that entered variants of one system often repeat whole sentences in
    each model card; quoting those once reads better and still uses only the
    participant's own words.  Returns (opening, per-text remainders, closing).
    """
    split = [sentences(text) for text in texts]
    budget = min(len(parts) for parts in split) - 1  # keep a remainder each

    head = 0
    while head < budget and all(
        parts[head] == split[0][head] for parts in split
    ):
        head += 1

    budget -= head
    tail = 0
    while tail < budget and all(
        parts[-1 - tail] == split[0][-1 - tail] for parts in split
    ):
        tail += 1

    opening = " ".join(split[0][:head])
    closing = " ".join(split[0][len(split[0]) - tail:]) if tail else ""
    remainders = [
        " ".join(parts[head:len(parts) - tail] if tail else parts[head:])
        for parts in split
    ]
    return opening, remainders, closing


def lead_in(organization: str) -> str:
    """Avoid a doubled period after names like ``Phrase a.s.``"""
    return organization if organization.endswith(".") else organization + "."


def entry_paragraph(entry: dict[str, Any]) -> str:
    """One flowing paragraph per organization, keeping the model-card wording."""
    organization = escape(entry["organization_name"])
    members = entry["systems"]
    names = []
    for member in members:
        name = escape(member["name"])
        if name not in names:
            names.append(name)

    if len(members) == 1:
        lead = lead_in(f"This submission comes from {organization}")
        return f"{lead} {quote(members[0]['text'])}"

    count = NUMBER_WORDS.get(len(members), str(len(members)))
    ids = join_names([member["id"] for member in members])
    # Several teams entered the same system twice; quote the shared write-up once
    # rather than repeating it verbatim.
    if len({member["text"] for member in members}) == 1:
        if len(names) == 1:
            lead = (
                f"{organization} submitted {count} systems, both named "
                f"{names[0]} (submissions {ids}), with the same description in "
                "their model cards."
            )
        else:
            lead = (
                f"{organization} submitted {count} systems, "
                f"{join_names(names)}, with the same description in their "
                "model cards."
            )
        return f"{lead} {quote(members[0]['text'])}"

    parts = [f"{organization} submitted {count} systems."]
    opening, remainders, closing = shared_run(
        [member["text"] for member in members]
    )
    if opening:
        parts.append(quote(opening))
    for index, member in enumerate(members):
        ordinal = ORDINAL_WORDS.get(index, f"{index + 1}th")
        name = escape(member["name"])
        if member["organization_name"] != entry["organization_name"]:
            name += f", submitted as {escape(member['organization_name'])}"
        parts.append(lead_in(f"The {ordinal} system is {name}"))
        parts.append(quote(remainders[index]))
    if closing:
        parts.append(f"Both descriptions close identically: {quote(closing)}")
    return " ".join(parts)


def descriptions_document(entries: list[dict[str, Any]]) -> list[str]:
    """Compute descriptions document from the supplied analysis state and return its representation."""
    lines = [
        "% System descriptions for the direct Task 3 submissions, quoted from",
        "% the participants' model cards and lightly copy-edited.  One paragraph",
        "% per organization; see DESCRIPTION_EDITS in run_latex_tables.py.",
    ]
    for entry in entries:
        names = []
        for member in entry["systems"]:
            name = escape(member["name"])
            if name not in names:
                names.append(name)
        lines += [
            "",
            f"\\paragraph{{{join_names(names)}}}",
            entry_paragraph(entry),
        ]
    return lines


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def render_tables() -> None:
    """Render the original ranking, breakdown and description tables unchanged."""
    systems, metrics = load_rows()
    summary = load_summary()
    all_pairs = tuple(sorted(summary))
    org_colors = assign_org_colors(systems)

    missing = set(PRIORITY_PAIRS) - set(all_pairs)
    if missing:
        raise SystemExit(f"priority pairs not in summary: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Table 1: official ranking over the priority pairs ----
    official = build_ranking(systems, metrics, PRIORITY_PAIRS, OFFICIAL_SOURCES)
    for row in official:
        row["n_pairs"] = len(PRIORITY_PAIRS)
    caption1 = (
        "WMT26 Task 3 priority-direction ranking: direct Task 3 submissions and "
        "converted organizer Task 1 baselines covering all "
        f"{len(PRIORITY_PAIRS)} priority language pairs "
        f"({', '.join(short_pair(p) for p in PRIORITY_PAIRS)}), macro averaged "
        "over those pairs. MCC is the primary metric; precision (P), recall (R) "
        "and F1 take error-free as the positive class. Light-grey rows are "
        "organizer baselines."
    ) + omitted_note(systems, PRIORITY_PAIRS, OFFICIAL_SOURCES, show_track=False)
    official_clusters = load_clusters("table1", official)
    if official_clusters:
        caption1 += CLUSTER_CAPTION
    write(
        OUTPUT_DIR / "table1_official_ranking.tex",
        macro_table(
            official, org_colors, caption1, "tab:wmt26-task3-priority",
            show_track=False, clusters=official_clusters,
        ),
    )
    official_omitted = build_omitted(systems, PRIORITY_PAIRS, OFFICIAL_SOURCES)
    caption1b = (
        "WMT26 Task 3 priority-direction ranking, per-language-pair MCC "
        f"($\\times 100$) for the systems of Table~\\ref{{tab:wmt26-task3-priority}}, whose "
        "conventions apply here. "
        "Avg.\\ is the macro average that determines the ranking; the best value "
        "per language pair is in bold. "
        "Below the rule are the systems that covered only some of the priority "
        "pairs: "
        "they are unranked and have no average, since it would be taken over a "
        "different set of pairs, but their per-pair scores are shown where they "
        "competed."
    )
    write(
        OUTPUT_DIR / "table2_official_ranking_per_pair.tex",
        per_pair_mcc_table(
            official, official_omitted, PRIORITY_PAIRS, metrics, org_colors,
            caption1b, "tab:wmt26-task3-priority-per-pair",
        ),
    )
    write_ranking_csv(OUTPUT_DIR / "table1_official_ranking.csv", official)

    # ---- Table 3: the priority pairs again, but across all tracks ----
    # Same language pairs and averaging as table 1, same system pool as table 4,
    # so the two can be read against each other.
    all_directories = [directory for directory, *_ in SOURCES]
    priority_all_tracks = build_ranking(
        systems, metrics, PRIORITY_PAIRS, all_directories
    )
    for row in priority_all_tracks:
        row["n_pairs"] = len(PRIORITY_PAIRS)
    caption1c = (
        "WMT26 Task 3 priority-direction ranking across all tracks: every "
        f"system covering the same {NUMBER_WORDS[len(PRIORITY_PAIRS)]} priority "
        f"language pairs as Table~\\ref{{tab:wmt26-task3-priority}}, macro averaged over those "
        "pairs, with converted Task 1 and Task 2 submissions added. Metrics and "
        f"row shading are as in Table~\\ref{{tab:wmt26-task3-priority}}. {TRACK_LEGEND_SHORT}"
    ) + omitted_note(systems, PRIORITY_PAIRS, all_directories, show_track=True)
    priority_clusters = load_clusters("table3", priority_all_tracks)
    if priority_clusters:
        caption1c += CLUSTER_CAPTION_REF
    write(
        OUTPUT_DIR / "table3_priority_all_tracks.tex",
        macro_table(
            priority_all_tracks, org_colors, caption1c,
            "tab:wmt26-task3-priority-all-tracks", show_track=True,
            clusters=priority_clusters,
        ),
    )
    write_ranking_csv(
        OUTPUT_DIR / "table3_priority_all_tracks.csv", priority_all_tracks
    )

    # ---- Table 4: every system covering all language pairs ----
    complete = build_ranking(systems, metrics, all_pairs, all_directories)
    for row in complete:
        row["n_pairs"] = len(all_pairs)
    caption2 = (
        f"WMT26 Task 3 ranking over all {len(all_pairs)} language pairs, "
        "restricted to systems that cover every pair and macro averaged over "
        f"pairs. Columns, tracks and row shading are as in "
        f"Table~\\ref{{tab:wmt26-task3-priority-all-tracks}}."
    ) + omitted_note(systems, all_pairs, all_directories, show_track=True)
    complete_clusters = load_clusters("table4", complete)
    if complete_clusters:
        caption2 += CLUSTER_CAPTION_REF
    write(
        OUTPUT_DIR / "table4_all_pairs.tex",
        macro_table(
            complete, org_colors, caption2, "tab:wmt26-task3-all-pairs",
            show_track=True, clusters=complete_clusters,
        ),
    )
    write_ranking_csv(OUTPUT_DIR / "table4_all_pairs.csv", complete)

    # ---- Table 5: per-language breakdowns ----
    blocks = []
    per_language_files = []
    for pair in all_pairs:
        name = f"per_language/{pair_slug(pair)}.tex"
        lines = per_language_table(pair, systems, metrics, summary[pair], org_colors)
        write(OUTPUT_DIR / name, lines)
        per_language_files.append(name)
        blocks.append((pair, lines))
    # Self-contained: the tables are inlined rather than \input, so this single
    # file can be dropped into an Overleaf project on its own.
    combined = [
        f"% All {len(all_pairs)} per-language-pair breakdown tables for Task 3,",
        "% one table per pair, ordered by language-pair code.",
        "%",
        "% Self-contained: upload this one file and \\input it. It pulls in no",
        "% other file from this directory.",
        "%     \\input{table5_per_language}",
        "%",
        "% Requires, in your preamble:",
        "%     \\usepackage{booktabs}",
        "%     \\usepackage[table]{xcolor}",
        "%     \\usepackage{amsmath}",
        "%",
        "% Each table is a full-width table* float with [p] placement, so they",
        "% land on float pages after the current text, not at the \\input point.",
    ]
    for pair, lines in blocks:
        combined += [
            "",
            f"% ---------- {short_pair(pair)} ({pair}) ----------",
            *lines,
        ]
    write(OUTPUT_DIR / "table5_per_language.tex", combined, note=GENERATED_NOTE)

    # ---- System descriptions from the Task 3 model cards ----
    descriptions = load_descriptions()
    if not descriptions:
        print(
            f"note: no model cards under {MODEL_CARD_DIR}; "
            "skipping system_descriptions.tex"
        )
        description_count = 0
    else:
        entries, missing = description_entries(systems, descriptions)
        for system in missing:
            print(f"warning: no model-card description for {system['display_name']}")
        write(
            OUTPUT_DIR / "system_descriptions.tex",
            descriptions_document(entries),
            note=GENERATED_NOTE,
        )
        description_count = len(entries)

    # ---- Draft: the six non-priority pairs PTT-MULTITAN-v2 targeted ----
    # Not part of the numbered set; drop DRAFT_PAIRS to remove it entirely.
    if DRAFT_PAIRS:
        draft_ranked = build_ranking(systems, metrics, DRAFT_PAIRS, all_directories)
        draft_omitted = build_omitted(systems, DRAFT_PAIRS, all_directories)
        draft_caption = (
            "WMT26 Task 3 results on the "
            f"{NUMBER_WORDS[len(DRAFT_PAIRS)]} language pairs targeted by "
            "PTT-MULTITAN-v2, none of which is a priority direction: "
            "per-language-pair MCC ($\\times 100$) for every system covering all "
            f"{NUMBER_WORDS[len(DRAFT_PAIRS)]}, macro averaged in Avg.\\ The best "
            "value per language pair is in bold. Below the rule are systems that "
            "covered only some of these pairs; they are unranked and have no "
            f"average. {TRACK_LEGEND}"
        )
        write(
            OUTPUT_DIR / "draft_altae_pairs.tex",
            per_pair_mcc_table(
                draft_ranked, draft_omitted, DRAFT_PAIRS, metrics, org_colors,
                draft_caption, "tab:wmt26-task3-altae-pairs", show_track=True,
            ),
        )
        (OUTPUT_DIR / "draft_preview.tex").write_text(
            "\n".join([
                "% Standalone render of the draft table only.",
                "\\documentclass[10pt]{article}",
                "\\usepackage[margin=1.5cm,landscape,a3paper]{geometry}",
                "\\usepackage[T1]{fontenc}",
                "\\usepackage[utf8]{inputenc}",
                "\\usepackage{booktabs}",
                "\\usepackage[table]{xcolor}",
                "\\usepackage{amsmath}",
                "\\begin{document}",
                "\\input{draft_altae_pairs}",
                "\\end{document}",
            ]) + "\n",
            encoding="utf-8",
        )
        print("wrote results/latex/draft_altae_pairs.tex (+ draft_preview.tex)")

    # ---- Standalone preview document ----
    preview = [
        "% Preview document: compile from this directory with",
        "%   pdflatex preview.tex",
        "\\documentclass[10pt]{article}",
        "\\usepackage[margin=1.5cm,landscape,a3paper]{geometry}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{booktabs}",
        "\\usepackage[table]{xcolor}",
        "\\usepackage{graphicx}",  # \\scalebox in the per-language tables
        "\\usepackage{amsmath}",  # \\text in the MCC header
        "\\begin{document}",
        "\\input{table1_official_ranking}",
        "\\input{table2_official_ranking_per_pair}",
        "\\input{table3_priority_all_tracks}",
        "\\input{table4_all_pairs}",
    ]
    preview += ["\\input{table5_per_language}"]
    if description_count:
        preview += ["\\clearpage", "\\input{system_descriptions}"]
    preview += ["\\end{document}"]
    (OUTPUT_DIR / "preview.tex").write_text("\n".join(preview) + "\n", encoding="utf-8")
    print("wrote results/latex/preview.tex")

    print(
        f"\n{len(official)} systems in tables 1-2, {len(priority_all_tracks)} in "
        f"table 3, {len(complete)} in table 4, "
        f"{len(all_pairs)} per-language tables, "
        f"{description_count} system descriptions."
    )


def main(argv: list[str] | None = None) -> int:
    """Apply release paths, then run the original table-rendering implementation."""
    global RESULTS_DIR, OUTPUT_DIR, MODEL_CARD_DIR, CLUSTER_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/task3"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task3"))
    args = parser.parse_args(argv)
    RESULTS_DIR = args.output_dir
    OUTPUT_DIR = args.output_dir / "latex"
    MODEL_CARD_DIR = args.data_dir / "submissions/task3"
    CLUSTER_FILE = OUTPUT_DIR / "significance_clusters.csv"
    render_tables()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
