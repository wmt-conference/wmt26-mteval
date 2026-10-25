"""Allow-list scrubbing of the participants' model cards.

A raw card is a JSON list of ``{"question", "required", "answer"}`` items.  It also holds personal
data (the participant's name, their e-mail address and the data-collection consent answer) that the
consent text promised would never be published.  The scrubber therefore works from an allow-list:
an item is kept only when its question starts with one of the prefixes in ``CARD_FIELDS``.  Every
other item is dropped without its answer being read, so nothing new that the form gains can leak.

Kept answers stay raw strings.  ``normalise_bool`` and ``parse_threshold`` type them later.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from wmt26_mteval.runs import RunSpec

# Question-text prefix -> key.  The prefixes are matched with ``str.startswith``.
CARD_FIELDS: dict[str, str] = {
    "Organization name": "org",
    "Submission name": "name",
    "Submission description": "description",
    "Does your system use the provided reference": "uses_reference",
    "If your system uses reference": "reference_usage",
    "Is an ensemble": "ensemble",
    "Is supervised": "fine_tuned",
    "Is based on open-weight": "open_weight",
    "Model size": "model_size",
    "Closed-source model name": "closed_model_name",
    "Closed-source API provider": "api_provider",
    "Is the submitted system open-source": "open_source",
    "Provide link": "link",
    "Allow automatic conversion from Task 1 to Task 2": "allow_t1_to_t2",
    "Allow automatic conversion from Task 1 to Task 3": "allow_t1_to_t3",
    "Allow automatic conversion from Task 2 to Task 3": "allow_t2_to_t3",
    "Task 2 to 3 conversion threshold": "t3_threshold",
    "Should the above threshold be inclusive": "threshold_inclusive",
}

# One allow-listed answer names individuals, next to the organisations they belong to.  It is
# replaced by the organisations alone, spelt as Table 3 of the findings paper prints them.
ANSWER_OVERRIDES: Mapping[tuple[str, str], str] = {
    ("871298", "org"): "IIT Patna; Shiv Nadar University",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_PLACEHOLDER = "[email removed]"
CARD_FILENAME = "model_card.json"

# What ``scrub_runs`` reports for a changed answer.
EMAIL_REMOVED = "e-mail address removed"
NAMES_REPLACED = "personal names replaced"

_TRUE = frozenset({"true", "yes"})
_FALSE = frozenset({"false", "no"})


def _field_key(question: str) -> str | None:
    """Key of the allow-listed field that ``question`` asks, or ``None`` if it is not listed."""
    for prefix, key in CARD_FIELDS.items():
        if question.startswith(prefix):
            return key
    return None


def _answer_text(answer: object, key: str) -> str:
    """A kept answer as text.  Some participants' cards hold JSON booleans, not strings."""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, bool):
        return "true" if answer else "false"
    raise TypeError(f"answer for {key!r} is a {type(answer).__name__}, not text")


def scrub_card_report(raw: list[dict]) -> tuple[dict[str, str], list[str]]:
    """``scrub_card`` plus the keys whose answers held an e-mail address (now removed).

    Only the ``question`` of a dropped item is looked at.  An allow-listed answer that contains an
    e-mail address, for example in the description or the link, has each address replaced by
    ``[email removed]``.  A question that is asked twice raises ``ValueError``.
    """
    answers: dict[str, str] = {}
    redacted: list[str] = []
    for item in raw:
        key = _field_key(item["question"])
        if key is None:
            continue
        if key in answers:
            raise ValueError(f"the card asks the question for {key!r} more than once")
        text = _answer_text(item["answer"], key)
        if EMAIL_RE.search(text):
            text = EMAIL_RE.sub(EMAIL_PLACEHOLDER, text)
            redacted.append(key)
        answers[key] = text
    return answers, redacted


def scrub_card(raw: list[dict]) -> dict[str, str]:
    """Raw answers of the allow-listed questions only, keyed as in ``CARD_FIELDS``."""
    return scrub_card_report(raw)[0]


def normalise_bool(s: str) -> bool | None:
    """``true``/``yes`` is ``True``, ``false``/``no`` is ``False`` (any case); else ``None``."""
    word = s.strip().lower()
    if word in _TRUE:
        return True
    if word in _FALSE:
        return False
    return None


def parse_threshold(threshold: str, inclusive: str) -> tuple[float | None, bool | None]:
    """The Task 2 to 3 conversion threshold and whether a score equal to it counts as error-free.

    The threshold is ``None`` unless it is a finite number.  ``inclusive`` is ``True`` for
    ``inclusive``, ``≥`` or ``>=``, ``False`` for ``exclusive`` or ``>``, and ``None`` when it
    says neither or, as when the whole question is pasted back, both.
    """
    try:
        value: float | None = float(threshold.strip())
    except ValueError:
        value = None
    if value is not None and not math.isfinite(value):
        value = None

    word = inclusive.strip().lower()
    says_inclusive = "inclusive" in word or "≥" in word or ">=" in word
    says_exclusive = "exclusive" in word or ">" in word.replace(">=", "")
    if says_inclusive == says_exclusive:
        return value, None
    return value, says_inclusive


def card_source(raw_root: Path, run: RunSpec) -> Path:
    """The raw ``model_card.json`` of ``run`` under ``raw_root``.

    A submission's card is ``submissions/<task>/<id>``.  A numeric baseline (one of the
    segment-level metrics) has its card in ``baselines/task2/<id>``.  An LLM or xCOMET baseline
    has one in ``baselines/task2/<run_id>_task2`` (it carries the threshold answers) and, failing
    that, in ``baselines/task1/<run_id>_task1``.  ``FileNotFoundError`` if there is none.
    """
    raw_root = Path(raw_root)
    if run.kind == "submission":
        candidates = [raw_root / "submissions" / run.task / run.run_id]
    elif run.run_id.isdigit():
        candidates = [raw_root / "baselines" / "task2" / run.run_id]
    else:
        candidates = [
            raw_root / "baselines" / "task2" / f"{run.run_id}_task2",
            raw_root / "baselines" / "task1" / f"{run.run_id}_task1",
        ]
    for folder in candidates:
        if (folder / CARD_FILENAME).is_file():
            return folder / CARD_FILENAME
    tried = ", ".join(str(folder.relative_to(raw_root) / CARD_FILENAME) for folder in candidates)
    raise FileNotFoundError(f"no model card for {run.kind} run {run.run_id}: tried {tried}")


def load_raw_card(path: Path) -> list[dict]:
    """A raw card: a JSON list of question items."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"{path}: a model card must be a JSON list of objects")
    return raw


def card_text(run_id: str, answers: dict[str, str]) -> str:
    """The scrubbed card of ``run_id`` as written to disk, the same bytes on every run."""
    document = {"run_id": run_id, "answers": answers}
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def scrub_runs(
    raw_root: Path,
    runs: list[RunSpec],
    out_dir: Path,
    overrides: Mapping[tuple[str, str], str] = ANSWER_OVERRIDES,
) -> list[tuple[str, str, str]]:
    """Write ``<run_id>.json`` under ``out_dir`` for each ``official`` run in ``runs``.

    An answer listed in ``overrides`` under ``(run_id, key)`` is replaced by the given text.
    Returns ``(run_id, key, change)`` for every answer that was altered, where ``change`` is
    ``EMAIL_REMOVED`` or ``NAMES_REPLACED``, so a report never needs to quote an answer.  A
    missing card, or two official runs with one run id, raises.
    """
    official = [run for run in runs if run.variant == "official"]
    seen: set[str] = set()
    for run in official:
        if run.run_id in seen:
            raise ValueError(f"two official runs share the run id {run.run_id!r}")
        seen.add(run.run_id)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    changes: list[tuple[str, str, str]] = []
    for run in official:
        answers, redacted = scrub_card_report(load_raw_card(card_source(raw_root, run)))
        changes.extend((run.run_id, key, EMAIL_REMOVED) for key in redacted)
        for (run_id, key), text in overrides.items():
            if run_id == run.run_id and key in answers:
                answers[key] = text
                changes.append((run.run_id, key, NAMES_REPLACED))
        with open(out_dir / f"{run.run_id}.json", "w", encoding="utf-8", newline="\n") as handle:
            handle.write(card_text(run.run_id, answers))
    return changes
