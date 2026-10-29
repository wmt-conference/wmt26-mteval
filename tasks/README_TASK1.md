# Reproducing the WMT26 Task 1 paper results

This repository also contains the Task 2 reproduction package. See
[`README_TASK2.md`](README_TASK2.md) for its setup and commands.

## Setup

Use Python 3.12 or newer. From the repository root, create a virtual
environment and install the dependencies declared in `pyproject.toml`.
The numerical/plotting versions are pinned there and shared by Tasks 1–3:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Installation needs Git and network access for the pinned MPP dependency. The
source-length analysis also needs the pinned XLM-R tokenizer, which it fetches
on first use unless it is already cached.

## Data preparation

Place the integrated `wmt26-grand.parquet` in the repository root. It contains
the official test set, converted human ground truth, Task 1 submissions, and all
three Task 1 baseline snapshots. The 64 public model cards are stored separately
in `model_cards/task1/`; they provide the reference-use policy and report
metadata. The raw Pearmut dump and annotation-conversion step are not needed
to run the five paper analyses.

The [parquet audit](docs/TASK1_PARQUET_AUDIT.md) records the comparison with
the legacy JSONL inputs before this migration.

## Reproduce the paper results

From the repository root, run the paper analyses without arguments:

```bash
python scripts/score_task1_submissions.py
python scripts/analyze_task1_reference_thinking_effects.py
python scripts/analyze_task1_reference_quality.py
python scripts/analyze_task1_score_statistics.py
python scripts/analyze_task1_gpt_sol_stop.py
```

These commands use the `task1_deduplicated` baseline snapshot, greedy MPP matching, and
the `default` run namespace. They write these artifacts below `outputs/task1`:

| Analysis | Output below `outputs/task1` |
| --- | --- |
| Eight priority-direction ranking | `scores/greedy/default/priority_ranking.tex` |
| Individual rankings for all 21 official directions | `scores/greedy/default/per_direction_rankings.tex` |
| Reference-use and thinking effects | `reference_thinking_effects/greedy/default/task1-reference-thinking-effect.tex` |
| Error-span statistics for Gemma 4 and Qwen 3.6 | `reference_thinking_effects/greedy/default/task1-reference-thinking-effect-spans.tex` |
| Reference-use MPP F-score figure | `reference_thinking_effects/greedy/default/task1-llm-reference-effect.pdf` |
| Reference type and quality figure | `reference_quality/greedy/default/plots/task1-reference-quality-effects.pdf` |
| Source-length quartile MPP F-score figure | `score_statistics/greedy/default/task1-length-effect.pdf` |
| GPT Sol medium versus xhigh | `gpt_sol_stop/greedy/default/gpt_sol_stop.tex` |

Each TeX or PDF artifact has a same-stem Markdown table that shows its results
without requiring a TeX or PDF viewer.

For each script's arguments and analysis details, see the
[Task 1 script index](docs/SCRIPTS.md).

All five scripts accept `--data-file` and `--model-cards-dir`. Use
`--baseline-snapshot task1` or `--baseline-snapshot
task1_prededup_and_enhy_fix` for a historical cohort, together with a distinct
`--run-specific-info` so its outputs and caches remain separate.
