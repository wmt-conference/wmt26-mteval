# Reproducing the WMT26 Task 2 paper results

Task 2 evaluates automatic segment-level quality scores against human holistic
scores. The official ranking equally averages screen-based PDP and source-based
SPA across the eight primary directions. Task 1-to-Task 2 conversions are a
separate comparison track and are not part of the direct-submission ranking.

## Setup and inputs

Use Python 3.12 or newer. From the repository root, install the project with
`python -m pip install -e .` (or create and activate a virtual environment
first). Numerical/plotting versions are pinned in the shared `pyproject.toml`
for Tasks 1–3. Download the Task 2 submissions, baselines, official test set, model-card
template, and raw Pearmut annotations from the data location published with the
release. Keep them in the paths below; they are intentionally not committed to
this repository.

```text
submissions/task2/                 participant Task 2 submissions
baselines/task2/                   direct Task 2 baselines
test-set-files/mteval-test26.jsonl official test set
test-set-files/model_card.json     model-card template
annotations.json                   raw Pearmut dump
human_annotations/ground-truth.jsonl  generated shared ground truth
```

Create the shared ground truth once, using the same converter as Task 1:

```bash
python scripts/convert_annotation_dump.py annotations.json \
  test-set-files/mteval-test26.jsonl human_annotations/ground-truth.jsonl
```

The generated JSONL retains screen identities, displayed systems, Task 1 span
annotations, and Task 2 holistic scores. Task 1 scoring consumes the span
annotations; Task 2 consumes the holistic scores. See
[`docs/GOLD_CONVERSION.md`](docs/GOLD_CONVERSION.md).

## Reproduce the Task 2 analyses

Run from the repository root. The default output is under `outputs/task2/`;
select another `--output-dir` if you want to preserve an earlier run.

```bash
python scripts/score_task2_submissions.py
python scripts/analyze_task2_references.py
python scripts/analyze_task2_meta_metrics.py
```

The scoring command writes the aggregate and per-direction paper tables in
LaTeX and Markdown, plus a machine-readable run manifest. The reference
analysis writes the compact LaTeX tables used in the paper. The meta-metric
analysis writes the screen/source grouping validation table and support figure.
Optional permutation significance is disabled by default; pass
`--permutations 1000` to the scoring script to compute and checkpoint it. The
same command can resume an interrupted run as long as its inputs and options
are unchanged.

To reproduce the separate Task 1-conversion ranking, provide the original Task
1 participant and baseline directories and use the conversion track:

```bash
python scripts/score_task2_submissions.py --track conversion \
  --task1-submissions-dir submissions/task1 \
  --task1-dedup-dir baselines/task1_deduplicated \
  --task1-original-dir baselines/task1_prededup_and_enhy_fix \
  --output-dir outputs/task2/conversion
```

The conversion catalog fixes the evaluator list used for the paper table.
Conversion results are kept separate from direct Task 2 results. Where both
Task 1 baseline variants were included in the paper comparison, the catalog
identifies the deduplicated and original-span versions separately.

For command options, input rules, metric definitions, and significance details,
see [`docs/SCRIPTS.md`](docs/SCRIPTS.md) and the Task 2 guides linked there.
