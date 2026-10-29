# Reference type and quality

## What the script does

`analyze_task1_reference_quality.py` compares how reference-based
auto-evaluators score the same MT output when given different reference
translations. It matches official test items with their RefQuality variants,
checks that the MT system and evaluated translation are the same, and computes
greedy MPP under each reference condition. The paper figure reports the mean
change in F1 across eligible Gemma 4 and Qwen 3.6 baselines for seven
reference-variant comparisons. Its Markdown table also identifies each
reference type and its cESA quality score.

## Inputs and outputs

The script reads RefQuality and official rows, evaluator predictions, and
converted ground truth from `wmt26-grand.parquet`. It reads evaluator metadata
from `model_cards/task1/`.

It writes
`outputs/task1/reference_quality/greedy/default/plots/task1-reference-quality-effects.pdf`
and a same-stem Markdown table. The table's ΔF1 is the score with reference B
minus the score with reference A. Reusable MPP components are stored under
`outputs/task1/greedy/default/`.

## Run

From the repository root, after the setup in [README_TASK1.md](../README_TASK1.md):

```bash
python scripts/analyze_task1_reference_quality.py
```

## Command-line options

Paths are relative to the repository root unless absolute. Repeat a selection
option to include more entries. Passing `--submission` or `--baseline` without
values selects no entries of that type; the paper figure requires its matched
Gemma 4 and Qwen 3.6 baselines.

| Option | Default | Effect |
| --- | --- | --- |
| `--data-file PATH` | `wmt26-grand.parquet` | Integrated data file. |
| `--model-cards-dir PATH` | `model_cards/task1` | Copied model-card root. |
| `--baseline-snapshot NAME` | `task1_deduplicated` | Select a Task 1 baseline snapshot. |
| `--submissions-dir PATH` | `submissions/task1` | Legacy logical participant path. |
| `--submission [ID ...]` | All submissions | Select exact submission IDs. |
| `--exclude-submission ID` | None | Exclude one submission ID; repeat as needed. |
| `--exclude-file PATH` | None | Read exclusions from a JSON object with an optional `submissions` ID array. |
| `--baselines-dir PATH` | `baselines/task1_deduplicated` | Legacy snapshot alias. |
| `--baseline [NAME ...]` | All baselines | Select exact baseline directory names. |
| `--test-set PATH` | Integrated test set | Legacy external JSONL override. |
| `--ground-truth PATH` | Integrated ground truth | Legacy external JSONL override. |
| `--model-card-template PATH` | None | Optional legacy card-validation template. |
| `--require-complete-ground-truth` | Off | Require human annotations for every policy-scorable official item-system pair. |
| `--output-dir PATH` | `outputs/task1/reference_quality` | Figure root. |
| `--run-specific-info NAME` | `default` | Namespace beneath the matching-algorithm report and cache roots. |
| `--mpp-cache-root PATH` | `outputs/task1` | MPP cache root. |
| `--recompute-mpp` | Off | Recompute MPP components. |
| `--verbose` | Off | Show detailed progress. |
| `-h`, `--help` | — | Show CLI help. |
