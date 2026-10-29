# Source-length effect

## What the script does

`analyze_task1_score_statistics.py` examines how evaluator MPP F1 varies with
source length. It counts source tokens with the pinned XLM-R tokenizer, then
assigns annotated item-system pairs to source-length quartiles within each
translation direction. For each quartile, it averages each eligible
evaluator's direction-level F1 scores equally, then summarizes the resulting
27 evaluator scores with a median and interquartile range. The set includes
xCOMET baselines.

## Inputs and outputs

The script reads the official test set, converted human ground truth, and
evaluator predictions from `wmt26-grand.parquet`, plus metadata from
`model_cards/task1/`. It fetches the pinned XLM-R tokenizer on
first use unless the tokenizer is already cached locally.

It writes `outputs/task1/score_statistics/greedy/default/task1-length-effect.pdf`
and `task1-length-effect.md`. The Markdown table gives each quartile's
evaluator median, first and third quartiles, and evaluator count. Reusable MPP
components are stored under `outputs/task1/greedy/default/`.

## Run

From the repository root, after the setup in [README_TASK1.md](../README_TASK1.md):

```bash
python scripts/analyze_task1_score_statistics.py
```

## Command-line options

Paths are relative to the repository root unless absolute. Repeat a selection
option to include more entries. Passing `--submission` or `--baseline` without
values selects no entries of that type.

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
| `--evaluator-characteristics PATH` | None | Optional CSV of `entry_type,entry_id,reasoning` metadata. |
| `--require-complete-ground-truth` | Off | Require human annotations for every policy-scorable official item-system pair. |
| `--allow-partial-predictions` | Off | Include incomplete selected directions in the analysis. |
| `--output-dir PATH` | `outputs/task1/score_statistics` | Figure root. |
| `--run-specific-info NAME` | `default` | Namespace beneath the matching-algorithm report and cache roots. |
| `--mpp-cache-root PATH` | `outputs/task1` | MPP cache root. |
| `--recompute-mpp` | Off | Recompute MPP components. |
| `--verbose` | Off | Show detailed progress. |
| `-h`, `--help` | — | Show CLI help. |
