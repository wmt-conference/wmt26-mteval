# GPT Sol medium versus xhigh

## What the script does

`analyze_task1_gpt_sol_stop.py` compares the GPT-5.6 Sol medium- and
xhigh-thinking baselines on the same human-annotated item-system pairs. It
retains pairs for which both models completed reasoning, then pools their MPP
components across directions to calculate micro precision, recall, and F1.
It also summarizes errors per prediction, mean span length, and mean severity
weight. The final row reports xhigh minus medium for each measure.

## Inputs and outputs

The script reads the two GPT Sol baselines, official test set, and converted
human ground truth from `wmt26-grand.parquet`, plus the baseline model cards
from `model_cards/task1/`. It writes
`outputs/task1/gpt_sol_stop/greedy/default/gpt_sol_stop.tex` and a same-stem
Markdown table. Errors per prediction include omissions; mean span length
uses non-omission spans; mean severity weight includes omissions.

## Run

From the repository root, after the setup in [README_TASK1.md](../README_TASK1.md):

```bash
python scripts/analyze_task1_gpt_sol_stop.py
```

## Command-line options

Paths are relative to the repository root unless absolute. The script uses
greedy MPP and selects the GPT Sol medium/xhigh pair from the baseline directory.

| Option | Default | Effect |
| --- | --- | --- |
| `--data-file PATH` | `wmt26-grand.parquet` | Integrated data file. |
| `--model-cards-dir PATH` | `model_cards/task1` | Copied model-card root. |
| `--baseline-snapshot NAME` | `task1_deduplicated` | Select a Task 1 baseline snapshot. |
| `--baselines-dir PATH` | `baselines/task1_deduplicated` | Legacy snapshot alias. |
| `--test-set PATH` | Integrated test set | Legacy external JSONL override. |
| `--ground-truth PATH` | Integrated ground truth | Legacy external JSONL override. |
| `--model-card-template PATH` | None | Optional legacy card-validation template. |
| `--require-complete-ground-truth` | Off | Require human annotations for every policy-scorable official item-system pair. |
| `--output-dir PATH` | `outputs/task1/gpt_sol_stop` | Report root. |
| `--run-specific-info NAME` | `default` | Namespace beneath `<output-dir>/greedy/`. |
| `--verbose` | Off | Show detailed progress. |
| `-h`, `--help` | — | Show CLI help. |
