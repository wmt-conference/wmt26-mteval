# Reference-use and thinking effects

## What the script does

`analyze_task1_reference_thinking_effects.py` compares matched organizer
baselines on the eight priority directions. Within a model family, a
reference-use pair changes whether the evaluator sees a reference translation;
a thinking pair changes whether it uses reasoning. The script calculates
greedy MPP for each condition and reports the change in F1. It also compares
prediction behavior: errors per prediction, mean error-span length, and mean
severity weight.

The reference-use and thinking tables include the matched Gemma 4 and Qwen
3.6 conditions; the reference-use F1 table also includes xCOMET XL and XXL.
The direction-level reference-use figure summarizes Gemma 4 and Qwen 3.6.

## Inputs and outputs

The script reads matched baselines, the official test set, and converted human
ground truth from `wmt26-grand.parquet`. It reads baseline metadata from
`model_cards/task1/`. Beneath
`outputs/task1/reference_thinking_effects/greedy/default/` it writes:

| File | Content |
| --- | --- |
| `task1-reference-thinking-effect.tex` and `.md` | Reference-use and thinking F1 comparisons. |
| `task1-reference-thinking-effect-spans.tex` and `.md` | Prediction-behavior comparisons. |
| `task1-llm-reference-effect.pdf` and `.md` | Mean reference-use F1 change by priority direction. |

In the behavior table, errors per prediction include omissions, span length
uses non-omission spans, and severity weight includes omissions. The MPP cache
is under `outputs/task1/greedy/default/`.

## Run

From the repository root, after the setup in [README_TASK1.md](../README_TASK1.md):

```bash
python scripts/analyze_task1_reference_thinking_effects.py
```

## Command-line options

Paths are relative to the repository root unless absolute. The script uses
greedy MPP. A baseline selection must contain both members of each comparison
you want to calculate.

| Option | Default | Effect |
| --- | --- | --- |
| `--data-file PATH` | `wmt26-grand.parquet` | Integrated data file. |
| `--model-cards-dir PATH` | `model_cards/task1` | Copied model-card root. |
| `--baseline-snapshot NAME` | `task1_deduplicated` | Select a Task 1 baseline snapshot. |
| `--baselines-dir PATH` | `baselines/task1_deduplicated` | Legacy snapshot alias. |
| `--baseline [NAME ...]` | All baselines | Select exact baseline directory names; keep matched pairs together. |
| `--test-set PATH` | Integrated test set | Legacy external JSONL override. |
| `--ground-truth PATH` | Integrated ground truth | Legacy external JSONL override. |
| `--model-card-template PATH` | None | Optional legacy card-validation template. |
| `--require-complete-ground-truth` | Off | Require human annotations for every policy-scorable official item-system pair. |
| `--output-dir PATH` | `outputs/task1/reference_thinking_effects` | Report root. |
| `--run-specific-info NAME` | `default` | Namespace beneath the matching-algorithm report and cache roots. |
| `--mpp-cache-root PATH` | `outputs/task1` | MPP cache root. |
| `--recompute-mpp` | Off | Recompute MPP components. |
| `--verbose` | Off | Show detailed progress. |
| `-h`, `--help` | — | Show CLI help. |
