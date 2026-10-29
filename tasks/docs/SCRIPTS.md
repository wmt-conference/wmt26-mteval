# Paper analysis script guide

[README_TASK1.md](../README_TASK1.md) covers installation, the integrated
parquet input, copied model cards, and paper reproduction commands. The five
Task 1 paper scripts default to `wmt26-grand.parquet`, `model_cards/task1`, and
the deduplicated baseline snapshot. They also accept `--data-file`,
`--model-cards-dir`, and `--baseline-snapshot`. The guides below explain each
script's calculation, outputs, and remaining command-line options.

| Script | Result | Guide |
| --- | --- | --- |
| `convert_annotation_dump.py` | Legacy preparation of human ground-truth JSONL; not required for parquet reproduction | [Ground-truth conversion](GOLD_CONVERSION.md) |
| `score_task1_submissions.py` | Priority and per-direction evaluator rankings | [Scoring](TASK1_SCORING.md) |
| `analyze_task1_reference_thinking_effects.py` | Reference-use, thinking, and prediction-behavior comparisons | [Reference and thinking effects](TASK1_REFERENCE_THINKING_EFFECTS.md) |
| `analyze_task1_reference_quality.py` | Reference-type and quality comparison figure | [Reference quality](TASK1_REFERENCE_QUALITY.md) |
| `analyze_task1_score_statistics.py` | Source-length quartile figure | [Source-length effect](TASK1_SCORE_STATISTICS.md) |
| `analyze_task1_gpt_sol_stop.py` | GPT Sol medium-versus-xhigh table | [GPT Sol comparison](TASK1_GPT_SOL_STOP.md) |

## Task 2

Task 2 commands share the annotation-conversion output documented in
[Ground-truth conversion](GOLD_CONVERSION.md). The full setup and default
paper-reproduction commands are in [README_TASK2.md](../README_TASK2.md).

| Script | Result | Guide |
| --- | --- | --- |
| `score_task2_submissions.py` | Direct or converted Task 2 ranking, with optional resumable significance | [Task 2 scoring](TASK2_SCORING.md) |
| `analyze_task2_references.py` | Matched reference-use, reference-quality, and score-scale tables | [Task 2 analyses](TASK2_ANALYSES.md) |
| `analyze_task2_meta_metrics.py` | Screen/source meta-metric validation and SPA support | [Task 2 analyses](TASK2_ANALYSES.md) |

The conversion method and reproduction command are described in
[Task 1-to-Task 2 conversion](TASK2_CONVERSION.md).

## Task 3

[README_TASK3.md](../README_TASK3.md) gives the complete replay commands and the
grand parquet and complete public-card inputs. Legacy ZIP repair is documented
separately. The original analysis logic is reused; see
[reproduction targets, methods and provenance](TASK3_REPRODUCTION.md).

Task 3 runs installed package modules directly, without launcher scripts. Each
command supports `--help`.

| Command | Result | Guide |
| --- | --- | --- |
| `python -m wmt26_mteval_task3.preparation` | Selected parquet records, complete public cards and input provenance; optional positional ZIP selects legacy preparation | [Task 3 reproduction](TASK3_REPRODUCTION.md) |
| `python -m wmt26_mteval_task3.pipeline` | Five separate evaluation/conversion families | [Task 3 reproduction](TASK3_REPRODUCTION.md) |
| `python -m wmt26_mteval_task3.significance` | Original 100,000-permutation tests and recomputed paper clusters | [Task 3 reproduction](TASK3_REPRODUCTION.md) |
| `python -m wmt26_mteval_task3.paper_tables` | Original merged paper rankings, per-language tables and descriptions | [Task 3 reproduction](TASK3_REPRODUCTION.md) |
| `python -m wmt26_mteval_task3.threshold.cli` | Original Task 2-to-3 post-hoc threshold sensitivity, CV and vector figures | [Task 3 reproduction](TASK3_REPRODUCTION.md) |
| `python -m wmt26_mteval_task3.reproduction audit ZIP` | Archive-only feasibility/metadata audit | [Task 3 reproduction](TASK3_REPRODUCTION.md) |
| `python -m wmt26_mteval_task3.reproduction verify OUTPUT_ROOT` | Check 12 frozen scientific CSVs and the derived baseline-delta TeX table | [Task 3 reproduction](TASK3_REPRODUCTION.md) |
