# Task 1 evaluator rankings

## What the script does

`score_task1_submissions.py` scores participant submissions and organizer
baselines against human error-span annotations. It reads each evaluator's
model-card metadata and predictions, aligns predictions with the official test set and
human ground truth, and computes greedy MPP precision, recall, and F1. Complete
direction coverage and the recorded reasoning-termination status determine
which evaluators enter each ranking.

The priority ranking averages each eligible evaluator's F1 equally across the
eight priority translation directions. The other table ranks evaluators within
each of the 21 official directions. Paired permutation tests group entries
into significance clusters; the default is 10,000 two-sided draws with seed 4.

## Inputs and outputs

The default data input is `wmt26-grand.parquet`, which includes the test set,
converted ground truth, submissions, and baseline predictions. Model-card
metadata is read from `model_cards/task1/`. The default baseline snapshot is
`task1_deduplicated`.

The script writes `priority_ranking.tex`, `priority_ranking.md`,
`per_direction_rankings.tex`, and `per_direction_rankings.md` beneath
`outputs/task1/scores/greedy/default/`. The Markdown tables show rank,
significance cluster, evaluator, precision, recall, and F1, along with
evaluator metadata. Reusable MPP and significance caches are stored beneath
`outputs/task1/greedy/default/`.

## Run

From the repository root, after the setup in [README_TASK1.md](../README_TASK1.md):

```bash
python scripts/score_task1_submissions.py
```

## Command-line options

Paths are relative to the repository root unless absolute. Repeat a selection
option to include more entries. Passing `--submission` or `--baseline` without
values selects no entries of that type.

| Option | Default | Effect |
| --- | --- | --- |
| `--data-file PATH` | `wmt26-grand.parquet` | Integrated data file. |
| `--model-cards-dir PATH` | `model_cards/task1` | Copied model-card root. |
| `--baseline-snapshot NAME` | `task1_deduplicated` | Select one of the three Task 1 snapshots. |
| `--submissions-dir PATH` | `submissions/task1` | Legacy logical participant path; normally unchanged. |
| `--submission [ID ...]` | All submissions | Select exact submission IDs. |
| `--baselines-dir PATH` | `baselines/task1_deduplicated` | Legacy snapshot alias; prefer `--baseline-snapshot`. |
| `--baseline [NAME ...]` | All baselines | Select exact baseline directory names. |
| `--test-set PATH` | Integrated test set | Legacy external JSONL override. |
| `--ground-truth PATH` | Integrated ground truth | Legacy external JSONL override. |
| `--model-card-template PATH` | None | Optional legacy card-validation template. |
| `--direction LANGUAGE_PAIR` | All official directions | Score a selected direction; repeat for several. |
| `--no-significance` | Off | Omit permutation tests and significance clusters. |
| `--significance-permutations N` | `10000` | Number of permutation draws. |
| `--significance-workers N` | `1` | Parallel evaluator-pair workers. |
| `--significance-seed N` | `4` | Nonnegative permutation seed. |
| `--significance-alpha VALUE` | `0.05` | Significance threshold. |
| `--significance-alternative {one-sided,two-sided}` | `two-sided` | Permutation-test alternative. |
| `--output-dir PATH` | `outputs/task1/scores` | Report root. |
| `--run-specific-info NAME` | `default` | Namespace beneath the matching-algorithm report and cache roots. |
| `--mpp-cache-root PATH` | `outputs/task1` | MPP and significance cache root. |
| `--recompute-mpp` | Off | Recompute MPP components. |
| `--recompute-significance` | Off | Refresh permutation results while reusing MPP components. |
| `--verbose` | Off | Show detailed progress. |
| `-h`, `--help` | — | Show CLI help. |
