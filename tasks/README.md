# Task code: Tasks 1, 2 and 3

This folder holds the organisers' code that scores the three tasks of the WMT26 shared task on
automated translation quality evaluation and computes the findings paper's tables and figures for
them. It comes from the organisers' code repository (`sted19/wmt26-mteval-task1`, commit
`a8b9b6f`), written by GitHub users `sted19`, `shuoyangd` and `prosho-97`:

| Task | Package | Guide |
| --- | --- | --- |
| 1: error spans | `src/wmt26_mteval_task1` | [`README_TASK1.md`](README_TASK1.md) |
| 2: segment scores | `src/wmt26_mteval_task2` | [`README_TASK2.md`](README_TASK2.md) |
| 3: error-free detection | `src/wmt26_mteval_task3` | [`README_TASK3.md`](README_TASK3.md) |

The per-task guides and `docs/` are the authors' own. This README runs all three tasks from the
Hugging Face release. Where the guides say otherwise about data (a raw annotation dump, or that
no Hugging Face repository exists yet), follow this README.

## Requirements

- Python 3.12 or newer, Git, and network access.
- A Hugging Face login with access to the dataset
  [`wmt/wmt26-mteval-data`](https://huggingface.co/datasets/wmt/wmt26-mteval-data):
  `hf auth login`, or a token in `HF_TOKEN`.
- About 14 GB of free disk.

All commands below run from this folder, `tasks/`.

## 1. Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## 2. Download the data

```bash
python ../scripts/download.py --config grand --out data/hub
ln -s data/hub/wmt26-grand.parquet wmt26-grand.parquet
shasum -a 256 wmt26-grand.parquet
python -m pytest -q
```

The code reads the dataset's `grand` config: a single Parquet file with the test set, the human
ground truth, every submission and every baseline snapshot. Its SHA-256 must be
`a61675552cebb68fb8303cf08a23fbec31e5bb4e0bd26a1fdee1a9bfafc2fb7a`; Task 3 refuses any other
file. `--revision` selects a Hub revision (default `main`). `data/` and `wmt26-grand.parquet` are
git-ignored.

## 3. Task 1

```bash
python scripts/score_task1_submissions.py
python scripts/analyze_task1_reference_thinking_effects.py
python scripts/analyze_task1_reference_quality.py
python scripts/analyze_task1_score_statistics.py
python scripts/analyze_task1_gpt_sol_stop.py
```

| Output below `outputs/task1/` | Paper |
| --- | --- |
| `scores/greedy/default/priority_ranking.tex` | Table 4 |
| `scores/greedy/default/per_direction_rankings.tex` | Tables 16–36 |
| `reference_thinking_effects/greedy/default/task1-reference-thinking-effect.tex` | Table 5 |
| `reference_thinking_effects/greedy/default/task1-reference-thinking-effect-spans.tex` | Table 6 |
| `gpt_sol_stop/greedy/default/gpt_sol_stop.tex` | Table 7 |
| `reference_thinking_effects/greedy/default/task1-llm-reference-effect.pdf` | Figure 1 |
| `reference_quality/greedy/default/plots/task1-reference-quality-effects.pdf` | Figure 2 |
| `score_statistics/greedy/default/task1-length-effect.pdf` | Figure 3 |

The scripts cache intermediate results in `outputs/task1/greedy/`. The authors' outputs are
committed in `outputs/task1/`, and `git checkout -- outputs/task1` restores them. Each TeX or PDF
file has a Markdown file with the same values next to it.

## 4. Task 2

```bash
python scripts/stage_task2_inputs.py
python scripts/score_task2_submissions.py
python scripts/analyze_task2_references.py
python scripts/analyze_task2_meta_metrics.py
python scripts/score_task2_submissions.py --track conversion \
  --task1-submissions-dir submissions/task1 \
  --task1-dedup-dir baselines/task1_deduplicated \
  --task1-original-dir baselines/task1_prededup_and_enhy_fix \
  --output-dir outputs/task2/conversion
```

The first command builds, from the grand parquet, the input folders that `README_TASK2.md`
expects (about 9 GB); it replaces that guide's download and annotation-conversion steps.
`python scripts/stage_task2_inputs.py --help` lists its options.

For the cluster columns, add `--permutations 1000` to both scoring commands and drop
`--output-dir`; the same files are then written to `outputs/task2/significance/direct_all/` and
`outputs/task2/significance/conversion_all/`. These permutation tests take many hours. They save
checkpoints, and the same command resumes an interrupted run.

| Output below `outputs/task2/` | Paper |
| --- | --- |
| `scores/direct_all/priority_ranking.tex` | Table 8 |
| `scores/direct_all/per_direction_rankings.tex` | Tables 37–57 |
| `conversion/priority_ranking.tex` | Table 9 |
| `meta_metric_validation/task2-meta-metric-correlations.tex` | Table 58 |
| `references/task2-reference-use-effects.tex` | Table 59 |
| `references/task2-main-reference-vs-pseudoref.tex` | Table 60 |
| `references/task2-from-scratch-vs-postedit.tex` | Table 61 |
| `references/task2-reference-score-behavior.tex` | Table 62 |
| `meta_metric_validation/task2-screen-vs-source-spa-support.pdf` | Figure 6 |

## 5. Task 3

```bash
python -m wmt26_mteval_task3.preparation
python -m wmt26_mteval_task3.pipeline
OPENBLAS_NUM_THREADS=1 python -m wmt26_mteval_task3.significance
python -m wmt26_mteval_task3.paper_tables
python -m wmt26_mteval_task3.threshold.cli
python -m wmt26_mteval_task3.reproduction verify outputs/task3
```

These write Tables 10–13 and 65–85 and Figures 4 and 5 below `outputs/task3/`. The last command
checks the outputs against the authors' frozen results.

## Cleaning up

The downloaded parquet, the staged inputs and the Task 1 caches can be deleted after a run; the
results stay in `outputs/`:

```bash
rm -rf wmt26-grand.parquet data test-set-files human_annotations submissions baselines \
  outputs/task1/greedy
```
