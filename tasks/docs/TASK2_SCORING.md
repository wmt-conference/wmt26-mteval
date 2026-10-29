# Task 2 scoring

`score_task2_submissions.py` reproduces the official Task 2 rankings. It reads
the official blind set, the shared converted annotation file, and either the
direct Task 2 submissions/baselines or the explicitly requested Task 1
conversion track.

## Metrics and aggregation

For each language direction, screen-based PDP is the Pearson correlation
between evaluator score differences and cESA score differences for distinct
translations displayed together on an annotation screen. Repeated annotations
of a co-screen pair are averaged before comparison. Source-based SPA compares
each evaluator’s system ordering against the human ordering over sources where
both systems have human scores; repeated human scores for one source/system are
averaged first. Its inner SPA computation uses 1,000 permutations with seed 4.
The direction score is the arithmetic mean of PDP and SPA. The eight-direction
paper ranking gives each primary direction equal weight.

The independent permutation significance procedure is optional. It uses
source-level exchange masks jointly across directions and both metrics, has
deterministic draw-indexed random streams, stores resumable checkpoints, and
reports one- and two-sided pairwise tests without multiplicity correction.
Use the two-sided clusters for the paper tables. Significance output and run
metadata are written beneath the selected output directory.

## Outputs and options

By default, the script writes `priority_ranking.tex` and
`per_direction_rankings.tex`, their Markdown counterparts, and `run.json` under
`outputs/task2/scores`. If significance is requested, it also writes resumable
SQLite checkpoints and the cluster columns in the same tables. Use
`--input-dir` more than once to specify direct submission and baseline roots.
Use `--track conversion` plus the Task 1 directories documented in
[`README_TASK2.md`](../README_TASK2.md) for the separate conversion ranking.

Run `python scripts/score_task2_submissions.py --help` for all supported
options. Detailed scoring definitions are in the paper and in the implementation
docstrings in `src/wmt26_mteval_task2/`.
