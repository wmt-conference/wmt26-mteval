# Task 2 supporting analyses

`analyze_task2_references.py` reproduces the reference-use, reference-quality,
and score-scale tables used in the paper. Matched reference-based and
reference-free evaluators are compared on identical human-supported test
translations; designated reference candidates are excluded from the evaluation
set. The score table describes evaluator output scales and is not a ranking.

`analyze_task2_meta_metrics.py` reproduces the validation comparison between
screen-based and source-based groupings, including their per-direction ranking
correlations and the SPA support figure. These analyses are descriptive and do
not change the official ranking.

Both scripts default to the official direct Task 2 submissions and baselines,
use the eight primary language directions where specified, and write paper-ready
LaTeX/Markdown or PDF outputs below `outputs/task2/`. Their `run.json` manifests
record input hashes and numerical settings. Use `--help` to see paths and audit
options. They do not include Task 1-to-Task 2 conversions unless the analysis
specifically reports that separate track.
