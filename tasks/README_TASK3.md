# Reproducing the WMT26 Task 3 results

Task 3 code lives in `src/wmt26_mteval_task3`. Run its modules directly with
`python -m` after installation. The original evaluation, conversion, table and
threshold logic is reused, not reimplemented. Task 1/2 defaults are unchanged.

## Inputs

The data will eventually be distributed on Hugging Face; no repository URL is
available yet. For now, treat the supplied `wmt26-grand.parquet` as that download
and place it in the repository root. Supply the public model cards separately:

```text
wmt26-grand.parquet
model_cards/
  task1/submissions/task1/<ID>/model_card.json
  task2/submissions/task2/<ID>/model_card.json
  task2/baselines/task2/<ID-or-baseline-name>/model_card.json
  task3/submissions/task3/<ID>/model_card.json
```

The updated Task 1/2 cards include their Task 3 conversion permissions and
threshold fields. Task 3 uses the 13 unchanged public direct-submission cards.
Only the selected `model_card.json` files are read from these directories;
neighboring prediction files or annotations are never used. Keep all raw data
out of Git. `model_cards/task3/` should contain only the 13 direct Task 3
cards in the layout above, not predictions, annotations or Task 1/2 cards.

The parquet contains the test set, converted human ground truth, participant
predictions and historical baseline snapshots. No old ZIP, raw annotation dump,
private model cards, old analysis checkout or previous result files is needed.
The original pre-dedup/pre-eng–hye-fix Task 3 snapshot is selected explicitly from
within this parquet; Task 1/2 baseline defaults are not changed.

Input SHA-256:
`a61675552cebb68fb8303cf08a23fbec31e5bb4e0bd26a1fdee1a9bfafc2fb7a`.
The packaged `parquet_inputs.json` pins the physical parquet and reconstructed
source records independently of the output targets. A changed input snapshot
requires a new audit, not silently accepting another parquet file.

See [targets, input mapping, methods and provenance](docs/TASK3_REPRODUCTION.md).

## Setup

Use Python 3.12 or newer; the validated replay used Python 3.12.3.
The numerical/plotting versions are pinned in the shared `pyproject.toml` for
Tasks 1–3; no separate constraints file is needed. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q tests/test_task3*.py
```

## Prepare inputs

```bash
python -m wmt26_mteval_task3.preparation
```

Defaults are `--data-file wmt26-grand.parquet`, `--model-cards-dir model_cards`
and `--output-dir data/task3`. Preparation reconstructs JSON records using the
existing audited Task 1 parquet reader, preserves source line order, copies
selected public cards unchanged, and writes input provenance atomically.
It performs **no model-card repairs** and refuses an existing destination.

The resulting `data/task3/input_manifest.json` records the parquet identity,
logical source identities, reconstructed-file hashes and public-card hashes.
`metadata_repairs.json` is empty. Reconstructed JSON whitespace can differ from
archived JSONL bytes; both identities are recorded separately and validated.

If a complete prepared `data/task3/` already exists, skip preparation. Subsequent
commands need only this prepared directory, not the parquet or original card
locations. Otherwise use a new `--output-dir` and pass that path as `--data-dir`
to the analysis commands. Never merge new inputs into an old staging directory.

## Reproduce the results

Run from the repository root after preparation:

```bash
python -m wmt26_mteval_task3.pipeline
OPENBLAS_NUM_THREADS=1 python -m wmt26_mteval_task3.significance
python -m wmt26_mteval_task3.paper_tables
python -m wmt26_mteval_task3.threshold.cli
python -m wmt26_mteval_task3.reproduction verify outputs/task3
```

Each module supports `--help`. Outputs are under `outputs/task3/`; use a clean
output directory for a different run. `pipeline`, `significance` and
`paper_tables` accept `--output-dir`. For threshold analysis, use `--results-dir`
to select the scoring output root; its `--output-dir` instead selects the
threshold-analysis subdirectory. The verifier takes the output root positionally.

Every analysis validates staged hashes before calculation. Default significance
uses 100,000 source-segment permutations and seed `20260911`. Threshold plotting
covers all eight priority pairs; `--plot-language-pair` limits plots without
changing the numerical analysis. Threshold analysis generates CSV/Markdown
results, PDF plots and paper TeX; it does not invoke LaTeX compilation.

The checker requires **12 frozen numerical CSV targets**: five per-language
metric tables, human coverage, three macro rankings, recomputed significance
clusters, and two threshold tables. It also checks the generated baseline-delta
TeX table. Historical targets are verification oracles, never evaluation inputs.
The fresh parquet replay matches all 12 CSVs **byte-for-byte**, and the generated
baseline-delta TeX table also matches its original. No metadata repairs were
performed. A separate post-run comparison also found all **41 threshold plot
PDFs pixel-identical at 72 dpi** to the earlier replay; that comparison is not a
pipeline dependency. The full repository suite passes (376 tests plus seven
subtests). PDF bytes are not expected to match across renderers.

The validated run is preserved under `outputs/task3/parquet/`, separate from the
previous ZIP run. Its `reproduction_check.json` and `reproduction_evidence.json`
record all 13 checks and the source audit. The run used `--output-dir
outputs/task3/parquet` for scoring, significance and tables, `--results-dir
outputs/task3/parquet` for threshold analysis, and that same root for verification.
`strace` confirmed no reads of old data ZIPs, private workspaces, prior-run result
artifacts or scientific files neighboring the relocated Task 3 cards.

## Optional preview checks

With `pdflatex` (including the required LaTeX packages) and Poppler's `pdfimages`
installed separately from Python:

```bash
(cd outputs/task3/latex && pdflatex -interaction=nonstopmode -halt-on-error preview.tex)
(cd outputs/task3/threshold && pdflatex -interaction=nonstopmode -halt-on-error minimal_preview.tex)
pdfimages -list outputs/task3/threshold/minimal_preview.pdf
```

The threshold preview is ten vector-only pages. When importing paper figure
fragments from a custom output namespace, override their default figure path
before `\input`, for example:

```tex
\newcommand{\taskThresholdFigureDir}{outputs/task3/parquet/threshold/paper_figures}
```

Raw Pearmut conversion is not part of this parquet workflow: the parquet already
supplies the converted human reference. The raw dump is not in the parquet.

## Legacy ZIP support

An explicit positional ZIP still selects the earlier preparation path:

```bash
python -m wmt26_mteval_task3.preparation /path/to/wmt26-mteval-data.zip \
  --metadata-task1-dir /private/task1_bulk_submissions \
  --metadata-task2-dir /private/task2_bulk_submissions
```

Those private original-card roots were a temporary, explicitly authorized repair
for the original ZIP's missing conversion fields. They are **not used or needed
by parquet preparation**. The original ZIP remains untouched. See the historical
repair details in [the reproduction contract](docs/TASK3_REPRODUCTION.md).
