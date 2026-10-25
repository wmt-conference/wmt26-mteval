# WMT26 MT-Eval

Data and tools of the **WMT26 Shared Task on Automated Translation Quality Evaluation Systems**,
the shared task on MT metrics and quality estimation
([task page](https://www2.statmt.org/wmt26/mteval-task.html)). The task asked for three kinds of
prediction on one common test set (Task 1, error spans with severities; Task 2, a segment-level
quality score; Task 3, error-free detection) and took challenge sets as a fourth subtask. The
findings paper is [Lavie et al., 2026](https://www2.statmt.org/wmt26/pdf/2026.wmt-1.49.pdf).

This repository holds the test set, the instructions that participants received, the documentation
of the data and the tools that build and check the release. The data themselves are on Hugging
Face, as typed Parquet tables.

**Version: v0.9, a pre-release.** It contains the typed data configs and the documentation.
Human-annotation gold, converted predictions, results tables and the scoring code are planned for
v1.0. The release date is in [`docs/CHANGELOG.md`](docs/CHANGELOG.md) and the release steps are in
[`docs/RELEASE.md`](docs/RELEASE.md).

## Data on Hugging Face

The dataset is [`wmt/wmt26-mteval-data`](https://huggingface.co/datasets/wmt/wmt26-mteval-data).
Every config has one split, named `evaluation` (the data is one evaluation set, not partitioned
into train/test sets), and the typed configs load with `datasets>=2.18`:

```python
from datasets import load_dataset

pairs = load_dataset("wmt/wmt26-mteval-data", "pairs", split="evaluation")
```

| config | one row per |
| --- | --- |
| `items` | test item: source, reference and the columns parsed from `item_id` |
| `pairs` | item and MT system: the candidate translations (the default config) |
| `submissions` | participant run, item and system: the predictions of every submission |
| `baselines` | organiser baseline run, item and system |
| `systems` | evaluator run: name, organisation, citation, allow-listed model-card answers |
| `mt_systems` | MT system |
| `challenge_sets` | challenge set |
| `grand` | test item: the organisers' single file, unchanged (needs `datasets>=4.7.0`) |

[`docs/FORMAT.md`](docs/FORMAT.md) describes every column, and the dataset card explains how to
read the data and what to watch for. The places where the findings paper and the data differ are
listed in [`docs/ERRATA.md`](docs/ERRATA.md). To fetch one config only:

```bash
uv run python scripts/download.py --config pairs --out data/hf
```

The human evaluation that these predictions are measured against was carried out by the WMT26
General MT task, and its data are public in
[their release](https://github.com/wmt-conference/wmt26-general-mt/releases/tag/humeval).
The multimodal inputs (screenshots and videos) are in `multimodal_inputs.zip` in that release.

## What is in this repository

```text
README.md  CITATION.cff
data/
  mteval-test26.jsonl.gz        the evaluation set
  task1_domain_coverage_sets.json  the priority language pairs and coverage sets
  model_cards/                  scrubbed model-card answers of every official run
  metadata/                     curated tables: language codes, run names, Table 2, challenge sets
  MANIFEST.tsv                  the source files recorded in the grand file (generated)
docs/
  FORMAT.md                     columns and conventions of every config (generated)
  ERRATA.md                     where the findings paper and the data differ
  CHANGELOG.md  RELEASE.md      what changed, and how to release
hf/README.template.md           the template of the dataset card
instructions/baseline-prompts.md  the prompts of the LLM baselines (Appendix E of the paper)
annotation-instructions.txt  translation-instructions.txt  model_card.json
src/wmt26_mteval/               the package behind the build and the checks
scripts/                        build_hf, validate_hf, check_loading, render_docs, download,
                                scrub_model_cards
tests/
```

`annotation-instructions.txt` and `translation-instructions.txt` are General MT's instructions to
annotators and translators, and `model_card.json` is the form participants filled in. The model
cards were scrubbed with an allow-list (`scripts/scrub_model_cards.py`): the names and e-mail
addresses of the submitters and their answers about data collection never reach the release.
Large files are not kept in git; they live on Hugging Face.

## Build and check

The tools use [uv](https://docs.astral.sh/uv/). `GRAND` is the organisers' `wmt26-grand.parquet`
(read only; the scripts check its sha256), and `OUT` a missing or empty folder.

```bash
uv sync --dev --locked
uv run python scripts/build_hf.py --grand "$GRAND" --out "$OUT"
uv run python scripts/validate_hf.py --out "$OUT" --grand "$GRAND" --expected tests/data/expected_full.json
uv run --no-project --python 3.11 --with "datasets==5.1.0" --with "numpy>=2" \
  python scripts/check_loading.py "$OUT" 5.1.0 --grand "$GRAND"
uv run python scripts/render_docs.py --grand "$GRAND"
uv run pytest -q -m "not fulldata"
```

`build_hf.py` writes the Parquet files, the dataset card and a manifest of checksums;
`validate_hf.py` runs the gates (the card and the manifest, row counts, Table 2, personal data,
schemas, the keys that link the configs, the texts of `items` and `pairs`, the Task 2 scores of the
`alt` baselines, and a round trip of every run against the grand file); `check_loading.py` loads
every config with the installed `datasets`; `render_docs.py` regenerates `docs/FORMAT.md` and
`data/MANIFEST.tsv`. The full procedure, with the loading matrix for `datasets` 2.18.0, 4.0.0 and
5.1.0, is in [`docs/RELEASE.md`](docs/RELEASE.md).

## Citation

If you use this data, please cite the findings paper (also in [`CITATION.cff`](CITATION.cff)):

> Alon Lavie, Greg Hanneman, Stefano Perrella, Shuoyang Ding, Eleftherios Avramidis, Lorenzo
> Proietti, Chi-kiu Lo, Ammon Shurtz, Chrysoula Zerva, Archchana Sindhujan, Vilém Zouhar, Diptesh
> Kanojia, Frédéric Blain, Brian Thompson, Giorgos Filandrianos, Orfeas Menis Mastromichalakis,
> Tom Kocmi and Pranav Gupta. 2026. Findings of the WMT26 Shared Task on Automated Translation
> Quality Evaluation Systems: Compact Open Models are Competitive Quality Evaluators. In
> *Proceedings of the Eleventh Conference on Machine Translation (WMT)*, pages 931-1030.
> https://www2.statmt.org/wmt26/pdf/2026.wmt-1.49.pdf

The BibTeX entries of the findings paper, of General MT, cESA, the pseudo-reference pipeline and
the challenge sets are on the dataset card.

## Contact

The organisers: `wmt-qe-metrics-organizers@googlegroups.com`.

