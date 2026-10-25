# Release guide

How to build, check and publish the release. **Nothing in this guide has been run against a
remote.** The steps that upload, push or open a pull request are marked "Not run", and they wait
for an explicit OK (step 4).

Set these once. `GRAND` is only ever read; `OUT` must be missing or empty; `TMPDIR` is where
`check_loading.py` stages its links and the `datasets` cache, so it should be on the volume that
holds `OUT` and `GRAND`, with room for the cache (the full `submissions` and `baselines` configs
are about 22 million rows).

```bash
export GRAND=/path/to/wmt26-grand.parquet
export OUT=/path/to/build/hf
export TMPDIR=/path/to/a/volume/with/space
uv sync --dev
df -h "$OUT" "$TMPDIR"
```

## 1. Build

```bash
uv run python scripts/build_hf.py --grand "$GRAND" --out "$OUT"
```

The script checks the sha256 of the grand file and refuses any other file. It takes about a minute
and a half and about 2 GB of memory, and writes about 250 MB: `data/<config>/…parquet`, the dataset
card `README.md` (rendered from `hf/README.template.md` and the numbers it has just counted) and
`build_manifest.json`. If `OUT` already holds files it stops; delete the scratch folder and run
again. A build that stopped half-way has no `build_manifest.json` and must not be used.

## 2. Validate

```bash
uv run python scripts/validate_hf.py --out "$OUT" --grand "$GRAND" --expected tests/data/expected_full.json
```

It must print `0 failures` (about five minutes). It checks the row counts, Table 2 of the findings
paper, personal data, the schemas and a round trip of every run against the grand file. Then run
the tests and check that the generated documents are current:

```bash
OUT="$OUT" GRAND="$GRAND" uv run pytest -q -m fulldata
uv run pytest -q -m "not fulldata"
uv run ruff check
uv run python scripts/render_docs.py --grand "$GRAND"
git diff --exit-code docs/FORMAT.md data/MANIFEST.tsv
```

## 3. Loading matrix

Every config must load with `datasets` 2.18.0, 4.0.0 and 5.1.0. Run the three versions one after
the other, and look at the free disk space between them. The script links `OUT` and the grand file
into a temporary folder under `TMPDIR` (hard links, so there is no copy when they share a volume),
loads each config listed in the card's YAML and prints one line per config.

```bash
uv run --no-project --python 3.11 --with "datasets==5.1.0" --with "numpy>=2" \
  python scripts/check_loading.py "$OUT" 5.1.0 --grand "$GRAND"
uv run --no-project --python 3.11 --with "datasets==4.0.0" --with "numpy>=2" \
  python scripts/check_loading.py "$OUT" 4.0.0 --grand "$GRAND"
uv run --no-project --python 3.11 --with "datasets==2.18.0" --with "pyarrow<21" --with "numpy<2" \
  python scripts/check_loading.py "$OUT" 2.18.0 --grand "$GRAND"
```

- 2.18.0 needs `pyarrow<21` and `numpy<2`, because newer pyarrow wheels need NumPy 2.
- Each line is `OK VERSION CONFIG ROWS` or `FAIL VERSION CONFIG …`. The seven typed configs must
  be `OK` in all three runs, with the row counts of `tests/data/expected_full.json` (the script
  does not compare them). The exit status is 0 only then.
- `grand` is expected to fail on 2.18.0 and 4.0.0, which cannot read its JSON columns, and to load
  on 5.1.0 (and on any `datasets>=4.7.0`). That failure does not change the exit status.

## 4. Stop for Diptesh's OK

Stop here. Do not upload, push or post anything before Diptesh has read the result and said OK in
writing. What to put in front of him:

- the rendered card, `$OUT/README.md`, read from top to bottom, and `docs/ERRATA.md`;
- the output of steps 2 and 3;
- the open decisions: the licence (the card says `license: other` and "to be confirmed", and the
  repository has no licence file yet), General MT's agreement to the use of their data and
  annotations, the agreement of whoever else has write access to the Hub repository, and the
  links that teams volunteered in `systems.link`;
- that the unchanged `grand` config still carries the raw annotation records while the typed
  configs of v0.9 do not.

## 5. Upload to Hugging Face as a pull request (Not run)

```bash
uv run hf upload wmt/wmt26-mteval-data "$OUT" . --repo-type dataset --create-pr \
  --commit-message "Add the v0.9 typed configs and the dataset card"
```

`hf` is the command line of `huggingface_hub` 2.x, which `uv.lock` pins; the older
`huggingface-cli` is not installed in that environment, and `hf upload` takes the same arguments.
The command needs a token with write access to the `wmt` organisation (`uv run hf auth login`).
It uploads only what is in `OUT` and does not delete anything, so `wmt26-grand.parquet` stays as it
is. The pull request is opened on the Hub, where it is reviewed before it is merged. Making the
dataset public is a separate decision.

## 6. GitHub pull request (Not run)

```bash
git push -u origin release/v0.9
gh pr create --repo wmt-conference/wmt26-mteval --base main --head release/v0.9 \
  --title "Release v0.9: typed data configs and documentation" \
  --body "Adds the build, validation and loading tools, the dataset card template, the generated format documents, the errata and the release guide. See docs/CHANGELOG.md."
```

Merge the two pull requests once both have been reviewed, then tag the merge commit `v0.9`.
