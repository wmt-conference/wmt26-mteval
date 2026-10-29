# Release guide

How to build, check and publish the Hugging Face release.

Set these once. `GRAND` is only read; `OUT` must be missing or empty; `TMPDIR` is where
`check_loading.py` stages its links and the `datasets` cache, so it should be on the volume that
holds `OUT` and `GRAND`, with room for the cache (the full `submissions` and `baselines` configs
are about 22 million rows).

```bash
export GRAND=/path/to/wmt26-grand.parquet
export OUT=/path/to/build/hf
export TMPDIR=/path/to/a/volume/with/space
uv sync --dev --locked
```

## 1. Build

```bash
uv run python scripts/build_hf.py --grand "$GRAND" --out "$OUT"
```

The script checks the sha256 of the grand file and refuses any other file. It writes
`data/<config>/…parquet`, the dataset card `README.md` (rendered from `hf/README.template.md` and
the numbers it has just counted) and `build_manifest.json`. A build that stopped half-way has no
`build_manifest.json` and must not be used. The manifest records `git_commit` and `git_dirty`:
build from a clean checkout of the commit you mean to release.

## 2. Validate

```bash
uv run python scripts/validate_hf.py --out "$OUT" --grand "$GRAND" --expected tests/data/expected_full.json
OUT="$OUT" GRAND="$GRAND" uv run pytest -q -m fulldata
uv run pytest -q -m "not fulldata"
uv run ruff check
uv run ruff format --check
uv run python scripts/render_docs.py --grand "$GRAND"
git diff --exit-code docs/FORMAT.md data/MANIFEST.tsv
```

`validate_hf.py` must print `0 failures`. It checks that the card and the manifest exist, the row
counts, Table 2 of the findings paper, personal data, the schemas, the keys that link the configs,
the texts of `items` and `pairs` against the grand file, that the Task 2 scores dropped from the
`alt` baselines are those of the official ones, and a round trip of every run against the grand
file.

## 3. Loading checks

Every config must load with `datasets` 2.18.0, 4.0.0 and 5.1.0:

```bash
uv run --no-project --python 3.11 --with "datasets==5.1.0" --with "numpy>=2" \
  python scripts/check_loading.py "$OUT" 5.1.0 --grand "$GRAND"
uv run --no-project --python 3.11 --with "datasets==4.0.0" --with "numpy>=2" \
  python scripts/check_loading.py "$OUT" 4.0.0 --grand "$GRAND"
uv run --no-project --python 3.11 --with "datasets==2.18.0" --with "pyarrow<21" --with "numpy<2" \
  python scripts/check_loading.py "$OUT" 2.18.0 --grand "$GRAND"
```

Each line is `OK VERSION CONFIG ROWS` or `FAIL VERSION CONFIG …`. The seven typed configs must be
`OK` in all three runs. `grand` needs `datasets>=4.7.0`, so it fails on 2.18.0 and 4.0.0; that
failure does not change the exit status.

## 4. Set the date

In `docs/CHANGELOG.md`, replace `YYYY-MM-DD` in the heading of the version with the release date
and, if the release is no longer a pre-release, change "pre-release" in that heading and in the
version line of `README.md`. Then check that no placeholder is left, and commit:

```bash
grep -rn "YYYY-MM-DD" README.md docs hf CITATION.cff --exclude=RELEASE.md    # must print nothing
git commit -am "docs: set the release date"
```

## 5. Publish

Upload the build to the Hub as a pull request, and open the matching GitHub pull request:

```bash
uv run hf upload wmt/wmt26-mteval-data "$OUT" . --repo-type dataset --create-pr \
  --commit-message "Release vX.Y"
gh pr create --repo wmt-conference/wmt26-mteval --base main
```

The upload adds and replaces files and deletes none. Merge both pull requests once they have been
reviewed, then tag the merge commit. Merge the GitHub pull request with a merge commit, not a
squash, so that the commit named by `git_commit` in `build_manifest.json` stays resolvable.
