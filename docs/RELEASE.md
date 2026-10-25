# Release guide

How to build, check and publish the release. The steps that read the Hub repository, upload, push
or open a pull request are marked "Not run": no tool or test in this repository runs them. They
wait for an explicit OK (step 4), and whoever makes the release runs them by hand.

Set these once. `GRAND` is only ever read; `OUT` must be missing or empty; `TMPDIR` is where
`check_loading.py` stages its links and the `datasets` cache, so it should be on the volume that
holds `OUT` and `GRAND`, with room for the cache (the full `submissions` and `baselines` configs
are about 22 million rows).

```bash
export GRAND=/path/to/wmt26-grand.parquet
export OUT=/path/to/build/hf
export TMPDIR=/path/to/a/volume/with/space
uv sync --dev --locked
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

`uv sync --locked` makes the environment the one in `uv.lock`, so the build and the checks run on
the pinned versions; a stale lock stops it instead of being re-resolved quietly. The manifest
records `git_commit` and `git_dirty`: build from a clean checkout of the commit you mean to
release, and keep that commit reachable (see step 8).

## 2. Validate

```bash
uv run python scripts/validate_hf.py --out "$OUT" --grand "$GRAND" --expected tests/data/expected_full.json
```

It must print `0 failures` (about six minutes). It checks that the card and the manifest exist, the
row counts, Table 2 of the findings paper, personal data, the schemas, the keys that link the
configs (every prediction is for a pair of `pairs` and a run of `systems`), the texts of `items`
and `pairs` against the grand file, that the Task 2 scores dropped from the `alt` baselines are
those of the official ones, and a round trip of every run against the grand file. Then run the
tests and check that the generated documents are current:

```bash
OUT="$OUT" GRAND="$GRAND" uv run pytest -q -m fulldata
uv run pytest -q -m "not fulldata"
uv run ruff check
uv run ruff format --check
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
- the open decisions: General MT's agreement to the use of their data and
  annotations, the agreement of whoever else has write access to the Hub repository, the links
  that teams volunteered in `systems.link`, the `ref_type` `unknown` (below) and the Hub card
  (below);
- `ref_type` `unknown` (Ruling R23): the 6,407 items whose reference text is present but whose
  type the source data records as `undefined` get the new value `unknown`, not `none`. That
  overrides decision A3 (the reference types `native`, `postedit`, `pseudo` and `none`), whose
  premise was that `undefined` meant no reference. It needs his OK, or the mapping goes back
  to `none` (`ref_type_orig` keeps `undefined` either way);
- the Hub card: the current `README.md` and file list of `wmt/wmt26-mteval-data` were never seen,
  because the token used for the preparation had no access to the `wmt` organisation, so whatever
  a colleague wrote there is unknown (step 6);
- that the unchanged `grand` config still carries the raw annotation records while the typed
  configs of v0.9 do not.

## 5. Set the date and status

Do this after the OK and before anything is uploaded or pushed, so that the published text is true.
In `docs/CHANGELOG.md`, replace `YYYY-MM-DD` in the heading of the version with the release date
and, if the release is no longer a pre-release, change "pre-release" in that heading and in the
version line of `README.md`. Then check that no placeholder is left, and commit:

```bash
grep -rn "YYYY-MM-DD" README.md docs hf CITATION.cff --exclude=RELEASE.md    # must print nothing
git commit -am "docs: set the release date"
```

## 6. Look at the Hub repository first (Not run)

The current card and file list of `wmt/wmt26-mteval-data` were never seen while this release was
prepared, because the token used had no access to the `wmt` organisation. The upload in step 7
adds files and deletes none, but its `README.md` replaces the card on the Hub, and a colleague may
have written text there. So, with a token that has read access to the organisation:

```bash
uv run hf auth whoami
uv run hf datasets ls wmt/wmt26-mteval-data -R
uv run hf download wmt/wmt26-mteval-data README.md --repo-type dataset --local-dir "$TMPDIR/hub-card"
diff "$TMPDIR/hub-card/README.md" "$OUT/README.md"
```

- Confirm that `wmt26-grand.parquet` is at the root of the repository, which is where the `grand`
  config of the card reads it, and that nothing else there clashes with `data/` or `README.md`.
- Read the whole difference between the two cards. Carry over any of the colleague's text that is
  still true, by editing `hf/README.template.md` (never `$OUT/README.md` by hand), and commit it.
  Then rebuild and check again (steps 1 to 3), because the card is rendered at build time and
  `build_manifest.json` has to describe what is uploaded.
- A file that is on the Hub and not in `$OUT` stays there after the upload. Say which files those
  are when you put the result in front of him (step 4), and let him decide whether they belong.

## 7. Upload to Hugging Face as a pull request (Not run)

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

## 8. GitHub pull request (Not run)

```bash
git push -u origin release/v0.9
gh pr create --repo wmt-conference/wmt26-mteval --base main --head release/v0.9 \
  --title "Release v0.9: typed data configs and documentation" \
  --body "Adds the build, validation and loading tools, the dataset card template, the generated format documents, the errata and the release guide. See docs/CHANGELOG.md."
```

Merge the two pull requests once both have been reviewed, then tag the merge commit `v0.9`.

`build_manifest.json` names the commit of the build in `git_commit`, and the dataset card and the
manifest are only reproducible from that commit. Merge the GitHub pull request with a merge
commit, not a squash, so that the commit stays on `main` and `git_commit` stays resolvable. If it
has to be squashed, tag the build commit first (`git tag v0.9-build <git_commit>`) and push the
tag with the merge. Do not rebase the branch after the build either.
