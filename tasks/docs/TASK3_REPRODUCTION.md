# Task 3 reproduction contract

## Results first

Targets were frozen from the active `analysis/docs/RUNBOOK.md`, five current
result families, paper-ranking CSVs, and `experiments/task2_threshold*` before
porting code. Historical aggregates are **expected outputs only**, never score,
threshold, prediction or annotation inputs. The executable contract is
[`targets.json`](../src/wmt26_mteval_task3/targets.json).

The migrated code reuses the original numerical and reporting implementations
from analysis commit `91210d87e0aeda557b3e5a1a5cf8cd1f8c6ba546` and
significance-test commit `02d971ae0c063a0136a5d1cf672f625bb9e166fb`.
Input loading is adapted to the release layout; numerical methods are preserved.
`upstream_logic.json` records regression fingerprints without requiring either
old checkout.
Installation and replay commands are in [README_TASK3.md](../README_TASK3.md).

## Reproduction targets

### R0: shared human evidence

The historical bundled JSONL and raw-dump converter output have SHA-256
`6ac39baaeea0092090f04320d65abefeff959cdeac0798eb854bc492443d9c8c`.
Parquet embeds this logical source identity. Reconstructed staging bytes use
separate audited hashes because JSON whitespace can change; the human evidence
and population below must remain identical.

- 7,250 annotated items; 57,258 retained screens; 149,117 unique item-system
  keys; 171,774 human assessments; 21 language pairs.
- 52,785 error-free and 118,989 error assessments.
- Human error-free: holistic score `>= 85`, empty `errors`, null `omission`,
  and null `instruction_fault`.
- Every retained assessment has unit weight. Repeat a system prediction against
  all assessments for its key. Do not introduce majority voting, latest-record
  selection or Task 1's per-key `1/n` weighting.
- Official records contribute to metrics; challenge predictions remain outside
  result tables, with the original direct-submission validation preserved.

CSV target: `submissions/human_reference_summary.csv`, including every
per-language evidence/coverage count.

### R1: five separate evaluation families

| Directory under `outputs/task3/` | Selection/status target | Metric rows |
| --- | --- | ---: |
| `submissions/` | 13 direct Task 3 submissions | 174 |
| `task1_converted/` | 12 converted, 3 opted out; 15 retained | 163 |
| `task2_converted/` | 14 converted, 4 opted out, 9 missing threshold; 27 retained | 264 |
| `task1_baselines_converted/` | 9 configured reference-free baselines | 163 |
| `task2_baselines_converted/` | 9 configured reference-free baselines | 163 |

Targets: each family's `metrics_by_language.csv`, including confusion counts,
coverage, human-population counts, MCC, precision, recall and F1. Conversion
reports also retain declared permissions, statuses, threshold/inclusivity and
span diagnostics. Original code preserves zero-length predicted spans rather
than silently repairing them.

Task 1-to-3 cleanliness is based on annotations, not holistic scores. Task 2-to-3
uses the declared `>=` or `>` comparison. Missing permission is not opt-in.
Organizer baseline inclusion is explicitly configured; participant-style
permission does not gate those baselines.

### R2: merged paper rankings and breakdowns

| Artifact under `outputs/task3/latex/` | Target |
| --- | --- |
| `table1_official_ranking.csv` | 18 systems, priority-eight macro metrics; direct Task 3 + Task 1 baselines |
| `table2_official_ranking_per_pair.tex` | Same official pool plus unranked partial-coverage systems |
| `table3_priority_all_tracks.csv` | 34 complete-priority systems, four tracks |
| `table4_all_pairs.csv` | 25 systems covering all 21 pairs |
| `per_language/*.tex`, `table5_per_language.tex` | All 21 breakdown tables |
| `system_descriptions.tex` | Original direct-submission descriptions and connecting prose |

The active CSVs contain **34 and 25**, not the stale **39 and 30** mentioned in
parts of the old LaTeX README. Anchor: submission `870955` leads the official
ranking at macro MCC `0.325531`, followed by `870958` at `0.316323`.

Keep evaluation and paper eligibility distinct:

- Exclude reference-based conversions from merged paper tables.
- Exclude organizer Task 2 conversions from *all* merged paper tables for the
  original comparability reason, while still reproducing that family and its
  sensitivity analysis separately.
- A ranking averages only systems covering every pair in its scope.
- Macro means are unweighted; they are not pooled confusion counts.
- Competition ranks use unrounded MCC. Original display aliases, ensemble
  presentation, colors, partial-coverage treatment and captions are reused.

Numerical fingerprints cover identities, ranks, scope and metrics, not model
names or provenance/display columns. Reviewed model overrides remain a packaged
copy of the original public analysis configuration.

### R3: recomputed significance

Target: **77 rows** in `latex/significance_clusters.csv`, for tables 1, 3 and 4.
These are recomputed, not copied from historical CSVs.

The original `significance-test/CLEAN_CODE/src/scoring.py` permutation and greedy
clustering kernels are reused:

- 100,000 permutations; base seed `20260911`.
- Exchange two systems' predictions by **source segment**, using the same
  exchange across target-language occurrences of that source segment.
- Source cluster key: `(set, source_language, domain, document, segment)`.
- Two-sided absolute macro-MCC difference, add-one correction, the original
  `1e-15` extremeness tolerance, no multiplicity adjustment.
- Per-scope RNG seed: first eight SHA-256 bytes of `base_seed:ordered,pairs`.
- Priority analysis jointly tests all 34 systems, then clusters the official
  18-system subset and all-track pool using that same matrix.
- All-pairs analysis uses its 25 complete-coverage systems.
- Alpha `0.05`; original greedy clustering starts a new cluster when a candidate
  significantly differs from any system already in the current cluster.

The original system catalog/settings are packaged resources. They contain
identities, display metadata and analysis settings, **not computed clusters**.
Pairwise p-values and seeds are emitted in `latex/significance_analysis.json`.
Changing permutations/seed is supported for diagnostics but is not the frozen
paper replay. `OPENBLAS_NUM_THREADS=1` avoids severe oversubscription.

### R4: Task 2 threshold sensitivity

Targets under `outputs/task3/threshold/`:

- `threshold_summary.csv`: **21 systems = 12 participant conversions + 9
  organizer baselines**, all covering eight priority language pairs.
- `per_language_threshold_summary.csv`: **168 rows**, including every selected
  threshold, confusion matrix, MCC and comparison against declared/pooled rules.
- Original `REPORT.md`, summary figures, pooled/per-language curves, and compact
  paper grids. Paper/preview TeX wrappers are explicit packaged templates.
- `paper_figures/paper_baseline_conversion_deltas.tex`: the original saved layout,
  populated from `100 * (local-optimal Task 2 MCC - Task 1 MCC)`, using generated
  CSVs. Its original bytes are an additional presentation target; no saved
  numerical cells are copied into generated results.

Primary experiment independently optimizes one threshold per language on the
same labels, then macro-averages MCC. These are **post-hoc optimistic upper
bounds**, not official results or held-out performance estimates. The original
exact sweep tests every realizable score boundary without splitting ties;
selected rules use `>=`, with original representative boundaries/tie-breaking.

The comparison chooses one threshold by pooled MCC. Deterministic five-fold CV
keeps complete `item_id` groups together: first 64 bits of SHA-256(item_id)
modulo five. Held-out items use a threshold fit on the other folds.

| Track | Median local macro gain over declared | Median extra over pooled optimum | Median pooled CV gain |
| --- | ---: | ---: | ---: |
| Participants | +0.0964 | +0.0303 | +0.0630 |
| Organizer baselines | +0.0339 | +0.0318 | +0.0000 |

Conversions `876242`/`876244` are excluded from aggregates for incomplete
priority coverage. All 12 eligible participants remain in the main numerical
analysis. Only the paper gain figure excludes reference-based IDs `870257`,
`870554`, `871298`, `875949`, `876231`.

Paper grids select participants `867652`, `872328`, `876271` and baselines
`gemma4-reasoning`, `gpt-5.6-sol-medium`, `xcomet-xxl-no-reference`, for
`eng-arz`, `eng-ekk`, `zho-jpn`. Original score scales, curves, operating-point
markers, Task 1 reference lines and zero-true-positive-tail clipping are kept.
Clipping is visualization-only and does not remove optimization candidates.

## Current inputs: grand parquet and complete public cards

The current workflow uses the supplied `wmt26-grand.parquet` as a stand-in for
its future Hugging Face distribution; no repository URL has been announced.
SHA-256: `a61675552cebb68fb8303cf08a23fbec31e5bb4e0bd26a1fdee1a9bfafc2fb7a`.

All predictions, the test set and converted human evidence come exclusively
from this file. Complete public cards are separate inputs under
`model_cards/task{1,2,3}/`, mirroring the logical submission/baseline paths.
Preparation reads only selected cards, even if neighboring legacy data is
present. There is no automatic fallback to the old ZIP or old prepared tree.

`parquet_source.py` extends the existing Task 1 parquet reader to all task
namespaces; its original record-order reconstruction is reused unchanged.
The historical system catalog still selects the same pools and baseline
snapshots. All selected embedded logical source hashes must match the frozen
historical identities. `parquet_inputs.json` independently pins the physical
parquet, record counts, reconstructed bytes and logical identities for all 66
selected sources. This is an input audit, not a derivation from result targets.
JSON whitespace may change on reconstruction, so historical source hashes and
new staged-file hashes are recorded separately. No record values are repaired.

Schema-2 input manifests bind that audited source catalog and the copied public
cards. Validation compares reconstructed scientific-file hashes against the
packaged catalog, not merely against an editable run manifest. Direct-submission
validation uses the already-validated reconstructed prediction bytes; its
prediction checks and numerical evaluation loop are unchanged. The updated
Task 1/2 cards contain the needed fields, and the 13 Task 3 cards are the same
public cards as before. `metadata_repairs.json` is empty for this workflow.

The raw `annotations.json` dump is not included in parquet and is not needed.
The previous byte-identical shared-converter check remains a legacy ZIP audit;
it is not claimed to run from parquet alone.

## Historical ZIP source mapping

The earlier replay used only the following archive plus authorized missing-field
repairs. This mapping documents provenance, not a dependency of parquet replay.

Archive: `wmt26-mteval-data.zip`, 816,296,098 bytes; SHA-256
`6db0ac730133e3dd010a594a197dc3e5d8d2022db02741c4fa1d4b207f5301ca`.
It has 398 members and about 10.4 GB uncompressed.

| Input | ZIP resolution |
| --- | --- |
| Human reference | `human_annotations/ground-truth.jsonl`, historical byte identity |
| Official test set | `test-set-files/mteval-test26.jsonl`, SHA-256 `aa2e17eb...` |
| Raw dump | `annotations.json`, SHA-256 `9fb5f27b...`; optional extraction for converter replay |
| Direct Task 3 | 13 numeric-ID directories under `submissions/task3/` |
| Task 1 retained pool | 15 selected directories under `submissions/task1/`; extra IDs are not selected |
| Task 2 retained pool | 20 directories under `submissions/task2/`, plus 7 numeric IDs under `baselines/task2/` |
| Historical organizer predictions | 9 exact payloads under `baselines/task1_prededup_and_enhy_fix/*__pre_dedup_pre_enhy_fix/predictions.jsonl`, containing both task fields |
| Organizer cards | Matching named baseline cards under `baselines/task2/`, independent of prediction paths |

The seven relocated numeric Task 2 entries are `872013`, `872028`, `872029`,
`872030`, `872031`, `873001`, `873002`. Directory placement does not change the
historical retained pool. Explicit catalogs, not discovery from present files,
define every replay selection.

### Snapshot trap

Current and deduplicated baseline snapshots are not the historical Task 3
snapshot. Four Gemma/Qwen variants differ on `eng-hye`. For example, historical
Gemma reasoning Task 1 counts are `TP=1026, FP=5168, FN=67, TN=450`; current
counts are `TP=657, FP=1378, FN=436, TN=4240`. This is scientific drift, not just
serialization.

All nine historical raw prediction hashes match the older variants **already
inside this same ZIP**. The Task 3 loader deliberately selects those variants.
This is not a change to Task 1/2 paper defaults. A corrected Task 3 analysis
would require a separate namespace and new explicitly reviewed targets.

## Temporary, authorized metadata repair

The supplied archive removed Task 1-to-3 permission from all 15 retained Task 1
participant cards, and Task 2-to-3 permission/threshold from all 20 cards under
`submissions/task2/`. The seven relocated Task 2 cards retain original fields.
Surviving inclusivity is insufficient to reconstruct the declared threshold.

The user authorized using local originals **only to repair those missing
model-card entries**, pending discussion with the archive author. Preparation:

1. Requires explicit local card roots; no implicit workspace paths or fallback.
2. Verifies each original card against its hash in the historical aggregate
   manifest before reading repair values.
3. Restores only absent permission/threshold entries. It never overwrites false
   opt-ins, blank thresholds, existing answers or prediction bytes.
4. Copies no personal/contact questions from originals. It does not read local
   predictions or another annotation dump.
5. Writes the repaired staged cards, exact repair rows and original-card hashes
   to `data/task3/metadata_repairs.json`/`input_manifest.json`.
6. Leaves the source ZIP untouched. Scoring, significance and threshold analysis
   thereafter need only the prepared directory, never the local originals.

**Legacy ZIP limitation:** the original ZIP alone is not sufficient until its
missing fields are restored by the archive author. This limitation is not
hidden by a hard-coded threshold catalog. The repair sidecar provides the
non-personal values needed for archive review. Once repaired, preparation runs
without local metadata roots. `--allow-repacked-dump` permits changed ZIP
packaging/cards while still enforcing every frozen prediction/reference hash.

Direct cards are already sanitized; the original private ZIP-wrapper/consent
schema is therefore not a release validation requirement. Release loaders
validate payload hashes and necessary card structure instead. ZIP/member and
staged-file provenance replaces the old private parser-checkout audit. No
restricted-evaluation package or private parser checkout is needed.

## Provenance, verification and validation performed

- `python -m wmt26_mteval_task3.preparation` defaults to parquet and complete
  public cards. An explicit positional ZIP selects legacy preparation. Both stage
  atomically and refuse existing destinations. `validate_inputs()` checks staged
  hashes and frozen scientific identities before analysis.
- Generated `run_manifest.json` binds implementation/resources, numerical
  environment, input-manifest identity and generated outputs. Logs and mutable
  checker reports are not treated as result artifacts.
- `python -m wmt26_mteval_task3.reproduction audit ZIP` remains an archive-only feasibility
  check. Its status 1 is expected for the original incomplete archive; it never
  consults local original cards or prints card answers.
- `python -m wmt26_mteval_task3.reproduction verify OUTPUT_ROOT` requires all 12 CSV targets
  plus the generated baseline-delta TeX presentation target. It sorts CSVs by
  stable keys, rejects duplicates/missing columns and hashes selected strings as
  compact JSON. It separately reports historical CSV byte identity. Numerical
  precision is strict; do not regenerate targets to bless changed outputs.
- Synthetic tests cover repair, atomic failure, no fallback, file integrity,
  original validation/conversion behavior, exact sweeps, CV and plotting rules.
  `upstream_logic.json` additionally guards 168 unchanged upstream definitions
  via AST fingerprints that ignore documentation changes.

Validated grand-parquet replay (`outputs/task3/parquet/`):

- Fresh preparation reconstructed all **66 scientific sources**, copied **64
  public cards**, and performed **zero repairs**. The 130 staged input files are
  bound to the independently pinned parquet input catalog.
- Fresh five-track scoring, 100,000-permutation significance, paper-table
  rendering and threshold analysis pass all **13 verification checks**. All
  **12 scientific CSVs** and the generated baseline-delta TeX table are
  byte-identical to their historical originals.
- Both LaTeX previews compile: paper tables **15 pages**, threshold **10 pages**.
  The checked threshold preview and both 3x3 grids contain no raster images.
  A separate post-run Poppler comparison finds all **41 threshold plots
  pixel-identical at 72 dpi** to the earlier replay. Prior results are read only
  for that optional verification, never as computation inputs.
- `strace` found no reads of old data ZIPs, private Codabench/analysis/restricted
  workspaces, previous-run results or scientific data next to the moved cards.
  Only the 13 direct `model_card.json` files were read from `model_cards/task3/`.
- The complete repository suite passes: **376 tests plus seven subtests**.
  `reproduction_evidence.json` records the run's data and verification summary.

Earlier legacy ZIP validation:

- All **76 raw/shared/selected-payload input checks** pass on the original ZIP.
- All **12 scientific CSV targets are byte-identical** after authorized repair;
  the generated baseline-delta TeX table also matches its saved original.
- Shared converter regeneration is byte-identical to the bundled reference.
- The paper-table preview and ten-page threshold preview compile with
  `pdflatex`; checked threshold PDFs contain no raster images.
- Both 3x3 paper grids were visually inspected for columns, labels and clipping.
- An isolated wheel-based replay, with only a copied prepared input tree, also
  reproduces all 12 CSVs byte-for-byte. `strace` confirmed no accesses to the old
  analysis checkout, Codabench data, other PEAR-MUT dumps or restricted packages.
  It did not import Task 3 code from the editable release checkout.
- The complete repository test suite passes with the official test-set path
  pointing to the copy extracted from this same ZIP.

PDF bytes depend on renderer versions/timestamps. Validate numeric sources,
selected systems/pairs, vector rendering and visual content rather than PDF-byte
identity. Preview compilation requires TeX; scoring/plotting requires no GPU.

## Publication and scope

Keep raw dumps, canonical human records, raw/completed cards, predictions,
metadata repairs, annotator-level data, environments, outputs, shell histories
and private filenames out of Git. Review aggregate display names and release
authorization. Hashes provide provenance, not confidentiality.

Historical WMT24 plots, July-dump explorations, GPU regeneration of submitted
models, TeX build logs and redundant previews are not this current scientific
snapshot. Their original inputs are not all in the ZIP. They are not silently
claimed reproducible by copying old exploratory files into the release.
