# Task 1 integrated-data audit

Before changing the Task 1 readers, the parquet manifest was compared against
the legacy inputs in `wmt26-mteval-task1_backup`. All 122 listed JSONL sources
matched their backup files by byte count and SHA-256. The Task 1 row audit then
reconstructed every record from parquet values and source provenance, compared
JSON values against the corresponding original line, and checked that each
source line appeared exactly once.

| Source | Files | Records | Result |
| --- | ---: | ---: | --- |
| Test set | 1 | 26,885 | Match |
| Converted ground truth | 1 | 7,250 | Match |
| Task 1 submissions | 17 | 197,695 | Match |
| Original Task 1 baselines | 16 | 354,615 | Match |
| Deduplicated Task 1 baselines | 16 | 354,615 | Match |
| Pre-dedup/pre-EnHy-fix Task 1 baselines | 15 | 270,897 | Match |

The 66 Task 1 sources contain 1,211,957 records in total. Four deduplicated
xCOMET sources reuse original-snapshot parquet columns; their source-specific
provenance and values matched as well. All 64 Task 1 model cards passed the
legacy model-card validation before being copied to `model_cards/task1/`.

The parquet does not contain model cards, the model-card template, or the raw
`annotations.json` dump. The cards remain necessary for reference-use decisions
and report metadata. Task 1 uses the converted ground truth already in the
parquet; raw annotation conversion is outside this reproduction phase.

At runtime, Task 1 reconstructs source order from parquet provenance. MPP
cache fingerprints include the physical parquet hash. Permutation draw seeds
use the audited logical source hashes so that repackaging identical data does
not change statistical results; the significance cache separately checks the
physical parquet hash and discards saved draws if it changes. The historical
cross-file comparison is an audit step, not part of each paper run.

## Reproduction check

All five Task 1 paper scripts ran from the parquet and copied cards with
isolated outputs. The fresh MPP cache computed 3,794,247 scored assessment
segments across 33 evaluator entries. The GPT Sol and reference/thinking
Markdown and TeX reports, plus the reference-quality and source-length
Markdown value tables, match the backup byte for byte. All three figure PDFs
render identically to the backup at 72 dpi. With significance disabled, both
ranking tables match the backup on every non-cluster field. The 5,422
evaluator-direction significance streams have identical support and MPP
component hashes in the backup and parquet runs.

The full default significance calculation then computed 216,880 fresh draw
chunks for 513 evaluator pairs, using 10,000 draws per direction. All four
final ranking Markdown and TeX files match the backup byte for byte, including
every significance cluster.
