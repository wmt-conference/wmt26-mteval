# Converting Pearmut annotations

## What the script does

`convert_annotation_dump.py` turns a raw Pearmut annotation dump into the
human ground-truth JSONL used by the Task 1 analyses. It matches annotations to
the official test set's item IDs, systems, and exact translation text, then
keeps the last applicable annotation in dump order for each annotator's
item-screen.
Pearmut reset markers remove that annotator's earlier work in the direction.
Duplicate identical screens are collapsed.

Pearmut spans use inclusive UTF-16 indices. The converter maps them to
half-open Unicode code-point spans (`start` inclusive, `end` exclusive),
preserves minor/major severity, and stores omission severity separately. An
instruction-fault marker becomes a null `instruction_fault` field in the
Task 1 assessment.

## Inputs and outputs

The command takes three positional paths:

1. The raw Pearmut dump, normally `annotations.json`.
2. The authoritative test set, `test-set-files/mteval-test26.jsonl`.
3. The destination path, normally `human_annotations/ground-truth.jsonl`.

The destination is a JSONL file with one row per annotated official item and a
`screens` array of its surviving human annotations. Each screen identifies its
annotator and three displayed systems; `task1_pred` contains their span
assessments. The shared source schema also preserves each system's holistic
score in `task2_pred`; Task 1 uses `task1_pred`. The converter prints a recap
of included screens and excluded non-official input. It validates the complete
result before writing the destination.

## Run

From the repository root, after the setup in [README_TASK1.md](../README_TASK1.md):

```bash
python scripts/convert_annotation_dump.py \
  annotations.json \
  test-set-files/mteval-test26.jsonl \
  human_annotations/ground-truth.jsonl
```

## Command-line arguments

The three paths are positional and required. By default, the converter keeps
dump directions also present in the official test set. A requested direction
must appear in both inputs.

| Argument | Effect |
| --- | --- |
| `annotations` | Pearmut dump JSON path. |
| `test_set` | Authoritative test-set JSONL path. |
| `destination` | Output ground-truth annotation JSONL path. |
| `--direction SRC-TGT` | Select one direction; repeat to select several. |
| `-h`, `--help` | Show CLI help. |
