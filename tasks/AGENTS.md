### Rules to ALWAYS apply when editing code

- Modularity and clarity are important: if in doubt, prefer modularity and clarity to compactness and efficiency (unless this forces you to make very bad design choices).
- Documentation is important: all classes, methods, and functions should have a high-level comment indicating their purpose, and clarifying input and ouptut. Also, code (under `./src` and `./scripts`) should always be consistent with the documentation (under `./docs`). Whenever you update the code, you should reflect the changes in the documentation, and vice versa.


### Task 1 rules

- Use Python 3.12 or newer. Install the project and its paper-analysis dependencies with `python -m pip install -e .`; a local virtual environment is recommended.
- Run the five paper scripts from the repository root without arguments. Their defaults use `wmt26-grand.parquet`, model cards in `model_cards/task1`, the `task1_deduplicated` baseline snapshot, greedy MPP, and the `default` run namespace.
- Use another baseline snapshot only when explicitly requested. When doing so, set both `--baseline-snapshot` and a distinct `--run-specific-info` to keep outputs and caches separate. Historical pairings are `task1` with `ammon-dedup` and `task1_prededup_and_enhy_fix` with `prededup`.


### Task 2 rules

- Keep direct Task 2 results and Task 1-to-Task 2 conversion results as separate ranking tracks. The conversion catalog defines which systems are included; do not infer the list from files merely being present in an input directory.
- Use the shared `human_annotations/ground-truth.jsonl` produced by `scripts/convert_annotation_dump.py`. Task 1 consumes `task1_pred` spans, while Task 2 consumes `task2_pred` holistic scores from the same retained screens.
- Run the three Task 2 paper scripts from the repository root as documented in `README_TASK2.md` and `docs/SCRIPTS.md`. Keep generated outputs and input data out of Git.
- Optional Task 2 significance checkpoints are tied to the input hashes, implementation version, and numerical environment. Resume only with the same arguments and inputs; use a new output directory for a changed run.
