# Task 1-to-Task 2 conversion

This comparison track derives holistic Task 2 scores from Task 1 error spans.
It is reported separately from direct Task 2 submissions and baselines. The
fixed evaluator catalog in `src/wmt26_mteval_task2/evaluators.tsv` specifies
which systems are included in the paper results.

For each translation, the conversion starts at 100 points and subtracts 10
points per major error and 2.5 points per minor error, with a floor of zero.
A major error covering the complete non-empty translation, or the specified
single major error on an empty translation, gives a score of zero. Omission
annotations count as one error at their recorded severity. The implementation
is in `src/wmt26_mteval_task2/conversion.py`.

The scoring command applies this rule directly to the Task 1 prediction files;
it does not require a second set of converted JSONL files. Baseline snapshots
are labeled separately when both deduplicated and original-span versions are
included. See [Task 2 scoring](TASK2_SCORING.md) for the command and options.
