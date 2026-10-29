# Task 1: reference use and thinking effects

## Reference use

| Auto-evaluator | Ref-free F1 | Ref-based F1 | ΔF1 |
| --- | --- | --- | --- |
| Gemma 4 [thinking] | 31.72 | 32.58 | +0.86 |
| Qwen 3.6 [thinking] | 29.77 | 29.38 | -0.39 |
| Gemma 4 | 32.02 | 32.08 | +0.06 |
| Qwen 3.6 | 24.62 | 23.46 | -1.16 |
| xCOMET XL | 13.77 | 16.74 | +2.97 |
| xCOMET XXL | 13.93 | 17.85 | +3.92 |

## Thinking

| Auto-evaluator | Non-thinking F1 | Thinking F1 | ΔF1 |
| --- | --- | --- | --- |
| Gemma 4 | 32.08 | 32.58 | +0.49 |
| Qwen 3.6 | 23.46 | 29.38 | +5.91 |
| Gemma 4 [reference-free] | 32.02 | 31.72 | -0.31 |
| Qwen 3.6 [reference-free] | 24.62 | 29.77 | +5.15 |

[thinking] marks thinking evaluators; [reference-free] marks reference-free evaluators.
