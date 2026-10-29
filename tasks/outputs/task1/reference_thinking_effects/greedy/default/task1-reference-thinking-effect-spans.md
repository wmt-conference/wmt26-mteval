# Task 1: prediction behavior by condition

## Reference use

| Auto-evaluator | Errors per prediction: Ref-free | Errors per prediction: Ref-based | Errors per prediction: Δ | Mean span length: Ref-free | Mean span length: Ref-based | Mean span length: Δ | Mean severity weight: Ref-free | Mean severity weight: Ref-based | Mean severity weight: Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 [thinking] | 2.40 | 3.13 | +0.73 | 21.20 | 18.03 | -3.16 | 0.57 | 0.52 | -0.05 |
| Qwen 3.6 [thinking] | 3.11 | 4.10 | +0.99 | 21.43 | 20.14 | -1.29 | 0.55 | 0.50 | -0.04 |
| Gemma 4 | 2.61 | 3.65 | +1.04 | 21.69 | 20.32 | -1.36 | 0.61 | 0.53 | -0.07 |
| Qwen 3.6 | 3.93 | 5.23 | +1.29 | 27.43 | 26.49 | -0.94 | 0.67 | 0.64 | -0.03 |

## Thinking

| Auto-evaluator | Errors per prediction: Non-thinking | Errors per prediction: Thinking | Errors per prediction: Δ | Mean span length: Non-thinking | Mean span length: Thinking | Mean span length: Δ | Mean severity weight: Non-thinking | Mean severity weight: Thinking | Mean severity weight: Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 | 3.71 | 3.18 | -0.53 | 19.69 | 17.50 | -2.19 | 0.53 | 0.51 | -0.02 |
| Qwen 3.6 | 5.36 | 4.15 | -1.21 | 25.84 | 19.88 | -5.95 | 0.63 | 0.50 | -0.13 |
| Gemma 4 [reference-free] | 2.67 | 2.44 | -0.24 | 21.25 | 20.27 | -0.98 | 0.60 | 0.57 | -0.04 |
| Qwen 3.6 [reference-free] | 3.97 | 3.15 | -0.82 | 27.34 | 20.95 | -6.39 | 0.67 | 0.54 | -0.13 |

Errors per prediction include omissions. Mean span length excludes omissions; mean severity weight includes them. [thinking] and [reference-free] identify the marked evaluator conditions.
