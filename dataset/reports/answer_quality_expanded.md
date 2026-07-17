# Expanded Grounded Answer Quality Evaluation

## Scope

- Evaluation: `answer_quality_v1`
- Answer model: `gpt-5.4-mini`
- Judge model: `gpt-5.4-mini`
- Held-out answerable questions: 12
- Held-out abstention questions: 10
- Retrieval mode: recall-oriented OpenAI-small

## Result

Status: **PASS**

| Metric | Result | Floor |
|---|---:|---:|
| Status accuracy | 1.000 | 0.900 |
| Answerable accuracy | 1.000 | - |
| Abstention accuracy | 1.000 | 0.950 |
| Citation presence | 1.000 | 1.000 |
| Citation validity | 1.000 | 1.000 |
| Citation predicate precision | 0.983 | 0.950 |
| Citation entailment | 0.958 | 0.900 |
| Unsupported-claim control | 0.917 | 0.900 |
| Uncertainty quality | 0.875 | 0.750 |
| Recommendation/fact separation | 1.000 | 0.750 |
| Counterevidence handling | 1.000 | 0.750 |
| Source diversity | 1.000 | - |

## Changes That Cleared The Gate

- Generated answers now consist of atomic claims with claim-specific atom IDs.
- The program renders inline citations from the structured claim mapping.
- Direct issue claims are separated from cited caveats and recommendations.
- Troubleshooting, support outcomes, and pilot exceptions are not treated as additional customer problems.
- The quality judge scores unsupported claims by evidence support rather than by whether a relevant detail was explicitly requested.

## Residual Risk

- One offline-task-viewing answer included a cited caveat outside the exact benchmark predicate, reducing citation predicate precision to `0.983` overall.
- Time-zone handling and offline-task-viewing cases still show occasional semantically adjacent atom selection.
- Judge scores can contain minor rationale/score inconsistencies, so deterministic citation checks remain the hard safety boundary.

The generated detailed result remains under `dataset/benchmark_results/` and is intentionally ignored because it contains rebuildable model output.
