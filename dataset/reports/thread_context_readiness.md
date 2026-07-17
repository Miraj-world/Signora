# Parent-Thread Context Readiness

## Decision

Status: **DEFER**

Do not enable automatic parent-thread expansion yet. The current corpus has
thread grouping and root/comment flags, but it does not contain stable
`parent_id` links. Inferring parents from row order or thread membership would
create unsupported relationships and weaken citation trust.

## Corpus Audit

| Measure | Count |
|---|---:|
| Total feedback items | 6,550 |
| Threaded items | 42 |
| Unique threads | 13 |
| Threads with multiple items | 7 |
| Root items | 12 |
| Comment items | 30 |
| Items with `parent_id` | 0 |

All 42 threaded items use the `reddit_discussion` source type. The expanded
answer-quality benchmark retrieved no threaded-source rows, so it cannot
measure whether parent context improves or harms grounded answers.

## Requirements Before Enablement

1. Collect stable parent IDs during threaded-source ingestion.
2. Resolve each parent ID to a feedback item without guessing from thread order.
3. Add held-out threaded questions where parent context materially changes the answer.
4. Compare citation entailment and unsupported-claim control with context off and on.

The audit is reproducible with:

```powershell
py -3.12 scripts\audit_thread_context.py
```
