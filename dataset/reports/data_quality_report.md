# Data Quality Report

Generated at: 2026-06-25T14:57:00Z

Critical errors: 0
Warnings: 0

## Evaluation Leakage
- `data/evaluation/**` is excluded from the normal vector index include manifest.
- `data/synthetic/reference/hidden_patterns.json` is excluded from normal retrieval.

## Counts
```json
{
  "accounts": 500,
  "customers": 1250,
  "support_tickets": 3000,
  "sales_call_excerpts": 1000,
  "product_reviews": 1000,
  "churn_feedback": 300,
  "feature_requests": 500,
  "customer_interviews": 200,
  "github_issues": 500,
  "opportunities": 2000,
  "usage_summaries": 1000,
  "public_normalized": 50,
  "public_quarantined": 11,
  "feedback_atoms": 8897
}
```
