# Final Dataset Report

Generated at: 2026-06-25T03:53:05Z

## What Was Created
- Stage 1 audit reports for the three provided Reddit CSV samples.
- Stage 2 CloudDesk reference profile, taxonomy, pricing, releases, documentation, and hidden-pattern ground truth.
- Stage 3 pilot files with `pilot_` prefixes.
- Final-scale synthetic CloudDesk corpus in canonical paths, sized to satisfy the reference target counts.
- Synthetic issue-tracker records covering the GitHub-issues source type without pretending live GitHub data was scraped.
- Public CSV normalized and quarantined records with product-identity validation.
- Rule-based atom suggestions, incident clusters, retrieval questions, release-impact cases, abstention questions, and answer-quality rubric.
- Vector-index include and exclusion manifests that keep evaluation answer keys outside normal retrieval.

## Product Contamination Findings
- `echo_dot_reddit_reviews.csv`: accepted 6 / 31; wrong-product 2; statuses {'quarantined_ambiguous_product': 14, 'quarantined_off_topic': 9, 'quarantined_wrong_product': 2, 'accepted': 6}.
- `sony_wh1000xm5_reddit_reviews.csv`: accepted 2 / 15; wrong-product 9; statuses {'quarantined_wrong_product': 9, 'quarantined_ambiguous_product': 1, 'accepted': 2, 'quarantined_off_topic': 3}.
- `wyze_cam_v3_reddit_reviews.csv`: accepted 8 / 15; wrong-product 0; statuses {'accepted': 8, 'quarantined_off_topic': 1, 'quarantined_ambiguous_product': 6}.

## Missing-Field Analysis
- `echo_dot_reddit_reviews.csv` lacks normalized product identity, parent IDs, thread context fields, ratings, fetched_at, source metadata, validation status, and provenance hashes. Fully empty supplied columns: none.
- `sony_wh1000xm5_reddit_reviews.csv` lacks normalized product identity, parent IDs, thread context fields, ratings, fetched_at, source metadata, validation status, and provenance hashes. Fully empty supplied columns: none.
- `wyze_cam_v3_reddit_reviews.csv` lacks normalized product identity, parent IDs, thread context fields, ratings, fetched_at, source metadata, validation status, and provenance hashes. Fully empty supplied columns: none.

## Proposed Normalized Schemas
- See `schemas/feedback_item.schema.json`, `schemas/feedback_atom.schema.json`, `schemas/support_ticket.schema.json`, and `schemas/retrieval_question.schema.json`.
- Public feedback preserves raw text, clean text, source URL, hashed author, product-identity decision, validation status, and content hashes.
- Synthetic records keep account/customer links for exact SQL counts while raw textual evidence remains separately addressable.

## Pilot Generation Configuration
```json
{
  "reference_target_counts": {
    "support_tickets": 3000,
    "sales_call_excerpts": 1000,
    "product_reviews": 1000,
    "churn_feedback": 300,
    "feature_requests": 500,
    "customer_interviews": 200,
    "github_issues": 500,
    "opportunities": 2000,
    "crm_accounts": 500,
    "release_notes": 20,
    "product_document_sections": 100
  },
  "pilot_counts": {
    "accounts": 25,
    "customers": 50,
    "support_tickets": 50,
    "product_reviews": 25,
    "sales_call_excerpts": 20,
    "churn_feedback": 15,
    "feature_requests": 20,
    "customer_interviews": 10,
    "github_issues": 20
  },
  "full_counts_generated": {
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
    "public_normalized": 16,
    "public_quarantined": 45,
    "feedback_atoms": 8819
  },
  "batching": [
    "pilot written with pilot_ prefix",
    "full corpus written to canonical synthetic paths"
  ],
  "validation_gate": "Critical validation errors must be zero before treating generated artifacts as ready for RAG experiments."
}
```

## Risks, Assumptions, And Limitations
- The public CSVs are user-provided samples; this run did not perform live scraping or current terms/robots review.
- GitHub issue coverage is generated CloudDesk issue-tracker data, clearly labeled synthetic, rather than scraped public GitHub records.
- Public product feature/version/plan enrichment remains placeholder-only until a lawful source review and collection pass is performed.
- Gold annotations and retrieval ground truth are machine-generated suggestions and remain pending human approval.
- Rule-based extraction is useful for bootstrapping but should be replaced or compared with human-reviewed LLM extraction.
- Evaluation files and hidden patterns are intentionally excluded from normal RAG indexing to avoid answer-key leakage.

## Data Quality
- Critical validation errors: 0
- Warnings: 0
- See `reports/data_quality_report.md` and `reports/data_quality_metrics.json`.