# Benchmark v2

This framework compares every embedding model against the identical corpus and
document recipe. It intentionally keeps dense-only scoring separate from the
production-like keyword-fusion pipeline.

```powershell
# Build a local baseline first.
py -3.12 scripts\benchmark\build_indexes.py --models minilm mpnet

# Evaluate each model two ways.
py -3.12 scripts\benchmark\evaluate.py minilm --mode dense
py -3.12 scripts\benchmark\evaluate.py minilm --mode pipeline

# Calibrate no-answer thresholds on validation cases, then score test cases.
py -3.12 scripts\benchmark\evaluate_abstention.py minilm --mode dense
py -3.12 scripts\benchmark\report.py
```

API providers require their existing environment variables (`OPENAI_API_KEY`,
`COHERE_API_KEY`, `GOOGLE_API_KEY`, and `VOYAGE_API_KEY`). Benchmark indexes and
results are rebuildable local artifacts and are intentionally ignored by Git.

The evaluator refuses to score a stale index: its model slug, corpus hashes,
document-recipe hash, vector dimensions, record count, and every gold atom's
metadata predicate must match. `canonical_recall` measures recovery of the
curated citation examples; precision, MRR, and nDCG use the full metadata-backed
relevance predicate.
