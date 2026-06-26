# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Signora** is a Voice of Customer (VoC) dataset build pipeline for e-commerce product feedback. It ingests public Reddit CSV exports, generates synthetic B2B SaaS (CloudDesk) feedback, extracts atomic feedback statements, builds a semantic retrieval index, and produces evaluation ground truth — all for training and benchmarking RAG systems.

## Commands

**Build the full dataset pipeline:**
```powershell
python scripts\build_voc_dataset.py
```
Reads `dataset/source_inputs/`, writes all artifacts under `dataset/` (JSONL, CSV, JSON, reports, manifests).

**Validate dataset integrity:**
```powershell
python scripts\tests\test_dataset_integrity.py
```
Expected output: `dataset integrity checks passed`. Checks manifest counts, schema constraints, and evaluation isolation.

**Install dependencies (Python 3.12):**
```powershell
py -3.12 -m pip install -r requirements.txt
```

**Build semantic retrieval index:**
```powershell
py -3.12 scripts\build_retrieval_index.py
```
Reads `dataset/data/processed/feedback_atoms.jsonl`, writes embeddings to `dataset/index/` (git-ignored, rebuildable).

**Query the retrieval index:**
```powershell
py -3.12 scripts\query_retrieval.py "query text" --top-k 5
py -3.12 scripts\query_retrieval.py "query text" --product-area onboarding --customer-segment enterprise --json
```

**Run retrieval index tests:**
```powershell
python scripts\tests\test_retrieval_index.py
```

## Architecture

### Data Flow

```
dataset/source_inputs/        (public Reddit CSVs)
        │
        ▼
scripts/build_voc_dataset.py  (core 1300-line pipeline)
        │
        ├── Stage 1: Audit & normalize public CSVs
        │     Rule-based product identity resolution (Amazon Echo families,
        │     Sony XM generations, Wyze models) → normalized / quarantined
        │
        ├── Stage 2: Generate CloudDesk reference corpus
        │     Taxonomy (10 product areas), pricing, releases, documentation
        │
        ├── Stage 3: Pilot generation (25 accounts, 50 customers, small scale)
        │
        ├── Stage 4: Full synthetic generation
        │     Support tickets, sales calls, reviews, churn feedback,
        │     feature requests, interviews, GitHub issues, usage summaries
        │     All seeded with RNG seed 73217 for reproducibility
        │
        ├── Stage 5: Feedback atom extraction
        │     Rule-based (not LLM) extraction breaks feedback into
        │     classified statements: feedback_type, product_area, topic,
        │     sentiment, severity, evidence_role
        │
        ├── Stage 6: Evaluation ground truth generation
        │     Gold annotations, 130 retrieval questions, abstention cases,
        │     release impact cases, answer quality rubric
        │     ⚠ EXCLUDED from normal RAG index to prevent leakage
        │
        └── Stage 7: Validation, manifests, reports
              dataset_manifest.json, vector_index_include_manifest.json,
              index_exclusion_manifest.json

scripts/build_retrieval_index.py
        Encodes feedback_atoms.jsonl with sentence-transformers
        (default: all-MiniLM-L6-v2) → normalized float32 embeddings
        Writes: index/feedback_atom_embeddings.npz + metadata JSONL

scripts/query_retrieval.py
        Semantic + keyword fusion ranking (75% / 25%)
        Metadata filtering, diversity deduplication by feedback_id
```

### Key Output Paths

| Path | Contents |
|------|----------|
| `dataset/data/public/normalized/` | Accepted Reddit feedback (product-normalized) |
| `dataset/data/public/quarantined/` | Rejected records |
| `dataset/data/synthetic/structured/` | CloudDesk CSV: accounts, customers, opportunities |
| `dataset/data/synthetic/raw/` | CloudDesk JSONL: tickets, calls, reviews, etc. |
| `dataset/data/processed/feedback_items.jsonl` | Combined public + synthetic feedback |
| `dataset/data/processed/feedback_atoms.jsonl` | Granular feedback statements (RAG index input) |
| `dataset/data/evaluation/` | Ground truth — excluded from RAG index |
| `dataset/data/manifests/` | Version, counts, inclusion/exclusion lists |
| `dataset/reports/` | Quality reports and build summaries |
| `dataset/index/` | Embeddings + metadata (git-ignored) |
| `dataset/schemas/` | JSON schemas for feedback_item, feedback_atom, etc. |

### Design Decisions

- **No LLMs in the pipeline**: All extraction, classification, and synthesis is rule-based + deterministic RNG (seed 73217). The pipeline can run offline without API keys.
- **Evaluation isolation**: `index_exclusion_manifest.json` explicitly lists evaluation paths so downstream RAG systems can exclude them from indexing. Hidden patterns in `data/synthetic/reference/` are answer keys, not corpus.
- **Dual scale**: Pilot (50 records) and Full (1250–3000) generated with identical logic for iterative testing.
- **Fusion retrieval**: `query_retrieval.py` fuses dense semantic scores with sparse keyword overlap; diversity deduplication prevents the same feedback_id from dominating top-k results.

## Tech Stack

- **Python 3.12** — no web framework, pure procedural pipeline
- **sentence-transformers ≥ 3.0** — embedding model for retrieval index
- **numpy ≥ 1.26** — embedding storage and cosine similarity
- No linters, formatters, or type checkers are currently configured
