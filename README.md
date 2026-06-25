# Signora

Signora is a Voice of Customer dataset build pipeline and generated dataset package for customer-feedback retrieval, product-identity cleanup, and RAG evaluation experiments.

## Google Drive Source

Google Drive link: [Voice of Customer source folder](https://drive.google.com/drive/u/0/folders/1RJ-O3WibYyPhTSc7A6c_J-irZuhrKyfK)

Use the Drive folder for source handoff files that should not live directly in git, such as raw archives, source documents, or replacement CSV exports.

## Project Journal

Build journal: [BUILD_LOG.md](https://www.notion.so/BUILD_LOG-md-38a777d8c0858081856fe19d80c18846?source=copy_link)

The journal tracks implementation decisions, commands run, retrieval experiments, and observations from testing Signora.

## What Is Included

- `scripts/build_voc_dataset.py` builds the dataset from the provided source inputs.
- `scripts/build_retrieval_index.py` builds a local semantic index from feedback atoms.
- `scripts/query_retrieval.py` runs local semantic retrieval with keyword score fusion and metadata filters.
- `scripts/tests/test_dataset_integrity.py` validates generated counts, public metadata cleanup, and index-exclusion rules.
- `scripts/tests/test_retrieval_index.py` validates generated retrieval index artifacts.
- `dataset/data/public/normalized/feedback_items.jsonl` contains accepted public Reddit feedback records.
- `dataset/data/public/quarantined/feedback_items.jsonl` contains records held out by validation rules.
- `dataset/data/processed/feedback_items.jsonl` and `dataset/data/processed/feedback_atoms.jsonl` contain the combined retrieval-ready corpus.
- `dataset/reports/` contains audit, quality, and final dataset reports.

## Public Reddit Metadata Cleanup

The public Reddit normalization keeps raw CSV inputs unchanged and fixes product identity in generated artifacts:

- Amazon Echo records are normalized under `Amazon Echo / Alexa` with `product_family` and `product_model`.
- Sony WH-1000X records are separated into XM4, XM5, XM6, and series-level identities.
- Reddit thread fields include `thread_id`, `root_post_id`, `parent_id`, and `is_root_post` where they can be derived conservatively.
- Wyze records with generic subreddit URLs keep unknown thread relationships as `null`.

## Rebuild

From the repository root:

```powershell
python scripts\build_voc_dataset.py
```

The build writes regenerated dataset artifacts, reports, manifests, and reference files under `dataset/`.

## Validate

```powershell
python scripts\tests\test_dataset_integrity.py
```

Expected result:

```text
dataset integrity checks passed
```

## Local Semantic Retrieval

Install the retrieval dependencies with Python 3.12:

```powershell
py -3.12 -m pip install -r requirements.txt
```

Build the local embedding index:

```powershell
py -3.12 scripts\build_retrieval_index.py
```

The generated semantic index is written to `dataset/index/`. It is intentionally ignored by git because it is rebuildable and model-dependent.

Run a retrieval query:

```powershell
py -3.12 scripts\query_retrieval.py "Why are enterprise administrators struggling with onboarding?" --top-k 5
```

Optional filters:

```powershell
py -3.12 scripts\query_retrieval.py "Which customers complain about delayed notifications?" --product-area notifications --top-k 5
```

## Suggested Next Decisions

- Decide whether to filter low-information atoms such as generic caveats before ranking.
- Decide whether parent-thread context should be retrieved automatically for Reddit comments.
- Decide whether the next milestone should be evaluation metrics, answer generation, or a small UI for querying evidence.

## Notes

- The generated CloudDesk corpus is synthetic.
- Public Reddit CSV samples are user-provided source material and should be reviewed before publication.
- Evaluation answer keys and hidden patterns are excluded from the normal vector-index include manifest.
