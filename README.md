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

Build the production embedding index. The default profile is `openai_small`,
which uses `text-embedding-3-small` because it was the practical winner in the
benchmark: it led the core retrieval metrics while being cheaper than
`text-embedding-3-large`.

```powershell
py -3.12 scripts\build_retrieval_index.py
```

For a fully local/offline baseline, build with MPNet instead:

```powershell
py -3.12 scripts\build_retrieval_index.py --profile mpnet
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

The query pipeline uses the same model recorded in the index manifest and fuses
dense semantic similarity with keyword overlap. The benchmark showed keyword
fusion improved retrieval ranking for both local models, so the production
default keeps fusion enabled with `--semantic-weight 0.75`.

## Repeatable Benchmark Habit

Use the v2 benchmark whenever the corpus, document recipe, embedding model,
filters, or ranking logic changes:

```powershell
py -3.12 scripts\benchmark\run_reproducible_eval.py --models openai_small mpnet
```

That command rebuilds model-specific indexes, evaluates dense retrieval and the
production-like keyword-fusion pipeline, scores abstention, and regenerates the
benchmark report under `dataset/benchmark_results/`.

Current interpretation from the completed experiment:

- `text-embedding-3-small` is the production default for this benchmark.
- `all-mpnet-base-v2` is the strongest local baseline.
- Keyword fusion improved retrieval ranking for both local models.
- MPNet dense retrieval produced the strongest held-out local abstention result.

The lower-level benchmark runner uses 59 predicate-backed retrieval questions
and 20 no-answer cases. It reports exact canonical citation recall separately
from predicate-backed evidence coverage so broad semantic retrieval questions
are not graded only against one small set of pre-selected atom IDs:

```powershell
py -3.12 scripts\evaluation\run_benchmark.py
```

Build one isolated index per model first, for example `py -3.12
scripts\embeddings\embed_minilm.py`; indexes live under
`dataset/index/<model-slug>/` and are intentionally ignored by git.

## Suggested Next Decisions

- Do error analysis on the weak queries for `text-embedding-3-small`.
- Decide whether parent-thread context should be retrieved automatically for Reddit comments.
- Add answer generation on top of retrieved evidence, with citations back to atom IDs and source feedback IDs.

## Notes

- The generated CloudDesk corpus is synthetic.
- Public Reddit CSV samples are user-provided source material and should be reviewed before publication.
- Evaluation answer keys and hidden patterns are excluded from the normal vector-index include manifest.
