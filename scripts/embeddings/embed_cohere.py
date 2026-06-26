import argparse
import os
import sys
import time
from pathlib import Path

import cohere
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from embed_utils import build_metadata, load_atoms_and_items, load_dotenv, normalize_matrix, save_index

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset"
MODEL_NAME = "embed-english-v3.0"
MODEL_SLUG = "cohere"
BATCH_SIZE = 20       # trial limit: 100k tokens/min; 20 texts × ~150 tokens = 3k tokens/batch
SLEEP_SECS = 4.0     # 4s between batches → ~15 batches/min → ~45k tokens/min (safe)


def embed_with_retry(co, batch, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = co.embed(texts=batch, model=MODEL_NAME, input_type="search_document")
            embs = response.embeddings
            if hasattr(embs, "float_"):
                embs = embs.float_
            return list(embs)
        except Exception as e:
            if "429" in str(e) or "TooManyRequests" in type(e).__name__ or "rate" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s (retry {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded — still rate limited after 5 attempts.")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=str(DATASET_ROOT / "index" / MODEL_SLUG))
    args = parser.parse_args()

    if not os.environ.get("COHERE_API_KEY"):
        print("Error: COHERE_API_KEY environment variable not set.")
        sys.exit(1)

    co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

    atoms, items_by_feedback_id = load_atoms_and_items(DATASET_ROOT)
    metadata, texts = build_metadata(atoms, items_by_feedback_id)

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        print(f"Encoding batch {batch_num}/{total_batches}...")
        batch = texts[i:i + BATCH_SIZE]
        embs = embed_with_retry(co, batch)
        all_embeddings.extend(embs)
        if i + BATCH_SIZE < len(texts):
            time.sleep(SLEEP_SECS)

    embeddings = normalize_matrix(np.array(all_embeddings, dtype="float32"))
    save_index(Path(args.index_dir), embeddings, metadata, MODEL_NAME)


if __name__ == "__main__":
    main()
