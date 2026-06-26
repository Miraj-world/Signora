import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from embed_utils import build_metadata, load_atoms_and_items, load_dotenv, normalize_matrix, save_index

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset"
MODEL_NAME = "text-embedding-3-large"
MODEL_SLUG = "openai_large"
BATCH_SIZE = 96


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=str(DATASET_ROOT / "index" / MODEL_SLUG))
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI()

    atoms, items_by_feedback_id = load_atoms_and_items(DATASET_ROOT)
    metadata, texts = build_metadata(atoms, items_by_feedback_id)

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        print(f"Encoding batch {batch_num}/{total_batches}...")
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=MODEL_NAME)
        embs = [e.embedding for e in response.data]
        all_embeddings.extend(embs)
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.5)

    embeddings = normalize_matrix(np.array(all_embeddings, dtype="float32"))
    save_index(Path(args.index_dir), embeddings, metadata, MODEL_NAME)


if __name__ == "__main__":
    main()
