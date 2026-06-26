import argparse
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from embed_utils import build_metadata, load_atoms_and_items, normalize_matrix, save_index

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset"
MODEL_NAME = "all-mpnet-base-v2"
MODEL_SLUG = "mpnet"
BATCH_SIZE = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=str(DATASET_ROOT / "index" / MODEL_SLUG))
    args = parser.parse_args()

    atoms, items_by_feedback_id = load_atoms_and_items(DATASET_ROOT)
    metadata, texts = build_metadata(atoms, items_by_feedback_id)

    model = SentenceTransformer(MODEL_NAME)
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        print(f"Encoding batch {batch_num}/{total_batches}...")
        batch = texts[i:i + BATCH_SIZE]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(emb)

    embeddings = normalize_matrix(np.vstack(all_embeddings).astype("float32"))
    save_index(Path(args.index_dir), embeddings, metadata, MODEL_NAME)


if __name__ == "__main__":
    main()
