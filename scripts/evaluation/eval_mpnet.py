import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils

MODEL_NAME = "all-mpnet-base-v2"
MODEL_SLUG = "mpnet"
INDEX_DIR = eval_utils.DATASET_ROOT / "index" / MODEL_SLUG


def main():
    if not INDEX_DIR.exists():
        print(f"Index not found. Run embed_mpnet.py first.")
        sys.exit(1)

    manifest, metadata, embeddings = eval_utils.load_index(INDEX_DIR)
    model = SentenceTransformer(MODEL_NAME)

    def encode_fn(text):
        return eval_utils.normalize_vector(
            model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        )

    questions = eval_utils.read_jsonl(eval_utils.EVAL_DIR / "retrieval_questions.jsonl")
    abstention_questions = eval_utils.read_jsonl(eval_utils.EVAL_DIR / "abstention_questions.jsonl")

    start = time.time()
    overall, by_difficulty = eval_utils.compute_metrics(questions, encode_fn, metadata, embeddings)
    abstention_correct, abstention_total = eval_utils.compute_abstention(
        abstention_questions, encode_fn, metadata, embeddings
    )
    elapsed = time.time() - start

    eval_utils.print_results(MODEL_NAME, overall, by_difficulty, abstention_correct, abstention_total, elapsed)
    eval_utils.save_results(MODEL_SLUG, MODEL_NAME, overall, by_difficulty, abstention_correct, abstention_total, elapsed)


if __name__ == "__main__":
    main()
