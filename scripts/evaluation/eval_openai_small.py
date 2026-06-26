import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils

MODEL_NAME = "text-embedding-3-small"
MODEL_SLUG = "openai_small"
INDEX_DIR = eval_utils.DATASET_ROOT / "index" / MODEL_SLUG


def main():
    eval_utils.load_dotenv()
    if not INDEX_DIR.exists():
        print(f"Index not found. Run embed_openai_small.py first.")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    manifest, metadata, embeddings = eval_utils.load_index(INDEX_DIR)
    client = OpenAI()

    def encode_fn(text):
        response = client.embeddings.create(input=[text], model=MODEL_NAME)
        emb = np.array(response.data[0].embedding, dtype="float32")
        return eval_utils.normalize_vector(emb)

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
