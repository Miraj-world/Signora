import os
import sys
import time
from pathlib import Path

import google.generativeai as genai
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils

MODEL_NAME = "text-embedding-004"
MODEL_SLUG = "google"
INDEX_DIR = eval_utils.DATASET_ROOT / "index" / MODEL_SLUG


def main():
    eval_utils.load_dotenv()
    if not INDEX_DIR.exists():
        print(f"Index not found. Run embed_google.py first.")
        sys.exit(1)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    manifest, metadata, embeddings = eval_utils.load_index(INDEX_DIR)

    def encode_fn(text):
        result = genai.embed_content(
            model=f"models/{MODEL_NAME}",
            content=text,
            task_type="retrieval_query",
        )
        emb = np.array(result["embedding"], dtype="float32")
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
