import os
import sys
import time
from pathlib import Path

import numpy as np
import voyageai

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils

MODEL_NAME = "voyage-large-2"
MODEL_SLUG = "voyage"
INDEX_DIR = eval_utils.DATASET_ROOT / "index" / MODEL_SLUG


def main():
    eval_utils.load_dotenv()
    if not INDEX_DIR.exists():
        print(f"Index not found. Run embed_voyage.py first.")
        sys.exit(1)

    if not os.environ.get("VOYAGE_API_KEY"):
        print("Error: VOYAGE_API_KEY environment variable not set.")
        sys.exit(1)

    manifest, metadata, embeddings = eval_utils.load_index(INDEX_DIR)
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    def encode_fn(text):
        result = vo.embed([text], model=MODEL_NAME, input_type="query")
        emb = np.array(result.embeddings[0], dtype="float32")
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
