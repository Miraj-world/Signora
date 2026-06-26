import os
import sys
import time
from pathlib import Path

import cohere
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils

MODEL_NAME = "embed-english-v3.0"
MODEL_SLUG = "cohere"
INDEX_DIR = eval_utils.DATASET_ROOT / "index" / MODEL_SLUG
SLEEP_PER_CALL = 0.7   # 0.7s → ~85 calls/min, safely under 100 calls/min trial limit


def main():
    eval_utils.load_dotenv()
    if not INDEX_DIR.exists():
        print("Index not found. Run embed_cohere.py first.")
        sys.exit(1)

    if not os.environ.get("COHERE_API_KEY"):
        print("Error: COHERE_API_KEY environment variable not set.")
        sys.exit(1)

    manifest, metadata, embeddings = eval_utils.load_index(INDEX_DIR)
    co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

    def embed_query(text, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = co.embed(texts=[text], model=MODEL_NAME, input_type="search_query")
                embs = response.embeddings
                if hasattr(embs, "float_"):
                    embs = embs.float_
                time.sleep(SLEEP_PER_CALL)
                return list(embs)[0]
            except Exception as e:
                if "429" in str(e) or "TooManyRequests" in type(e).__name__ or "rate" in str(e).lower():
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s (retry {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded.")

    def encode_fn(text):
        raw = embed_query(text)
        emb = np.array(raw, dtype="float32")
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
