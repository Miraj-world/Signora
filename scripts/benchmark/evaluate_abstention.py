"""Calibrate a no-answer threshold on validation cases and score it on test cases."""
import argparse
from pathlib import Path

from common import DEFAULT_INDEX_ROOT, DEFAULT_RESULTS_ROOT, EVAL_DIR, load_index, read_jsonl, write_json
from evaluate import ranked, validate
from models import create_encoder


def top_score(question, encode, metadata, embeddings, mode):
    results = ranked(encode([question["question"]], "query")[0], question, metadata, embeddings, mode, 1)
    return results[0][0] if results else float("-inf")


def select_threshold(positive_scores, negative_scores):
    """Choose a threshold without ever consulting the held-out test split."""
    candidates = sorted(set(positive_scores + negative_scores))
    if not candidates:
        raise ValueError("validation split contains no scores")
    return max(candidates, key=lambda threshold: (
        sum(score >= threshold for score in positive_scores) / len(positive_scores)
        + sum(score < threshold for score in negative_scores) / len(negative_scores)
    ) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--mode", choices=["dense", "pipeline"], default="dense")
    parser.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()

    spec, encode = create_encoder(args.model)
    retrieval = read_jsonl(EVAL_DIR / "retrieval_questions_v2.jsonl")
    abstention = read_jsonl(EVAL_DIR / "abstention_questions_v2.jsonl")
    manifest, metadata, embeddings = load_index(Path(args.index_root) / args.model)
    validate(manifest, metadata, embeddings, retrieval, args.model)
    validation_positive = [top_score(q, encode, metadata, embeddings, args.mode) for q in retrieval if q["split"] == "validation"]
    validation_negative = [top_score(q, encode, metadata, embeddings, args.mode) for q in abstention if q["split"] == "validation"]
    threshold = select_threshold(validation_positive, validation_negative)
    test_positive = [top_score(q, encode, metadata, embeddings, args.mode) for q in retrieval if q["split"] == "test"]
    test_negative = [top_score(q, encode, metadata, embeddings, args.mode) for q in abstention if q["split"] == "test"]
    true_positive = sum(score >= threshold for score in test_positive)
    true_negative = sum(score < threshold for score in test_negative)
    result = {
        "benchmark_version": "v2", "model": spec["name"], "model_slug": args.model, "mode": args.mode,
        "validation_threshold": threshold,
        "test_answerable_accuracy": true_positive / len(test_positive),
        "test_abstention_accuracy": true_negative / len(test_negative),
        "test_balanced_accuracy": ((true_positive / len(test_positive)) + (true_negative / len(test_negative))) / 2,
        "test_answerable_count": len(test_positive), "test_abstention_count": len(test_negative),
    }
    output = Path(args.results_root) / f"{args.model}_{args.mode}_abstention.json"
    write_json(output, result)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
