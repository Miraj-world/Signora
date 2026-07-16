"""Evaluate one model index as dense-only, pipeline, or recall-oriented retrieval."""
import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from common import (DEFAULT_INDEX_ROOT, DEFAULT_RESULTS_ROOT, EVAL_DIR, corpus_fingerprint,
                    keyword_score, load_index, read_jsonl, row_matches_filters,
                    row_matches_relevance, tokenize, write_json)
from models import create_encoder


def validate(index_manifest, metadata, embeddings, questions, slug):
    if index_manifest.get("format_version") != "benchmark_index_v2":
        raise ValueError("index is not a benchmark v2 index; rebuild it with build_indexes.py")
    if index_manifest.get("model_slug") != slug:
        raise ValueError("index model slug does not match requested model")
    if index_manifest.get("corpus_fingerprint") != corpus_fingerprint():
        raise ValueError("index was built from a different corpus or document recipe; rebuild it")
    if len(metadata) != len(embeddings) or index_manifest.get("record_count") != len(metadata):
        raise ValueError("index manifest, metadata, and vector count disagree")
    if embeddings.ndim != 2 or index_manifest.get("embedding_dimensions") != embeddings.shape[1]:
        raise ValueError("index embedding dimensions are malformed")
    by_id = {row.get("atom_id"): row for row in metadata}
    for question in questions:
        for atom_id in question.get("expected_atom_ids", []):
            row = by_id.get(atom_id)
            if not row:
                raise ValueError(f"{question['question_id']} references an atom missing from this index")
            if not row_matches_filters(row, question.get("filters")):
                raise ValueError(f"{question['question_id']} gold atom violates its filters")
            if not row_matches_relevance(row, question.get("relevance_rules")):
                raise ValueError(f"{question['question_id']} gold atom violates its relevance rules")


def metadata_topic_score(row, question_tokens):
    """Score query overlap against structured metadata fields."""
    if not question_tokens:
        return 0.0
    metadata_text = " ".join(str(row.get(field) or "") for field in ("topic", "subtopic", "product_area", "feedback_type", "severity", "customer_segment"))
    metadata_tokens = set(tokenize(metadata_text.replace("_", " ")))
    if not metadata_tokens:
        return 0.0
    return len(set(question_tokens) & metadata_tokens) / max(1, min(len(set(question_tokens)), len(metadata_tokens)))


def pipeline_score(semantic, question_tokens, row):
    keyword = keyword_score(question_tokens, row.get("search_text"))
    return 0.75 * semantic + 0.25 * keyword


def recall_score(semantic, question_tokens, row):
    keyword = keyword_score(question_tokens, row.get("search_text"))
    metadata = metadata_topic_score(row, question_tokens)
    return (0.55 * semantic) + (0.20 * keyword) + (0.25 * metadata)


def ranked(query_vector, question, metadata, embeddings, mode, top_k, candidate_pool=250):
    scores = embeddings @ query_vector
    question_tokens = Counter(tokenize(question["question"]))
    candidates = []
    for row in metadata:
        if row_matches_filters(row, question.get("filters")):
            score = float(scores[row["position"]])
            if mode == "pipeline":
                score = pipeline_score(score, question_tokens, row)
            elif mode == "recall":
                score = recall_score(score, question_tokens, row)
            candidates.append((score, row))
    candidates.sort(key=lambda pair: pair[0], reverse=True)

    if mode == "recall":
        broad = [
            pair for pair in candidates
            if metadata_topic_score(pair[1], question_tokens) > 0 or keyword_score(question_tokens, pair[1].get("search_text")) > 0
        ]
        broad.sort(key=lambda pair: pair[0], reverse=True)
        by_atom = {}
        for score, row in candidates[:candidate_pool] + broad[:candidate_pool]:
            atom_id = row.get("atom_id")
            if atom_id not in by_atom or score > by_atom[atom_id][0]:
                by_atom[atom_id] = (score, row)
        candidates = sorted(by_atom.values(), key=lambda pair: pair[0], reverse=True)

    selected, seen_feedback = [], set()
    for score, row in candidates:
        if row["feedback_id"] in seen_feedback:
            continue
        selected.append((score, row))
        seen_feedback.add(row["feedback_id"])
        if len(selected) == top_k:
            break
    return selected


def score_questions(questions, encode, metadata, embeddings, mode, top_k):
    rows, metrics = [], {"canonical_recall": [], "precision": [], "predicate_coverage": [], "mrr": [], "ndcg": []}
    for question in questions:
        started = time.perf_counter()
        vector = encode([question["question"]], "query")[0]
        results = ranked(vector, question, metadata, embeddings, mode, top_k)
        relevance = [row_matches_relevance(row, question["relevance_rules"]) for _, row in results]
        expected = set(question["expected_atom_ids"])
        returned = [row["atom_id"] for _, row in results]
        canonical_recall = len(expected & set(returned)) / len(expected)
        precision = sum(relevance) / len(results) if results else 0.0
        predicate_denom = min(top_k, question.get("matching_atom_count") or len(expected))
        predicate_coverage = sum(relevance) / predicate_denom if predicate_denom else 0.0
        first = next((i for i, relevant in enumerate(relevance, 1) if relevant), None)
        mrr = 1 / first if first else 0.0
        dcg = sum(1 / math.log2(i + 1) for i, relevant in enumerate(relevance, 1) if relevant)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(sum(relevance), top_k) + 1))
        ndcg = dcg / ideal if ideal else 0.0
        for name, value in [("canonical_recall", canonical_recall), ("precision", precision), ("predicate_coverage", predicate_coverage), ("mrr", mrr), ("ndcg", ndcg)]:
            metrics[name].append(value)
        rows.append({
            "question_id": question["question_id"],
            "canonical_recall": canonical_recall,
            "precision_at_k": precision,
            "predicate_coverage_at_k": predicate_coverage,
            "mrr": mrr,
            "ndcg": ndcg,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "results": [{"atom_id": row["atom_id"], "score": score, "relevant": relevant} for (score, row), relevant in zip(results, relevance)],
        })
    return rows, {f"{name}_at_{top_k}": sum(values) / len(values) if values else 0.0 for name, values in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=None)
    parser.add_argument("--mode", choices=["dense", "pipeline", "recall"], default="dense")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()
    spec, encode = create_encoder(args.model)
    questions = [q for q in read_jsonl(EVAL_DIR / "retrieval_questions_v2.jsonl") if q.get("split") == args.split]
    manifest, metadata, embeddings = load_index(Path(args.index_root) / args.model)
    validate(manifest, metadata, embeddings, questions, args.model)
    details, metrics = score_questions(questions, encode, metadata, embeddings, args.mode, args.top_k)
    result = {"benchmark_version": "v2", "model": spec["name"], "model_slug": args.model, "mode": args.mode, "split": args.split, "top_k": args.top_k, "question_count": len(questions), "metrics": metrics, "details": details}
    destination = Path(args.results_root) / f"{args.model}_{args.mode}_{args.split}.json"
    write_json(destination, result)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
