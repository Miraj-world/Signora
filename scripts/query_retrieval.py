"""Query the Signora production retrieval index."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from retrieval_models import create_encoder


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"
DEFAULT_INDEX_DIR = DATASET_ROOT / "index"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str | None) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9][a-z0-9_-]*", (text or "").lower()) if len(token) > 1]


def keyword_score(query_tokens: Counter, text: str | None) -> float:
    if not query_tokens:
        return 0.0
    text_counts = Counter(tokenize(text))
    if not text_counts:
        return 0.0
    overlap = sum(min(text_counts[token], count) for token, count in query_tokens.items())
    return overlap / max(1, sum(query_tokens.values()))


def passes_filters(row: dict, args: argparse.Namespace) -> bool:
    checks = {
        "product_area": args.product_area,
        "customer_segment": args.customer_segment,
        "source_type": args.source_type,
        "target_product": args.target_product,
        "severity": args.severity,
    }
    return all(not expected or row.get(key) == expected for key, expected in checks.items())


def load_index(index_dir: Path) -> tuple[dict, list[dict], np.ndarray]:
    manifest = read_json(index_dir / "index_manifest.json")
    metadata = read_jsonl(index_dir / manifest["metadata_file"])
    embeddings = np.load(index_dir / manifest["embedding_file"])["embeddings"]
    if len(metadata) != len(embeddings):
        raise ValueError(f"Index metadata/vector mismatch: {len(metadata)} metadata rows, {len(embeddings)} embeddings")
    return manifest, metadata, embeddings


def rank(query: str, metadata: list[dict], embeddings: np.ndarray, manifest: dict, args: argparse.Namespace) -> list[tuple]:
    profile = manifest.get("embedding_profile")
    _, _, encode = create_encoder(profile)
    query_embedding = encode([query], "query")[0]
    semantic_scores = embeddings @ query_embedding
    query_tokens = Counter(tokenize(query))
    candidates = []

    for row in metadata:
        if not passes_filters(row, args):
            continue
        position = row["position"]
        semantic = float(semantic_scores[position])
        keyword = keyword_score(query_tokens, row.get("search_text"))
        fused = (args.semantic_weight * semantic) + ((1 - args.semantic_weight) * keyword)
        candidates.append((fused, semantic, keyword, row))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def diverse_top_k(candidates: list[tuple], top_k: int) -> list[tuple]:
    selected = []
    seen_feedback_ids = set()
    seen_statements = set()

    for fused, semantic, keyword, row in candidates:
        feedback_id = row.get("feedback_id")
        statement_key = " ".join(tokenize(row.get("statement")))
        if feedback_id and feedback_id in seen_feedback_ids:
            continue
        if statement_key and statement_key in seen_statements:
            continue
        seen_feedback_ids.add(feedback_id)
        seen_statements.add(statement_key)
        selected.append((fused, semantic, keyword, row))
        if len(selected) >= top_k:
            break

    return selected


def format_result(rank_number: int, scored: tuple) -> dict:
    fused, semantic, keyword, row = scored
    return {
        "rank": rank_number,
        "score": round(float(fused), 4),
        "semantic_score": round(float(semantic), 4),
        "keyword_score": round(float(keyword), 4),
        "atom_id": row.get("atom_id"),
        "feedback_id": row.get("feedback_id"),
        "statement": row.get("statement"),
        "source_context": row.get("source_context"),
        "product_area": row.get("product_area"),
        "topic": row.get("topic"),
        "customer_segment": row.get("customer_segment"),
        "severity": row.get("severity"),
        "source_type": row.get("source_type"),
        "source_url": row.get("source_url"),
        "thread_id": row.get("thread_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Signora retrieval index.")
    parser.add_argument("query", help="Natural-language retrieval query.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directory containing index_manifest.json.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of diverse results to return.")
    parser.add_argument("--semantic-weight", type=float, default=0.75, help="Fusion weight for semantic score; keyword gets the remainder.")
    parser.add_argument("--abstain-threshold", type=float, help="If the top fused score is below this, return no answer.")
    parser.add_argument("--product-area", help="Filter by product_area.")
    parser.add_argument("--customer-segment", help="Filter by customer_segment.")
    parser.add_argument("--source-type", help="Filter by source_type.")
    parser.add_argument("--target-product", help="Filter by target_product.")
    parser.add_argument("--severity", help="Filter by severity.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    manifest, metadata, embeddings = load_index(index_dir)
    candidates = rank(args.query, metadata, embeddings, manifest, args)
    selected = diverse_top_k(candidates, args.top_k)
    results = [format_result(i + 1, scored) for i, scored in enumerate(selected)]
    top_score = results[0]["score"] if results else 0.0
    should_abstain = args.abstain_threshold is not None and top_score < args.abstain_threshold

    if args.json:
        print(json.dumps({
            "query": args.query,
            "index_model": manifest.get("model"),
            "embedding_profile": manifest.get("embedding_profile"),
            "semantic_weight": args.semantic_weight,
            "top_score": top_score,
            "should_abstain": should_abstain,
            "results": [] if should_abstain else results,
        }, ensure_ascii=False, indent=2))
        return

    print(f"Query: {args.query}")
    print(f"Index model: {manifest.get('model')} ({manifest.get('embedding_profile')})")
    print(f"Results: {0 if should_abstain else len(results)}")
    print()

    if should_abstain:
        print(f"No confident evidence found. top_score={top_score:.4f} threshold={args.abstain_threshold:.4f}")
        return

    for result in results:
        citation = result.get("source_url") or result.get("feedback_id")
        print(f"{result['rank']}. score={result['score']} atom={result['atom_id']} citation={citation}")
        print(
            f"   area={result.get('product_area')} topic={result.get('topic')} "
            f"segment={result.get('customer_segment')} severity={result.get('severity')} source={result.get('source_type')}"
        )
        print(f"   {result.get('statement')}")
        context = result.get("source_context")
        if context and context != result.get("statement"):
            print(f"   context: {context[:280]}")
        print()


if __name__ == "__main__":
    main()
