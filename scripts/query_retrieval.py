"""Query the Signora production retrieval index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_models import DEFAULT_PROFILE, create_encoder, profile_for_model, resolve_profile
from retrieval_strategy import ScoredRow, rank_rows, tokenize


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"
DEFAULT_INDEX_ROOT = DATASET_ROOT / "index"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def passes_filters(row: dict, filters: dict | None) -> bool:
    checks = filters or {}
    return all(not expected or row.get(key) == expected for key, expected in checks.items())


def load_index(index_dir: Path) -> tuple[dict, list[dict], np.ndarray]:
    manifest = read_json(index_dir / "index_manifest.json")
    metadata = read_jsonl(index_dir / manifest["metadata_file"])
    embeddings = np.load(index_dir / manifest["embedding_file"])["embeddings"]
    if len(metadata) != len(embeddings):
        raise ValueError(f"Index metadata/vector mismatch: {len(metadata)} metadata rows, {len(embeddings)} embeddings")
    return manifest, metadata, embeddings


def resolve_index_dir(requested: str | None) -> Path:
    if requested:
        return Path(requested)
    profile, _ = resolve_profile(None)
    profile_dir = DEFAULT_INDEX_ROOT / profile
    if (profile_dir / "index_manifest.json").exists():
        return profile_dir
    return DEFAULT_INDEX_ROOT


def manifest_profile(manifest: dict) -> str | None:
    return manifest.get("embedding_profile") or profile_for_model(manifest.get("model"))


def rank(
    query: str,
    metadata: list[dict],
    embeddings: np.ndarray,
    manifest: dict,
    mode: str,
    candidate_pool: int,
    semantic_weight: float,
    filters: dict | None = None,
) -> list[ScoredRow]:
    profile = manifest_profile(manifest)
    if not profile:
        raise ValueError(
            "Index manifest does not identify a supported embedding profile. "
            "Rebuild it with scripts/build_retrieval_index.py."
        )
    _, _, encode = create_encoder(profile)
    query_embedding = encode([query], "query")[0]
    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            f"Index/query dimension mismatch: index={embeddings.shape[1]} query={query_embedding.shape[0]} "
            f"profile={profile}. Rebuild the index for that profile."
        )
    semantic_scores = embeddings @ query_embedding
    filtered = [row for row in metadata if passes_filters(row, filters)]
    return rank_rows(
        query,
        filtered,
        semantic_scores,
        mode=mode,
        candidate_pool=candidate_pool,
        semantic_weight=semantic_weight,
    )


def diverse_top_k(candidates: list[ScoredRow], top_k: int) -> list[ScoredRow]:
    selected = []
    seen_feedback_ids = set()
    seen_statements = set()

    for candidate in candidates:
        row = candidate.row
        feedback_id = row.get("feedback_id")
        statement_key = " ".join(tokenize(row.get("statement")))
        if feedback_id and feedback_id in seen_feedback_ids:
            continue
        if statement_key and statement_key in seen_statements:
            continue
        seen_feedback_ids.add(feedback_id)
        seen_statements.add(statement_key)
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    return selected


def format_result(rank_number: int, scored: ScoredRow) -> dict:
    row = scored.row
    return {
        "rank": rank_number,
        "score": round(scored.score, 4),
        "semantic_score": round(scored.semantic, 4),
        "keyword_score": round(scored.keyword, 4),
        "metadata_score": round(scored.metadata, 4),
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


def retrieve(
    query: str,
    index_dir: str | None = None,
    top_k: int = 8,
    mode: str = "recall",
    candidate_pool: int = 250,
    semantic_weight: float = 0.75,
    abstain_threshold: float | None = None,
    filters: dict | None = None,
) -> dict:
    resolved_index = resolve_index_dir(index_dir)
    manifest, metadata, embeddings = load_index(resolved_index)
    candidates = rank(
        query,
        metadata,
        embeddings,
        manifest,
        mode=mode,
        candidate_pool=candidate_pool,
        semantic_weight=semantic_weight,
        filters=filters,
    )
    selected = diverse_top_k(candidates, top_k)
    results = [format_result(i + 1, scored) for i, scored in enumerate(selected)]
    top_score = results[0]["score"] if results else 0.0
    should_abstain = abstain_threshold is not None and top_score < abstain_threshold
    return {
        "query": query,
        "index_model": manifest.get("model"),
        "index_dir": str(resolved_index),
        "embedding_profile": manifest_profile(manifest),
        "retrieval_mode": mode,
        "candidate_pool": candidate_pool,
        "semantic_weight": semantic_weight,
        "abstain_threshold": abstain_threshold,
        "top_score": top_score,
        "should_abstain": should_abstain,
        "results": [] if should_abstain else results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Signora retrieval index.")
    parser.add_argument("query", help="Natural-language retrieval query.")
    parser.add_argument("--index-dir", help=f"Directory containing index_manifest.json. Defaults to dataset/index/{DEFAULT_PROFILE} when available.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of diverse results to return.")
    parser.add_argument("--mode", choices=["pipeline", "recall"], default="recall", help="Retrieval strategy. Recall broadens candidates with metadata/topic signals before reranking.")
    parser.add_argument("--candidate-pool", type=int, default=250, help="Candidates gathered per semantic, keyword, and metadata route in recall mode.")
    parser.add_argument("--semantic-weight", type=float, default=0.75, help="Fusion weight for semantic score; keyword gets the remainder.")
    parser.add_argument("--abstain-threshold", type=float, help="If the top fused score is below this, return no answer.")
    parser.add_argument("--product-area", help="Filter by product_area.")
    parser.add_argument("--customer-segment", help="Filter by customer_segment.")
    parser.add_argument("--source-type", help="Filter by source_type.")
    parser.add_argument("--target-product", help="Filter by target_product.")
    parser.add_argument("--severity", help="Filter by severity.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    filters = {
        "product_area": args.product_area,
        "customer_segment": args.customer_segment,
        "source_type": args.source_type,
        "target_product": args.target_product,
        "severity": args.severity,
    }
    payload = retrieve(
        args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
        mode=args.mode,
        candidate_pool=args.candidate_pool,
        semantic_weight=args.semantic_weight,
        abstain_threshold=args.abstain_threshold,
        filters=filters,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"Query: {payload['query']}")
    print(f"Index model: {payload['index_model']} ({payload['embedding_profile']})")
    print(f"Retrieval mode: {payload['retrieval_mode']}")
    print(f"Results: {len(payload['results'])}")
    print()

    if payload["should_abstain"]:
        print(f"No confident evidence found. top_score={payload['top_score']:.4f} threshold={args.abstain_threshold:.4f}")
        return

    for result in payload["results"]:
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
