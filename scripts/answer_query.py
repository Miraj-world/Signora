"""Retrieve evidence and generate a citation-checked grounded answer."""
from __future__ import annotations

import argparse
import json

from answer_generation import generate_answer, load_threshold
from query_retrieval import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a Voice of Customer question from retrieved evidence.")
    parser.add_argument("query")
    parser.add_argument("--model", help="OpenAI generation model. Defaults to SIGNORA_ANSWER_MODEL or gpt-5.4-mini.")
    parser.add_argument("--index-dir")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mode", choices=["pipeline", "recall"], default="recall")
    parser.add_argument("--candidate-pool", type=int, default=250)
    parser.add_argument("--semantic-weight", type=float, default=0.75)
    parser.add_argument("--abstain-threshold", type=float)
    parser.add_argument("--product-area")
    parser.add_argument("--customer-segment")
    parser.add_argument("--source-type")
    parser.add_argument("--target-product")
    parser.add_argument("--severity")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    filters = {
        "product_area": args.product_area,
        "customer_segment": args.customer_segment,
        "source_type": args.source_type,
        "target_product": args.target_product,
        "severity": args.severity,
    }
    initial = retrieve(
        args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
        mode=args.mode,
        candidate_pool=args.candidate_pool,
        semantic_weight=args.semantic_weight,
        filters=filters,
    )
    threshold = args.abstain_threshold
    if threshold is None:
        threshold = load_threshold(initial["embedding_profile"], args.mode)
    initial["abstain_threshold"] = threshold
    initial["should_abstain"] = initial["top_score"] < threshold
    if initial["should_abstain"]:
        initial["results"] = []

    answer = generate_answer(args.query, initial, model=args.model)
    if args.json:
        print(json.dumps(answer, ensure_ascii=False, indent=2))
        return

    print(answer["answer"])
    if answer["uncertainty"]:
        print(f"\nUncertainty: {answer['uncertainty']}")
    if answer["recommendations"]:
        print("\nRecommendations:")
        for recommendation in answer["recommendations"]:
            print(f"- {recommendation}")
    if answer["citations"]:
        print("\nCitations:")
        for citation in answer["citations"]:
            source = citation.get("source_url") or citation.get("feedback_id")
            print(f"- {citation['atom_id']} ({source})")


if __name__ == "__main__":
    main()
