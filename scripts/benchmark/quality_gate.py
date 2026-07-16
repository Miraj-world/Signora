"""Fail when recall-oriented retrieval drops below the accepted quality floor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import DEFAULT_RESULTS_ROOT, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Check recall retrieval against baseline and quality floors.")
    parser.add_argument("--model", default="openai_small")
    parser.add_argument("--baseline", default="pipeline")
    parser.add_argument("--candidate", default="recall")
    parser.add_argument("--min-predicate-coverage", type=float, default=0.95)
    parser.add_argument("--min-abstention", type=float, default=0.95)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()

    root = Path(args.results_root)
    baseline = read_json(root / f"{args.model}_{args.baseline}_test.json")
    candidate = read_json(root / f"{args.model}_{args.candidate}_test.json")
    abstention = read_json(root / f"{args.model}_{args.candidate}_abstention.json")
    top_k = candidate["top_k"]
    suffix = f"_at_{top_k}"

    checks = {
        "predicate_coverage_floor": candidate["metrics"]["predicate_coverage" + suffix] >= args.min_predicate_coverage,
        "abstention_floor": abstention["test_balanced_accuracy"] >= args.min_abstention,
        "precision_no_regression": candidate["metrics"]["precision" + suffix] >= baseline["metrics"]["precision" + suffix],
        "canonical_recall_no_regression": candidate["metrics"]["canonical_recall" + suffix] >= baseline["metrics"]["canonical_recall" + suffix],
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "model": args.model,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "top_k": top_k,
        "checks": checks,
        "metrics": {
            "baseline_precision": baseline["metrics"]["precision" + suffix],
            "candidate_precision": candidate["metrics"]["precision" + suffix],
            "baseline_canonical_recall": baseline["metrics"]["canonical_recall" + suffix],
            "candidate_canonical_recall": candidate["metrics"]["canonical_recall" + suffix],
            "candidate_predicate_coverage": candidate["metrics"]["predicate_coverage" + suffix],
            "candidate_abstention_balanced_accuracy": abstention["test_balanced_accuracy"],
        },
    }
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
