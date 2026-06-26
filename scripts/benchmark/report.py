"""Turn benchmark result JSON files into a compact Markdown comparison."""
import argparse
from pathlib import Path

from common import DEFAULT_RESULTS_ROOT, read_json


def value(number):
    return "--" if number is None else f"{number:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.results_root)
    retrieval = {}
    abstention = {}
    for path in root.glob("*.json"):
        payload = read_json(path)
        key = (payload.get("model_slug"), payload.get("mode"))
        if path.stem.endswith("_abstention"):
            abstention[key] = payload
        elif "metrics" in payload:
            retrieval[key] = payload
    lines = ["# Signora Embedding Benchmark v2", "", "All rows use the same corpus snapshot and document recipe.", "", "| Model | Mode | Canonical Recall | Precision | MRR | nDCG | Abstention Balanced Accuracy |", "|---|---|---:|---:|---:|---:|---:|"]
    for key in sorted(retrieval):
        result = retrieval[key]
        metrics = result["metrics"]
        suffix = f"_at_{result['top_k']}"
        abstain = abstention.get(key, {})
        lines.append("| {model} | {mode} | {recall} | {precision} | {mrr} | {ndcg} | {abstain} |".format(
            model=result["model"], mode=result["mode"],
            recall=value(metrics.get("canonical_recall" + suffix)), precision=value(metrics.get("precision" + suffix)),
            mrr=value(metrics.get("mrr" + suffix)), ndcg=value(metrics.get("ndcg" + suffix)),
            abstain=value(abstain.get("test_balanced_accuracy")),
        ))
    lines.extend(["", "`Canonical Recall` measures recovery of curated evidence examples. Precision, MRR, and nDCG use the full predicate-backed relevance definition. Abstention thresholds are selected only from the validation split."])
    output = Path(args.output) if args.output else root / "benchmark_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
