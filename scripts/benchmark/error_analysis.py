"""Explain benchmark retrieval misses question-by-question.

This script consumes the JSON files produced by `evaluate.py` and joins them
back to the benchmark questions plus corpus metadata.  It is intentionally
offline: it does not call embedding providers or rebuild indexes.  Its job is
to make low Canonical Recall inspectable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import DEFAULT_INDEX_ROOT, DEFAULT_RESULTS_ROOT, EVAL_DIR, load_index, read_json, read_jsonl, write_json


def pct(value: float) -> str:
    return f"{value:.3f}"


def truncate(text: str | None, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def result_path(results_root: Path, model: str, mode: str, split: str) -> Path:
    return results_root / f"{model}_{mode}_{split}.json"


def load_questions(split: str) -> dict[str, dict]:
    return {
        question["question_id"]: question
        for question in read_jsonl(EVAL_DIR / "retrieval_questions_v2.jsonl")
        if question.get("split") == split
    }


def atom_lookup(metadata: list[dict]) -> dict[str, dict]:
    return {row.get("atom_id"): row for row in metadata}


def classify_question(detail: dict) -> str:
    recall = detail["canonical_recall"]
    precision = detail["precision_at_k"]
    if recall >= 0.5:
        return "good_canonical_recovery"
    if precision >= 0.8:
        return "alternative_relevant_evidence"
    if precision >= 0.5:
        return "mixed_retrieval"
    return "retrieval_failure"


def analyze_one(model: str, mode: str, split: str, index_root: Path, results_root: Path) -> dict:
    source = result_path(results_root, model, mode, split)
    if not source.exists():
        raise FileNotFoundError(f"Missing result file: {source}. Run evaluate.py first.")

    result = read_json(source)
    questions = load_questions(split)
    manifest, metadata, _ = load_index(index_root / model)
    atoms = atom_lookup(metadata)

    rows = []
    category_counts = Counter()
    total_expected = 0
    total_returned = 0
    total_missed = 0
    total_retrieved_relevant_not_canonical = 0

    for detail in result["details"]:
        question = questions[detail["question_id"]]
        expected_ids = set(question.get("expected_atom_ids", []))
        returned_ids = [row["atom_id"] for row in detail.get("results", [])]
        returned_set = set(returned_ids)
        hit_ids = expected_ids & returned_set
        missed_ids = expected_ids - returned_set
        relevant_returned = [row for row in detail.get("results", []) if row.get("relevant")]
        relevant_noncanonical = [row for row in relevant_returned if row["atom_id"] not in expected_ids]
        category = classify_question(detail)
        category_counts[category] += 1

        total_expected += len(expected_ids)
        total_returned += len(returned_ids)
        total_missed += len(missed_ids)
        total_retrieved_relevant_not_canonical += len(relevant_noncanonical)

        rows.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "category": category,
            "canonical_recall": detail["canonical_recall"],
            "precision_at_k": detail["precision_at_k"],
            "mrr": detail["mrr"],
            "ndcg": detail["ndcg"],
            "matching_atom_count": question.get("matching_atom_count"),
            "expected_count": len(expected_ids),
            "returned_count": len(returned_ids),
            "canonical_hits": sorted(hit_ids),
            "missed_canonical_atoms": [
                {
                    "atom_id": atom_id,
                    "feedback_id": atoms.get(atom_id, {}).get("feedback_id"),
                    "statement": atoms.get(atom_id, {}).get("statement"),
                    "topic": atoms.get(atom_id, {}).get("topic"),
                    "customer_segment": atoms.get(atom_id, {}).get("customer_segment"),
                    "severity": atoms.get(atom_id, {}).get("severity"),
                }
                for atom_id in sorted(missed_ids)
            ],
            "retrieved": [
                {
                    "rank": rank,
                    "atom_id": item["atom_id"],
                    "score": item["score"],
                    "relevant": item["relevant"],
                    "canonical_hit": item["atom_id"] in expected_ids,
                    "feedback_id": atoms.get(item["atom_id"], {}).get("feedback_id"),
                    "statement": atoms.get(item["atom_id"], {}).get("statement"),
                    "topic": atoms.get(item["atom_id"], {}).get("topic"),
                    "customer_segment": atoms.get(item["atom_id"], {}).get("customer_segment"),
                    "severity": atoms.get(item["atom_id"], {}).get("severity"),
                }
                for rank, item in enumerate(detail.get("results", []), 1)
            ],
            "interpretation": interpret_question(detail, question, len(relevant_noncanonical)),
        })

    worst = sorted(rows, key=lambda row: (row["canonical_recall"], -row["precision_at_k"], row["question_id"]))
    best = sorted(rows, key=lambda row: (-row["canonical_recall"], -row["precision_at_k"], row["question_id"]))

    return {
        "benchmark_version": result.get("benchmark_version"),
        "model": result.get("model"),
        "model_slug": model,
        "mode": mode,
        "split": split,
        "top_k": result.get("top_k"),
        "source_result": str(source),
        "index_manifest": {
            "model": manifest.get("model"),
            "model_slug": manifest.get("model_slug"),
            "record_count": manifest.get("record_count"),
            "created_at": manifest.get("created_at"),
        },
        "metrics": result.get("metrics", {}),
        "summary": {
            "question_count": len(rows),
            "zero_canonical_recall_questions": sum(1 for row in rows if row["canonical_recall"] == 0),
            "low_recall_high_precision_questions": sum(
                1 for row in rows if row["canonical_recall"] < 0.5 and row["precision_at_k"] >= 0.8
            ),
            "perfect_precision_zero_recall_questions": sum(
                1 for row in rows if row["canonical_recall"] == 0 and row["precision_at_k"] == 1.0
            ),
            "category_counts": dict(category_counts),
            "total_expected_canonical_atoms": total_expected,
            "total_missed_canonical_atoms": total_missed,
            "total_returned_atoms": total_returned,
            "total_retrieved_relevant_noncanonical_atoms": total_retrieved_relevant_not_canonical,
            "average_matching_atom_count": sum(row.get("matching_atom_count") or 0 for row in rows) / len(rows) if rows else 0,
        },
        "worst_questions": worst[:10],
        "best_questions": best[:10],
        "questions": rows,
    }


def interpret_question(detail: dict, question: dict, relevant_noncanonical_count: int) -> str:
    recall = detail["canonical_recall"]
    precision = detail["precision_at_k"]
    matching = question.get("matching_atom_count")
    if recall == 0 and precision == 1.0:
        return (
            "All returned atoms matched the broader relevance predicate, but none were the exact curated canonical IDs. "
            "This points to canonical-label narrowness or many equivalent evidence atoms, not a simple retrieval failure."
        )
    if recall < 0.5 and precision >= 0.8:
        return (
            f"Low exact-ID recovery, but high relevance among returned results. Retrieved {relevant_noncanonical_count} "
            "relevant non-canonical atoms, so inspect whether these are acceptable sibling evidence."
        )
    if precision < 0.5:
        return "Both exact canonical recovery and broader relevance are weak; this is a true retrieval/ranking failure candidate."
    if matching and matching > 100:
        return "This question has a large relevant pool, so exact canonical recovery at small top-k is naturally difficult."
    return "Canonical recovery is moderate/high; this question is less urgent for recall debugging."


def markdown_for(analysis: dict) -> str:
    summary = analysis["summary"]
    metrics = analysis["metrics"]
    lines = [
        f"# Retrieval Error Analysis: {analysis['model']} / {analysis['mode']}",
        "",
        f"- Split: `{analysis['split']}`",
        f"- Top-k: `{analysis['top_k']}`",
        f"- Questions analyzed: `{summary['question_count']}`",
        f"- Index records: `{analysis['index_manifest'].get('record_count')}`",
        "",
        "## Benchmark metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {pct(value)} |")

    lines.extend([
        "",
        "## Error-analysis summary",
        "",
        "| Signal | Value |",
        "|---|---:|",
        f"| Zero Canonical Recall questions | {summary['zero_canonical_recall_questions']} |",
        f"| Low recall but high precision questions | {summary['low_recall_high_precision_questions']} |",
        f"| Perfect precision but zero recall questions | {summary['perfect_precision_zero_recall_questions']} |",
        f"| Total expected canonical atoms | {summary['total_expected_canonical_atoms']} |",
        f"| Total missed canonical atoms | {summary['total_missed_canonical_atoms']} |",
        f"| Retrieved relevant non-canonical atoms | {summary['total_retrieved_relevant_noncanonical_atoms']} |",
        f"| Average matching atom pool per question | {summary['average_matching_atom_count']:.1f} |",
        "",
        "## Category counts",
        "",
        "| Category | Count | Meaning |",
        "|---|---:|---|",
    ])
    meanings = {
        "alternative_relevant_evidence": "Low exact canonical recall, but returned evidence is mostly relevant.",
        "good_canonical_recovery": "Recovered at least half of the curated canonical atoms.",
        "mixed_retrieval": "Some relevant evidence, but ranking/relevance is weaker.",
        "retrieval_failure": "Low canonical recall and weak broader relevance.",
    }
    for category, count in sorted(summary["category_counts"].items()):
        lines.append(f"| {category} | {count} | {meanings.get(category, '')} |")

    lines.extend([
        "",
        "## Main finding",
        "",
        (
            "Low Canonical Recall is mostly not behaving like a pure retrieval failure when precision is high. "
            "The common pattern is that the model retrieves relevant atoms that satisfy the broader predicate, "
            "but those atoms are not the exact curated canonical IDs. That means the next work should inspect missed "
            "canonical atoms and decide whether the canonical labels are too narrow, whether top-k is too small, or "
            "whether ranking needs recall-oriented broadening."
        ),
        "",
        "## Worst canonical-recall questions",
        "",
    ])

    for row in analysis["worst_questions"]:
        lines.extend(question_block(row))

    lines.extend([
        "",
        "## Best canonical-recall questions",
        "",
    ])
    for row in analysis["best_questions"][:5]:
        lines.extend(question_block(row, compact=True))

    return "\n".join(lines) + "\n"


def question_block(row: dict, compact: bool = False) -> list[str]:
    lines = [
        f"### {row['question_id']}: {row['question']}",
        "",
        (
            f"- Category: `{row['category']}` | Canonical Recall: `{pct(row['canonical_recall'])}` | "
            f"Precision: `{pct(row['precision_at_k'])}` | MRR: `{pct(row['mrr'])}` | nDCG: `{pct(row['ndcg'])}`"
        ),
        f"- Matching atom pool: `{row.get('matching_atom_count')}` | Expected canonical atoms: `{row['expected_count']}`",
        f"- Interpretation: {row['interpretation']}",
        "",
    ]
    if compact:
        return lines

    lines.extend(["Top retrieved atoms:", ""])
    for item in row["retrieved"][:5]:
        marker = "canonical" if item["canonical_hit"] else ("relevant" if item["relevant"] else "not relevant")
        lines.append(
            f"- #{item['rank']} `{item['atom_id']}` ({marker}, score {item['score']:.3f}): {truncate(item['statement'])}"
        )
    lines.extend(["", "Missed canonical atoms:", ""])
    for item in row["missed_canonical_atoms"][:5]:
        lines.append(f"- `{item['atom_id']}`: {truncate(item['statement'])}")
    if len(row["missed_canonical_atoms"]) > 5:
        lines.append(f"- ... {len(row['missed_canonical_atoms']) - 5} more missed canonical atoms")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain benchmark retrieval misses from existing result JSON.")
    parser.add_argument("model", help="Model slug, e.g. openai_small or mpnet.")
    parser.add_argument("--mode", choices=["dense", "pipeline"], default="dense")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()

    index_root = Path(args.index_root)
    results_root = Path(args.results_root)
    analysis = analyze_one(args.model, args.mode, args.split, index_root, results_root)

    stem = f"{args.model}_{args.mode}_{args.split}_error_analysis"
    json_path = results_root / f"{stem}.json"
    md_path = results_root / f"{stem}.md"
    write_json(json_path, analysis)
    md_path.write_text(markdown_for(analysis), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
