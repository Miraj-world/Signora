"""
Runs evaluation for every available model index and prints results tables.
Local models (MiniLM, MPNet) are evaluated fully.
API models require their SDK and API key — otherwise marked as unavailable.
"""
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset"
RESULTS_DIR = DATASET_ROOT / "results"

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils

TODAY = "2026-06-29"
EVAL_VERSION = "retrieval_questions_v2_predicate_backed (59q + 20 abstain)"

MODELS = [
    {"slug": "minilm",       "name": "all-MiniLM-L6-v2",        "backend": "st",     "dims": 384,  "cost_per_1k": 0.0},
    {"slug": "mpnet",        "name": "all-mpnet-base-v2",        "backend": "st",     "dims": 768,  "cost_per_1k": 0.0},
    {"slug": "openai_small", "name": "text-embedding-3-small",   "backend": "openai", "dims": 1536, "cost_per_1k": 0.00002},
    {"slug": "openai_large", "name": "text-embedding-3-large",   "backend": "openai", "dims": 3072, "cost_per_1k": 0.00013},
    {"slug": "cohere",       "name": "embed-english-v3.0",       "backend": "cohere", "dims": 1024, "cost_per_1k": 0.0001},
    {"slug": "google",       "name": "text-embedding-004",       "backend": "google", "dims": 768,  "cost_per_1k": 0.000025},
    {"slug": "voyage",       "name": "voyage-large-2",           "backend": "voyage", "dims": 1536, "cost_per_1k": 0.00012},
]

AVG_QUERY_TOKENS = 10


def get_encode_fn(model_info):
    backend = model_info["backend"]
    name = model_info["name"]

    if backend == "st":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(name)
            def encode_fn(text):
                return eval_utils.normalize_vector(
                    model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
                )
            return encode_fn, None
        except Exception as e:
            return None, f"sentence-transformers error: {e}"

    if backend == "openai":
        try:
            from openai import OpenAI
            if not os.environ.get("OPENAI_API_KEY"):
                return None, "OPENAI_API_KEY not set"
            client = OpenAI()
            def encode_fn(text):
                r = client.embeddings.create(input=[text], model=name)
                return eval_utils.normalize_vector(np.array(r.data[0].embedding, dtype="float32"))
            return encode_fn, None
        except ImportError:
            return None, "openai package not installed"

    if backend == "cohere":
        try:
            import cohere
            if not os.environ.get("COHERE_API_KEY"):
                return None, "COHERE_API_KEY not set"
            co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
            def encode_fn(text):
                r = co.embed(texts=[text], model=name, input_type="search_query")
                embs = r.embeddings
                if hasattr(embs, "float_"):
                    embs = embs.float_
                return eval_utils.normalize_vector(np.array(embs[0], dtype="float32"))
            return encode_fn, None
        except ImportError:
            return None, "cohere package not installed"

    if backend == "google":
        try:
            import google.generativeai as genai
            if not os.environ.get("GOOGLE_API_KEY"):
                return None, "GOOGLE_API_KEY not set"
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            def encode_fn(text):
                r = genai.embed_content(model=f"models/{name}", content=text, task_type="retrieval_query")
                return eval_utils.normalize_vector(np.array(r["embedding"], dtype="float32"))
            return encode_fn, None
        except ImportError:
            return None, "google-generativeai package not installed"

    if backend == "voyage":
        try:
            import voyageai
            if not os.environ.get("VOYAGE_API_KEY"):
                return None, "VOYAGE_API_KEY not set"
            vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
            def encode_fn(text):
                r = vo.embed([text], model=name, input_type="query")
                return eval_utils.normalize_vector(np.array(r.embeddings[0], dtype="float32"))
            return encode_fn, None
        except ImportError:
            return None, "voyageai package not installed"

    return None, f"unknown backend: {backend}"


def compute_extended_metrics(questions, abstention_questions, encode_fn, metadata, embeddings):
    """
    Compute canonical citation recall, predicate coverage, precision,
    abstention accuracy, and latency.

    V2 uses corpus-verified metadata predicates for relevance. Canonical atom IDs
    remain useful citation examples, while precision accepts all matching evidence.
    """
    recalls_5, recalls_10, recalls_20, recalls_50 = [], [], [], []
    predicate_coverage_20, predicate_coverage_50 = [], []
    predicate_hit_20, predicate_hit_50 = [], []
    prec_5, latencies, top1_semantics = [], [], []
    non_abstain = [q for q in questions if not q.get("should_abstain")]

    for q in non_abstain:
        expected = set(q.get("expected_atom_ids", []))
        query = q["question"]

        t0 = time.perf_counter()
        qe = encode_fn(query)
        results_50 = eval_utils.retrieve(qe, query, metadata, embeddings, top_k=50, filters=q.get("filters"))
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        ids_50 = [row.get("atom_id") for _, row in results_50]
        ids_5 = ids_50[:5]
        ids_10 = ids_50[:10]
        ids_20 = ids_50[:20]
        top1_semantics.append(float(embeddings[results_50[0][1]["position"]] @ qe) if results_50 else 0.0)

        # Canonical citation recall is reported separately. Precision measures
        # all corpus rows satisfying the question's verified relevance rule.
        r5 = len(set(ids_5) & expected) / len(expected) if expected else 0.0
        r10 = len(set(ids_10) & expected) / len(expected) if expected else 0.0
        r20 = len(set(ids_20) & expected) / len(expected) if expected else 0.0
        r50 = len(set(ids_50) & expected) / len(expected) if expected else 0.0
        relevance = q.get("relevance_rules")
        p5 = (sum(eval_utils.row_matches_relevance(row, relevance) for _, row in results_50[:5]) / len(ids_5)) if ids_5 and relevance else (len(set(ids_5) & expected) / len(ids_5) if ids_5 else 0.0)

        matching_count = q.get("matching_atom_count") or len(expected)
        predicate_hits_20 = sum(eval_utils.row_matches_relevance(row, relevance) for _, row in results_50[:20]) if relevance else len(set(ids_20) & expected)
        predicate_hits_50 = sum(eval_utils.row_matches_relevance(row, relevance) for _, row in results_50[:50]) if relevance else len(set(ids_50) & expected)
        predicate_denom_20 = min(20, matching_count)
        predicate_denom_50 = min(50, matching_count)
        pc20 = predicate_hits_20 / predicate_denom_20 if predicate_denom_20 else 0.0
        pc50 = predicate_hits_50 / predicate_denom_50 if predicate_denom_50 else 0.0

        recalls_5.append(r5)
        recalls_10.append(r10)
        recalls_20.append(r20)
        recalls_50.append(r50)
        predicate_coverage_20.append(pc20)
        predicate_coverage_50.append(pc50)
        predicate_hit_20.append(1.0 if predicate_hits_20 else 0.0)
        predicate_hit_50.append(1.0 if predicate_hits_50 else 0.0)
        prec_5.append(p5)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "recall_at_5":  avg(recalls_5),
        "recall_at_10": avg(recalls_10),
        "recall_at_20": avg(recalls_20),
        "recall_at_50": avg(recalls_50),
        "predicate_coverage_at_20": avg(predicate_coverage_20),
        "predicate_coverage_at_50": avg(predicate_coverage_50),
        "predicate_hit_at_20": avg(predicate_hit_20),
        "predicate_hit_at_50": avg(predicate_hit_50),
        "precision_at_5": avg(prec_5),
        "citation_accuracy": avg(prec_5),
        "unsupported_claim_rate": 1.0 - avg(prec_5),
        "avg_latency_ms": avg(latencies),
        "mean_top1_semantic": avg(top1_semantics),
        "n_questions": len(non_abstain),
        "ground_truth_status": "predicate_verified_from_corpus_v2",
    }


def top_score(question, encode_fn, metadata, embeddings):
    qe = encode_fn(question["question"])
    results = eval_utils.retrieve(qe, question["question"], metadata, embeddings, top_k=1, filters=question.get("filters"))
    return results[0][0] if results else float("-inf")


def calibrate_and_score_abstention(validation_questions, validation_abstention, test_questions, test_abstention, encode_fn, metadata, embeddings):
    """Calibrate a separate no-answer threshold per model, then score only held-out cases."""
    positives = [top_score(q, encode_fn, metadata, embeddings) for q in validation_questions]
    negatives = [top_score(q, encode_fn, metadata, embeddings) for q in validation_abstention]
    candidates = sorted(set(positives + negatives))
    if not candidates:
        return 0.5, 0, len(test_abstention)
    threshold = max(candidates, key=lambda t: ((sum(s >= t for s in positives) / len(positives)) + (sum(s < t for s in negatives) / len(negatives))) / 2)
    correct = sum(top_score(q, encode_fn, metadata, embeddings) < threshold for q in test_abstention)
    return threshold, correct, len(test_abstention)


def fmt(val, kind="pct"):
    if val is None:
        return "—"
    if kind == "pct":
        return f"{val:.3f}"
    if kind == "ms":
        return f"{val:.1f} ms"
    if kind == "usd":
        return f"${val:.6f}"
    return str(val)


def print_model_table(model_info, metrics, reason=None):
    name = model_info["name"]
    slug = model_info["slug"]
    print(f"\n### {name}  (`{slug}`)\n")
    print(f"| Metric                  | Current result     | Evaluation version                         | Last updated |")
    print(f"|-------------------------|--------------------|--------------------------------------------|--------------|")

    if reason:
        rows = [
            ("Recall@5",               "--"),
            ("Recall@10",              "--"),
            ("Recall@20",              "--"),
            ("Recall@50",              "--"),
            ("Predicate coverage@20",  "--"),
            ("Predicate coverage@50",  "--"),
            ("Precision@5",            "--"),
            ("Citation accuracy",      "--"),
            ("Unsupported-claim rate", "--"),
            ("Abstention accuracy",    "--"),
            ("Average latency",        "--"),
            ("Cost per query",         "--"),
        ]
        for i, (metric, val) in enumerate(rows):
            ev = f"not run: {reason}" if i == 0 else ""
            lu = TODAY if i == 0 else ""
            print(f"| {metric:<23} | {val:<18} | {ev:<42} | {lu:<12} |")
        return

    m = metrics
    cost = (model_info["cost_per_1k"] * AVG_QUERY_TOKENS) / 1000
    cost_str = "$0.000000 (local)" if cost == 0 else f"~${cost:.7f}"
    abstain_str = f"{m['abstention_accuracy_num']}/{m['abstention_accuracy_den']}"
    gt_note = "canonical citations"

    rows = [
        ("Recall@5",               f"{fmt(m['recall_at_5'])} ({gt_note})"),
        ("Recall@10",              f"{fmt(m['recall_at_10'])} ({gt_note})"),
        ("Recall@20",              f"{fmt(m['recall_at_20'])} ({gt_note})"),
        ("Recall@50",              f"{fmt(m['recall_at_50'])} ({gt_note})"),
        ("Predicate coverage@20",  f"{fmt(m['predicate_coverage_at_20'])} (verified predicate)"),
        ("Predicate coverage@50",  f"{fmt(m['predicate_coverage_at_50'])} (verified predicate)"),
        ("Precision@5",            f"{fmt(m['precision_at_5'])} ({gt_note})"),
        ("Citation accuracy",      f"{fmt(m['citation_accuracy'])} ({gt_note})"),
        ("Unsupported-claim rate", f"{fmt(m['unsupported_claim_rate'])} ({gt_note})"),
        ("Abstention accuracy",    abstain_str),
        ("Average latency",        fmt(m["avg_latency_ms"], "ms")),
        ("Cost per query",         cost_str),
    ]
    for i, (metric, val) in enumerate(rows):
        ev = EVAL_VERSION if i == 0 else ""
        lu = TODAY if i == 0 else ""
        print(f"| {metric:<23} | {val:<18} | {ev:<42} | {lu:<12} |")
    print(f"\n> Canonical IDs are verified citations; predicate coverage uses all corpus rows matching the question's metadata predicate.")
    print(f"> Abstention thresholds are calibrated per model on validation, then scored on test.")
    print(f"> Mean top-1 semantic similarity: **{m['mean_top1_semantic']:.3f}**.")


def save_model_results(model_info, metrics, reason=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_info["slug"]
    data = {
        "model": model_info["name"],
        "slug": slug,
        "evaluation_version": EVAL_VERSION,
        "last_updated": TODAY,
        "status": "unavailable" if reason else "ok",
        "reason": reason,
        "metrics": metrics,
    }
    (RESULTS_DIR / f"{slug}_benchmark.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    eval_utils.load_dotenv()
    questions = eval_utils.read_jsonl(eval_utils.EVAL_DIR / "retrieval_questions_v2.jsonl")
    abstention_questions = eval_utils.read_jsonl(eval_utils.EVAL_DIR / "abstention_questions_v2.jsonl")
    test_questions = [q for q in questions if q.get("split") == "test"]
    validation_questions = [q for q in questions if q.get("split") == "validation"]
    test_abstention = [q for q in abstention_questions if q.get("split") == "test"]
    validation_abstention = [q for q in abstention_questions if q.get("split") == "validation"]

    print("# Signora Retrieval Benchmark\n")
    print(f"Evaluation set: {len(test_questions)} test retrieval questions + {len(test_abstention)} test abstention cases")
    print(f"Date: {TODAY}\n")

    all_results = []

    for model_info in MODELS:
        slug = model_info["slug"]
        index_dir = DATASET_ROOT / "index" / slug

        if not index_dir.exists():
            reason = f"index not built — run embed_{slug}.py first"
            print_model_table(model_info, None, reason=reason)
            save_model_results(model_info, None, reason=reason)
            all_results.append((model_info, None, reason))
            continue

        encode_fn, reason = get_encode_fn(model_info)
        if encode_fn is None:
            print_model_table(model_info, None, reason=reason)
            save_model_results(model_info, None, reason=reason)
            all_results.append((model_info, None, reason))
            continue

        manifest, metadata, embeddings = eval_utils.load_index(index_dir)
        if manifest.get("model") != model_info["name"]:
            reason = f"index model mismatch: manifest has {manifest.get('model')}, expected {model_info['name']}"
            print_model_table(model_info, None, reason=reason)
            save_model_results(model_info, None, reason=reason)
            all_results.append((model_info, None, reason))
            continue
        if embeddings.ndim != 2 or embeddings.shape[1] != model_info["dims"]:
            reason = f"index dimension mismatch: index has {embeddings.shape[1] if embeddings.ndim == 2 else 'invalid'}, expected {model_info['dims']}"
            print_model_table(model_info, None, reason=reason)
            save_model_results(model_info, None, reason=reason)
            all_results.append((model_info, None, reason))
            continue
        eval_utils.validate_benchmark(questions, metadata, embeddings, model_info["name"])
        print(f"\n[{slug}] Evaluating {model_info['name']} ({manifest['record_count']} atoms)...")
        t_start = time.time()
        metrics = compute_extended_metrics(test_questions, test_abstention, encode_fn, metadata, embeddings)
        threshold, abstain_correct, abstain_total = calibrate_and_score_abstention(validation_questions, validation_abstention, test_questions, test_abstention, encode_fn, metadata, embeddings)
        metrics["abstention_accuracy_num"] = abstain_correct
        metrics["abstention_accuracy_den"] = abstain_total
        metrics["abstention_threshold"] = threshold
        elapsed = time.time() - t_start
        print(f"[{slug}] Done in {elapsed:.1f}s")

        print_model_table(model_info, metrics)
        save_model_results(model_info, metrics)
        all_results.append((model_info, metrics, None))

    # Summary comparison table
    print("\n\n---\n## Summary comparison\n")
    print("> Recall measures canonical-citation coverage; predicate coverage measures broader verified evidence coverage.")
    print("> `Semantic@1` = mean top-1 cosine similarity.\n")
    header = "| Model | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Predicate cov@20 | Predicate cov@50 | Precision@5 | Abstention | Latency | Semantic@1 | Cost/query |"
    sep    = "|-------|----------|-----------|-----------|-----------|------------------|------------------|-------------|------------|---------|------------|------------|"
    print(header)
    print(sep)
    for model_info, metrics, reason in all_results:
        name = model_info["name"]
        if metrics:
            m = metrics
            ab = f"{m['abstention_accuracy_num']}/{m['abstention_accuracy_den']}"
            cost = (model_info["cost_per_1k"] * AVG_QUERY_TOKENS) / 1000
            cost_str = "$0" if cost == 0 else f"~${cost:.6f}"
            print(f"| {name} | {m['recall_at_5']:.3f} | {m['recall_at_10']:.3f} | {m['recall_at_20']:.3f} | {m['recall_at_50']:.3f} | {m['predicate_coverage_at_20']:.3f} | {m['predicate_coverage_at_50']:.3f} | {m['precision_at_5']:.3f} | {ab} | {m['avg_latency_ms']:.0f} ms | {m['mean_top1_semantic']:.3f} | {cost_str} |")
        else:
            note = reason or "unavailable"
            print(f"| {name} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")


if __name__ == "__main__":
    main()
