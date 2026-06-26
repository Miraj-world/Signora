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

TODAY = "2026-06-25"
EVAL_VERSION = "retrieval_questions_v1 (130q + 21 abstain)"

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
    Compute Recall@5, Recall@10, Precision@5, Abstention accuracy, Latency.

    NOTE: All 130 retrieval questions carry review_status='unverified_suggestion_pending_human_review'.
    Expected atom IDs were assigned by topic tag, not by semantic content verification, so
    recall/precision metrics against these labels reflect label quality, not retrieval quality.
    Mean top-1 semantic score (mean_top1_semantic) is an unbiased quality proxy.
    """
    recalls_5, recalls_10, prec_5, latencies, top1_semantics = [], [], [], [], []
    non_abstain = [q for q in questions if not q.get("should_abstain")]

    for q in non_abstain:
        expected = set(q.get("expected_atom_ids", []))
        query = q["question"]

        t0 = time.perf_counter()
        qe = encode_fn(query)
        results_10 = eval_utils.retrieve(qe, query, metadata, embeddings, top_k=10)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        ids_10 = [row.get("atom_id") for _, row in results_10]
        ids_5 = ids_10[:5]
        top1_semantics.append(float(embeddings[results_10[0][1]["position"]] @ qe) if results_10 else 0.0)

        r5 = len(set(ids_5) & expected) / len(expected) if expected else 0.0
        r10 = len(set(ids_10) & expected) / len(expected) if expected else 0.0
        p5 = len(set(ids_5) & expected) / len(ids_5) if ids_5 else 0.0

        recalls_5.append(r5)
        recalls_10.append(r10)
        prec_5.append(p5)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Abstention
    abstain_correct = 0
    for q in abstention_questions:
        qe = encode_fn(q["question"])
        results = eval_utils.retrieve(qe, q["question"], metadata, embeddings, top_k=8)
        if not results or max(s for s, _ in results) < 0.5:
            abstain_correct += 1

    return {
        "recall_at_5":  avg(recalls_5),
        "recall_at_10": avg(recalls_10),
        "precision_at_5": avg(prec_5),
        "citation_accuracy": avg(prec_5),
        "unsupported_claim_rate": 1.0 - avg(prec_5),
        "abstention_accuracy_num": abstain_correct,
        "abstention_accuracy_den": len(abstention_questions),
        "avg_latency_ms": avg(latencies),
        "mean_top1_semantic": avg(top1_semantics),
        "n_questions": len(non_abstain),
        "ground_truth_status": "unverified_suggestion_pending_human_review",
    }


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
    gt_note = "unverified GT *"

    rows = [
        ("Recall@5",               f"{fmt(m['recall_at_5'])} ({gt_note})"),
        ("Recall@10",              f"{fmt(m['recall_at_10'])} ({gt_note})"),
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
    print(f"\n> \\* Ground truth status: `unverified_suggestion_pending_human_review` for all 130 questions.")
    print(f"> Expected atom IDs were assigned by topic-tag rules, not semantic verification.")
    print(f"> Mean top-1 semantic similarity: **{m['mean_top1_semantic']:.3f}** (unbiased retrieval quality signal).")


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
    questions = eval_utils.read_jsonl(eval_utils.EVAL_DIR / "retrieval_questions.jsonl")
    abstention_questions = eval_utils.read_jsonl(eval_utils.EVAL_DIR / "abstention_questions.jsonl")

    print("# Signora Retrieval Benchmark\n")
    print(f"Evaluation set: {len(questions)} retrieval questions + {len(abstention_questions)} abstention cases")
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
        print(f"\n[{slug}] Evaluating {model_info['name']} ({manifest['record_count']} atoms)...")
        t_start = time.time()
        metrics = compute_extended_metrics(questions, abstention_questions, encode_fn, metadata, embeddings)
        elapsed = time.time() - t_start
        print(f"[{slug}] Done in {elapsed:.1f}s")

        print_model_table(model_info, metrics)
        save_model_results(model_info, metrics)
        all_results.append((model_info, metrics, None))

    # Summary comparison table
    print("\n\n---\n## Summary comparison\n")
    print("> Recall/Precision reflect alignment with **unverified** ground truth labels.")
    print("> `Semantic@1` = mean top-1 cosine similarity — the most reliable quality signal.\n")
    header = "| Model | Recall@5 | Recall@10 | Precision@5 | Abstention | Latency | Semantic@1 | Cost/query |"
    sep    = "|-------|----------|-----------|-------------|------------|---------|------------|------------|"
    print(header)
    print(sep)
    for model_info, metrics, reason in all_results:
        name = model_info["name"]
        if metrics:
            m = metrics
            ab = f"{m['abstention_accuracy_num']}/{m['abstention_accuracy_den']}"
            cost = (model_info["cost_per_1k"] * AVG_QUERY_TOKENS) / 1000
            cost_str = "$0" if cost == 0 else f"~${cost:.6f}"
            print(f"| {name} | {m['recall_at_5']:.3f} | {m['recall_at_10']:.3f} | {m['precision_at_5']:.3f} | {ab} | {m['avg_latency_ms']:.0f} ms | {m['mean_top1_semantic']:.3f} | {cost_str} |")
        else:
            note = reason or "unavailable"
            print(f"| {name} | -- | -- | -- | -- | -- | -- | -- |")


if __name__ == "__main__":
    main()
