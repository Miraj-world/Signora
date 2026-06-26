import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(env_path=None):
    path = Path(env_path) if env_path else ROOT / ".env"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
DATASET_ROOT = ROOT / "dataset"
EVAL_DIR = DATASET_ROOT / "data/evaluation"
RESULTS_DIR = DATASET_ROOT / "results"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_index(index_dir):
    index_dir = Path(index_dir)
    if not index_dir.exists():
        return None, None, None
    manifest = read_json(index_dir / "index_manifest.json")
    metadata = read_jsonl(index_dir / manifest["metadata_file"])
    embeddings = np.load(index_dir / manifest["embedding_file"])["embeddings"]
    return manifest, metadata, embeddings


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9][a-z0-9_-]*", (text or "").lower()) if len(t) > 1]


def keyword_score(query_tokens, text):
    if not query_tokens:
        return 0.0
    text_counts = Counter(tokenize(text))
    if not text_counts:
        return 0.0
    overlap = sum(min(text_counts[t], c) for t, c in query_tokens.items())
    return overlap / max(1, sum(query_tokens.values()))


def retrieve(query_embedding, query_text, metadata, embeddings, top_k=8, semantic_weight=0.75):
    semantic_scores = embeddings @ query_embedding
    query_tokens = Counter(tokenize(query_text))
    candidates = []
    for row in metadata:
        position = row["position"]
        semantic = float(semantic_scores[position])
        keyword = keyword_score(query_tokens, row.get("search_text", ""))
        fused = semantic_weight * semantic + (1 - semantic_weight) * keyword
        candidates.append((fused, row))
    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen_feedback_ids = set()
    seen_statements = set()
    for fused, row in candidates:
        fid = row.get("feedback_id")
        sk = " ".join(tokenize(row.get("statement", "")))
        if fid in seen_feedback_ids:
            continue
        if sk and sk in seen_statements:
            continue
        seen_feedback_ids.add(fid)
        seen_statements.add(sk)
        selected.append((fused, row))
        if len(selected) >= top_k:
            break
    return selected


def compute_metrics(questions, encode_fn, metadata, embeddings, top_k=8, semantic_weight=0.75):
    groups = {"easy": [], "medium": [], "hard": []}
    all_recall = []
    all_precision = []
    all_mrr = []
    all_hit = []
    all_ndcg = []

    for q in questions:
        if q.get("should_abstain"):
            continue
        expected = set(q.get("expected_atom_ids", []))
        query = q["question"]
        difficulty = q.get("difficulty", "medium")

        qe = encode_fn(query)
        results = retrieve(qe, query, metadata, embeddings, top_k=top_k, semantic_weight=semantic_weight)
        returned_ids = [row.get("atom_id") for _, row in results]

        recall = len(set(returned_ids) & expected) / len(expected) if expected else 0.0
        precision = len(set(returned_ids) & expected) / len(returned_ids) if returned_ids else 0.0

        mrr = 0.0
        for rank, aid in enumerate(returned_ids, 1):
            if aid in expected:
                mrr = 1.0 / rank
                break

        hit = 1.0 if any(aid in expected for aid in returned_ids) else 0.0

        dcg = sum(1.0 / math.log2(rank + 1) for rank, aid in enumerate(returned_ids, 1) if aid in expected)
        ideal_hits = min(len(expected), top_k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        all_recall.append(recall)
        all_precision.append(precision)
        all_mrr.append(mrr)
        all_hit.append(hit)
        all_ndcg.append(ndcg)

        if difficulty in groups:
            groups[difficulty].append(recall)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    overall = {
        "recall_at_8": avg(all_recall),
        "precision_at_8": avg(all_precision),
        "mrr": avg(all_mrr),
        "hit_rate_at_8": avg(all_hit),
        "ndcg_at_8": avg(all_ndcg),
        "questions_evaluated": len(all_recall),
    }
    by_difficulty = {k: avg(v) for k, v in groups.items()}
    return overall, by_difficulty


def compute_abstention(abstention_questions, encode_fn, metadata, embeddings, top_k=8, semantic_weight=0.75):
    correct = 0
    for q in abstention_questions:
        qe = encode_fn(q["question"])
        results = retrieve(qe, q["question"], metadata, embeddings, top_k=top_k, semantic_weight=semantic_weight)
        if not results or max(score for score, _ in results) < 0.5:
            correct += 1
    return correct, len(abstention_questions)


def print_results(model_name, overall, by_difficulty, abstention_correct, abstention_total, elapsed):
    print(f"Model: {model_name}")
    print(f"  Overall Recall@8:    {overall['recall_at_8']:.2f}")
    print(f"  Easy    Recall@8:    {by_difficulty['easy']:.2f}")
    print(f"  Medium  Recall@8:    {by_difficulty['medium']:.2f}")
    print(f"  Hard    Recall@8:    {by_difficulty['hard']:.2f}")
    print(f"  Precision@8:         {overall['precision_at_8']:.2f}")
    print(f"  Hit Rate@8:          {overall['hit_rate_at_8']:.2f}")
    print(f"  MRR:                 {overall['mrr']:.2f}")
    print(f"  NDCG@8:              {overall['ndcg_at_8']:.2f}")
    print(f"  Abstention accuracy: {abstention_correct}/{abstention_total}")
    print(f"  Questions evaluated: {overall['questions_evaluated']}")
    print(f"  Total time: {int(elapsed)}s")


def save_results(slug, model_name, overall, by_difficulty, abstention_correct, abstention_total, elapsed):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model_name,
        "overall": overall,
        "by_difficulty": by_difficulty,
        "abstention_correct": abstention_correct,
        "abstention_total": abstention_total,
        "elapsed_seconds": int(elapsed),
    }
    (RESULTS_DIR / f"{slug}_eval.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
