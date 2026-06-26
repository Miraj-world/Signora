"""Shared, provider-neutral benchmark primitives."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset"
ATOMS_PATH = DATASET_ROOT / "data" / "processed" / "feedback_atoms.jsonl"
ITEMS_PATH = DATASET_ROOT / "data" / "processed" / "feedback_items.jsonl"
EVAL_DIR = DATASET_ROOT / "data" / "evaluation"
DEFAULT_INDEX_ROOT = DATASET_ROOT / "benchmark_indexes"
DEFAULT_RESULTS_ROOT = DATASET_ROOT / "benchmark_results"


def load_dotenv(path=None):
    """Load local benchmark credentials without overwriting real environment values."""
    path = Path(path) if path else ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize(values):
    values = np.asarray(values, dtype="float32")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def document_text(atom, item):
    """The fixed document recipe used for every provider in benchmark v2."""
    fields = [
        atom.get("statement"), atom.get("feedback_type"), atom.get("product_area"),
        atom.get("topic"), atom.get("subtopic"), atom.get("customer_segment"),
        atom.get("severity"), (item or {}).get("clean_text"), (item or {}).get("source_type"),
    ]
    return " | ".join(str(value) for value in fields if value)


def document_recipe_hash():
    # Changing document_text requires changing this version, which makes stale
    # indexes fail validation rather than be compared accidentally.
    return hashlib.sha256(b"signora-benchmark-document-v1").hexdigest()


def load_corpus():
    atoms = read_jsonl(ATOMS_PATH)
    items = {item.get("feedback_id"): item for item in read_jsonl(ITEMS_PATH)}
    rows = []
    for position, atom in enumerate(atoms):
        item = items.get(atom.get("feedback_id"), {})
        rows.append({
            "position": position,
            "atom_id": atom.get("atom_id"),
            "feedback_id": atom.get("feedback_id"),
            "statement": atom.get("statement"),
            "topic": atom.get("topic"),
            "subtopic": atom.get("subtopic"),
            "customer_segment": atom.get("customer_segment"),
            "severity": atom.get("severity"),
            "product_area": atom.get("product_area"),
            "feedback_type": atom.get("feedback_type"),
            "search_text": document_text(atom, item),
        })
    return rows


def corpus_fingerprint():
    return {"atoms_sha256": sha256_file(ATOMS_PATH), "items_sha256": sha256_file(ITEMS_PATH), "document_recipe_sha256": document_recipe_hash()}


def load_index(index_dir):
    index_dir = Path(index_dir)
    manifest = read_json(index_dir / "index_manifest.json")
    metadata = read_jsonl(index_dir / manifest["metadata_file"])
    embeddings = np.load(index_dir / manifest["embedding_file"])["embeddings"]
    return manifest, metadata, embeddings


def row_matches_filters(row, filters):
    return all(not allowed or row.get(field) in allowed for field, allowed in (filters or {}).items())


def row_matches_relevance(row, rules):
    return bool(rules) and all(row.get(key[:-4]) in allowed for key, allowed in rules.items() if key.endswith("_any"))
