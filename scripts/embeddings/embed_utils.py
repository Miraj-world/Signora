import json
import os
from datetime import datetime, timezone
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


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_matrix(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def search_text(atom, item):
    parts = [
        atom.get("statement"),
        atom.get("feedback_type"),
        atom.get("product_area"),
        atom.get("topic"),
        atom.get("subtopic"),
        atom.get("customer_segment"),
        atom.get("severity"),
        item.get("clean_text") if item else None,
        item.get("source_type") if item else None,
        item.get("target_product") if item else None,
    ]
    return " | ".join(str(part) for part in parts if part)


def build_metadata(atoms, items_by_feedback_id):
    rows = []
    texts = []
    for position, atom in enumerate(atoms):
        item = items_by_feedback_id.get(atom.get("feedback_id"), {})
        text = search_text(atom, item)
        texts.append(text)
        rows.append({
            "position": position,
            "atom_id": atom.get("atom_id"),
            "feedback_id": atom.get("feedback_id"),
            "statement": atom.get("statement"),
            "feedback_type": atom.get("feedback_type"),
            "product_area": atom.get("product_area"),
            "topic": atom.get("topic"),
            "subtopic": atom.get("subtopic"),
            "customer_segment": atom.get("customer_segment"),
            "sentiment": atom.get("sentiment"),
            "severity": atom.get("severity"),
            "evidence_role": atom.get("evidence_role"),
            "product_version": atom.get("product_version"),
            "source_type": item.get("source_type"),
            "source_name": item.get("source_name"),
            "source_url": item.get("source_url"),
            "source_context": item.get("clean_text") or item.get("raw_text"),
            "created_at": item.get("created_at"),
            "target_product": item.get("target_product"),
            "thread_id": item.get("thread_id"),
            "parent_id": item.get("parent_id"),
            "is_root_post": item.get("is_root_post"),
            "search_text": text,
        })
    return rows, texts


def load_atoms_and_items(dataset_root):
    atoms = read_jsonl(dataset_root / "data/processed/feedback_atoms.jsonl")
    items = read_jsonl(dataset_root / "data/processed/feedback_items.jsonl")
    items_by_feedback_id = {item.get("feedback_id"): item for item in items}
    return atoms, items_by_feedback_id


def save_index(output_dir, embeddings, metadata, model_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "feedback_atom_embeddings.npz", embeddings=embeddings)
    write_jsonl(output_dir / "feedback_atom_metadata.jsonl", metadata)
    write_json(output_dir / "index_manifest.json", {
        "model": model_name,
        "embedding_dimensions": int(embeddings.shape[1]),
        "record_count": len(metadata),
        "created_at": now_iso(),
        "embedding_file": "feedback_atom_embeddings.npz",
        "metadata_file": "feedback_atom_metadata.jsonl",
    })
    print(f"Index written to {output_dir}")
