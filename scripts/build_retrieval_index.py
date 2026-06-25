import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
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


def main():
    parser = argparse.ArgumentParser(description="Build a local semantic retrieval index for Signora feedback atoms.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformers model name or local model path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--output-dir", default=str(DATASET_ROOT / "index"), help="Directory for index artifacts.")
    args = parser.parse_args()

    atoms_path = DATASET_ROOT / "data/processed/feedback_atoms.jsonl"
    items_path = DATASET_ROOT / "data/processed/feedback_items.jsonl"
    output_dir = Path(args.output_dir)

    atoms = read_jsonl(atoms_path)
    items = read_jsonl(items_path)
    items_by_feedback_id = {item.get("feedback_id"): item for item in items}
    metadata, texts = build_metadata(atoms, items_by_feedback_id)

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = normalize_matrix(embeddings.astype("float32"))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "feedback_atom_embeddings.npz", embeddings=embeddings)
    write_jsonl(output_dir / "feedback_atom_metadata.jsonl", metadata)
    write_json(output_dir / "index_manifest.json", {
        "created_at": now_iso(),
        "model": args.model,
        "embedding_file": "feedback_atom_embeddings.npz",
        "metadata_file": "feedback_atom_metadata.jsonl",
        "source_atoms": str(atoms_path.relative_to(ROOT)).replace("\\", "/"),
        "source_items": str(items_path.relative_to(ROOT)).replace("\\", "/"),
        "record_count": len(metadata),
        "embedding_dimensions": int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 0,
    })
    print(json.dumps({
        "status": "ok",
        "records_indexed": len(metadata),
        "model": args.model,
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
