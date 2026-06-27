"""Build the production retrieval index for Signora feedback atoms."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_models import DEFAULT_PROFILE, create_encoder


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"
ATOMS_PATH = DATASET_ROOT / "data" / "processed" / "feedback_atoms.jsonl"
ITEMS_PATH = DATASET_ROOT / "data" / "processed" / "feedback_items.jsonl"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "index"
DOCUMENT_RECIPE_VERSION = "signora-production-retrieval-document-v2"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def document_text(atom: dict, item: dict | None) -> str:
    """Fixed production document recipe for each atom.

    It includes the atom statement, metadata useful for filtered retrieval, and
    source context.  If this changes, the manifest recipe version changes too.
    """
    item = item or {}
    fields = [
        atom.get("statement"),
        atom.get("feedback_type"),
        atom.get("product_area"),
        atom.get("topic"),
        atom.get("subtopic"),
        atom.get("customer_segment"),
        atom.get("severity"),
        atom.get("sentiment"),
        atom.get("evidence_role"),
        item.get("clean_text") or item.get("raw_text"),
        item.get("source_type"),
        item.get("target_product"),
    ]
    return " | ".join(str(value) for value in fields if value)


def build_metadata() -> tuple[list[dict], list[str]]:
    atoms = read_jsonl(ATOMS_PATH)
    items = {item.get("feedback_id"): item for item in read_jsonl(ITEMS_PATH)}
    rows = []
    texts = []

    for position, atom in enumerate(atoms):
        item = items.get(atom.get("feedback_id"), {})
        text = document_text(atom, item)
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


def chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Signora production retrieval index.")
    parser.add_argument(
        "--profile",
        default=None,
        help=f"Embedding profile. Default: {DEFAULT_PROFILE}. Local fallback: mpnet.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for index artifacts.")
    args = parser.parse_args()

    profile_slug, profile, encode = create_encoder(args.profile)
    metadata, texts = build_metadata()
    vectors = []
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size

    for number, batch in enumerate(chunks(texts, args.batch_size), 1):
        print(f"[{profile_slug}] embedding document batch {number}/{total_batches}")
        vectors.append(encode(batch, "document"))

    embeddings = np.vstack(vectors).astype("float32")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "feedback_atom_embeddings.npz", embeddings=embeddings)
    write_jsonl(output_dir / "feedback_atom_metadata.jsonl", metadata)
    write_json(output_dir / "index_manifest.json", {
        "format_version": "signora_retrieval_index_v2",
        "created_at": now_iso(),
        "embedding_profile": profile_slug,
        "model": profile["name"],
        "backend": profile["backend"],
        "benchmark_role": profile["role"],
        "benchmark_note": profile["benchmark_note"],
        "embedding_file": "feedback_atom_embeddings.npz",
        "metadata_file": "feedback_atom_metadata.jsonl",
        "source_atoms": str(ATOMS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_items": str(ITEMS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_fingerprint": {
            "atoms_sha256": sha256_file(ATOMS_PATH),
            "items_sha256": sha256_file(ITEMS_PATH),
            "document_recipe": DOCUMENT_RECIPE_VERSION,
        },
        "record_count": len(metadata),
        "embedding_dimensions": int(embeddings.shape[1]),
        "retrieval_defaults": {
            "semantic_weight": 0.75,
            "keyword_weight": 0.25,
            "diverse_by_feedback_id": True,
        },
    })

    print(json.dumps({
        "status": "ok",
        "records_indexed": len(metadata),
        "embedding_profile": profile_slug,
        "model": profile["name"],
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
