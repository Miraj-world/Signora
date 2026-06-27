"""Build self-describing, model-specific indexes from one immutable corpus snapshot."""
import argparse
from pathlib import Path

import numpy as np

from common import DEFAULT_INDEX_ROOT, corpus_fingerprint, load_corpus, now_iso, write_json, write_jsonl
from models import available_slugs, create_encoder


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def build(slug, index_root, batch_size):
    spec, encode = create_encoder(slug)
    rows = load_corpus()
    vectors = []
    texts = [row["search_text"] for row in rows]
    for number, batch in enumerate(chunks(texts, batch_size), 1):
        print(f"[{slug}] embedding batch {number}/{(len(texts) + batch_size - 1) // batch_size}")
        vectors.append(encode(batch, "document"))
    embeddings = np.vstack(vectors).astype("float32")
    output = Path(index_root) / slug
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "embeddings.npz", embeddings=embeddings)
    write_jsonl(output / "metadata.jsonl", rows)
    write_json(output / "index_manifest.json", {
        "format_version": "benchmark_index_v2",
        "model_slug": slug, "model": spec["name"], "backend": spec["backend"],
        "created_at": now_iso(), "record_count": len(rows), "embedding_dimensions": int(embeddings.shape[1]),
        "embedding_file": "embeddings.npz", "metadata_file": "metadata.jsonl",
        "corpus_fingerprint": corpus_fingerprint(),
    })
    print(f"[{slug}] wrote {len(rows)} vectors to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=available_slugs())
    parser.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    for slug in args.models:
        build(slug, args.index_root, args.batch_size)


if __name__ == "__main__":
    main()
