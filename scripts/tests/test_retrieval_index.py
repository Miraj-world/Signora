import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = ROOT / "dataset" / "index"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_retrieval_index_artifacts_exist_after_build():
    manifest_path = INDEX_ROOT / "index_manifest.json"
    metadata_path = INDEX_ROOT / "feedback_atom_metadata.jsonl"
    embeddings_path = INDEX_ROOT / "feedback_atom_embeddings.npz"

    assert manifest_path.exists()
    assert metadata_path.exists()
    assert embeddings_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = read_jsonl(metadata_path)
    assert manifest["record_count"] == len(metadata)
    assert manifest["record_count"] > 0
    assert manifest["embedding_dimensions"] > 0


def test_retrieval_metadata_has_citation_fields():
    metadata = read_jsonl(INDEX_ROOT / "feedback_atom_metadata.jsonl")
    sample = metadata[0]
    assert sample["atom_id"]
    assert sample["feedback_id"]
    assert sample["statement"]
    assert "source_type" in sample
    assert "source_url" in sample
    assert "source_context" in sample
    assert "search_text" in sample


if __name__ == "__main__":
    test_retrieval_index_artifacts_exist_after_build()
    test_retrieval_metadata_has_citation_fields()
    print("retrieval index checks passed")
