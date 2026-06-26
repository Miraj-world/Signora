import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_predicate_backed_benchmark_is_valid():
    atoms = read_jsonl(DATASET / "data" / "processed" / "feedback_atoms.jsonl")
    questions = read_jsonl(DATASET / "data" / "evaluation" / "retrieval_questions_v2.jsonl")
    abstention = read_jsonl(DATASET / "data" / "evaluation" / "abstention_questions_v2.jsonl")
    by_id = {atom["atom_id"]: atom for atom in atoms}
    assert 50 <= len(questions) <= 100
    assert len({q["question"] for q in questions}) == len(questions)
    assert {q["split"] for q in questions} == {"validation", "test"}
    assert {q["split"] for q in abstention} == {"validation", "test"}
    for q in questions:
        assert q["review_status"] == "predicate_verified_from_corpus_v2"
        assert len(q["expected_atom_ids"]) >= 4
        assert len(q["expected_atom_ids"]) == len(set(q["expected_atom_ids"]))
        for atom_id in q["expected_atom_ids"]:
            atom = by_id[atom_id]
            assert atom["topic"] in q["relevance_rules"]["topic_any"]
            for field, allowed in q["filters"].items():
                assert atom.get(field) in allowed


if __name__ == "__main__":
    test_predicate_backed_benchmark_is_valid()
    print("benchmark v2 checks passed")
