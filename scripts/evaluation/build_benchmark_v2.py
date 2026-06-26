"""Build a reproducible, predicate-backed retrieval benchmark from the corpus.

The v1 benchmark assigned gold atoms by row number.  This builder only selects
atoms whose stored metadata satisfies the topic, segment, and severity asserted
by each question.  It is deliberately deterministic so an index can be rebuilt
and evaluated against the same question set.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
EVAL_DIR = DATASET / "data" / "evaluation"
ATOM_PATH = DATASET / "data" / "processed" / "feedback_atoms.jsonl"
VERSION = "retrieval_questions_v2_predicate_backed"

# Topics with enough coverage for the three explicit relevance predicates below.
TOPICS = [
    ("delayed_notifications", "delayed notifications"),
    ("duplicate_notifications", "duplicate notifications"),
    ("time_zone_handling", "time-zone handling"),
    ("scheduled_reports", "scheduled reports"),
    ("offline_task_viewing", "offline task viewing"),
    ("large_dataset_performance", "large-dataset performance"),
    ("csv_export", "CSV exports"),
    ("archived_tasks", "archived tasks"),
    ("response_time", "support response time"),
    ("exact_match_dependency", "exact-match search behavior"),
    ("delayed_synchronization", "delayed synchronization"),
    ("workarounds", "workarounds"),
    ("account_manager_help", "account-manager help"),
    ("offline_editing", "offline editing"),
    ("filtering", "filtering"),
    ("slack", "the Slack integration"),
    ("salesforce", "the Salesforce integration"),
    ("jira", "the Jira integration"),
    ("google_drive", "the Google Drive integration"),
    ("plan_limits", "plan limits"),
]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def stable_sample(rows, question_id, count=8):
    """Select canonical, diverse citations without treating other matching atoms as wrong."""
    ranked = sorted(rows, key=lambda a: hashlib.sha256(f"{question_id}:{a['atom_id']}".encode()).hexdigest())
    selected, seen_feedback = [], set()
    for atom in ranked:
        if atom["feedback_id"] in seen_feedback:
            continue
        selected.append(atom)
        seen_feedback.add(atom["feedback_id"])
        if len(selected) == count:
            break
    return selected


def make_question(question_id, topic, label, kind, atoms):
    if kind.startswith("segment:"):
        segment = kind.split(":", 1)[1]
        display = segment.replace("_", "-")
        filters = {"customer_segment": [segment]}
        question = f"What problems do {display} customers report with {label}?"
        rules = {"topic_any": [topic], "customer_segment_any": [segment]}
    else:
        filters = {"severity": ["high"]}
        question = f"Which high-severity customer issues involve {label}?"
        rules = {"topic_any": [topic], "severity_any": ["high"]}

    matching = [a for a in atoms if a.get("topic") in rules["topic_any"]]
    for field, values in rules.items():
        if field.endswith("_any") and field != "topic_any":
            matching = [a for a in matching if a.get(field[:-4]) in values]
    canonical = stable_sample(matching, question_id)
    if len(canonical) < 4:
        raise ValueError(f"{question_id} has only {len(canonical)} distinct feedback records")
    return {
        "question_id": question_id,
        "question": question,
        "question_type": "filtered_topic_retrieval",
        "split": "validation" if int(question_id.rsplit("_", 1)[1]) % 4 == 0 else "test",
        "filters": filters,
        "relevance_rules": rules,
        "expected_atom_ids": [a["atom_id"] for a in canonical],
        "expected_feedback_ids": [a["feedback_id"] for a in canonical],
        "expected_answer_summary": f"Return evidence about {label} that satisfies the stated filter.",
        "should_abstain": False,
        "difficulty": "metadata_filtered",
        "review_status": "predicate_verified_from_corpus_v2",
        "ground_truth_method": "metadata_predicate_plus_canonical_citations",
        "matching_atom_count": len(matching),
    }


def main():
    atoms = read_jsonl(ATOM_PATH)
    questions = []
    for i, (topic, label) in enumerate(TOPICS, 1):
        segment_counts = Counter(a.get("customer_segment") for a in atoms if a.get("topic") == topic)
        # Use the two best-supported segments for this topic. This keeps every
        # question answerable instead of asserting a segment that has no evidence.
        segments = [segment for segment, _ in segment_counts.most_common(2)]
        for kind in [*(f"segment:{segment}" for segment in segments), "high_severity"]:
            question_id = f"v2_q_{len(questions) + 1:03d}"
            questions.append(make_question(question_id, topic, label, kind, atoms))

    # These terms are intentionally absent from the indexed topic taxonomy and are
    # used only to calibrate each model's no-answer threshold on its validation split.
    absent_topics = [
        "cryptocurrency payments", "virtual-reality workspaces", "drone integrations",
        "blockchain identity", "quantum encryption", "facial-recognition attendance",
        "metaverse offices", "autonomous delivery routing", "NFT billing", "satellite connectivity",
        "biometric payroll", "carbon-credit trading", "holographic meetings", "brain-computer interfaces",
        "self-driving fleet management", "genealogy imports", "weather-derivative reporting", "robotics safety",
        "cryptographic key escrow", "space-station collaboration",
    ]
    abstention = [
        {
            "question_id": f"v2_abs_{i:03d}",
            "question": f"What feedback do customers provide about {term}?",
            "split": "validation" if i % 4 == 0 else "test",
            "should_abstain": True,
            "forbidden_topics": [term],
            "review_status": "taxonomy_absence_verified_v2",
            "ground_truth_method": "absent_topic_taxonomy",
        }
        for i, term in enumerate(absent_topics, 1)
    ]

    # A final audit turns a silently bad benchmark into a build failure.
    atom_by_id = {a["atom_id"]: a for a in atoms}
    for q in questions:
        if len(set(q["expected_atom_ids"])) != len(q["expected_atom_ids"]):
            raise ValueError(f"duplicate gold atom in {q['question_id']}")
        for atom_id in q["expected_atom_ids"]:
            atom = atom_by_id[atom_id]
            rules = q["relevance_rules"]
            assert atom["topic"] in rules["topic_any"]
            for field, values in q["filters"].items():
                assert atom.get(field) in values, (q["question_id"], atom_id, field)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(EVAL_DIR / "retrieval_questions_v2.jsonl", questions)
    write_jsonl(EVAL_DIR / "abstention_questions_v2.jsonl", abstention)
    (EVAL_DIR / "benchmark_v2_manifest.json").write_text(json.dumps({
        "version": VERSION,
        "question_count": len(questions),
        "abstention_count": len(abstention),
        "corpus_atom_count": len(atoms),
        "question_types": dict(Counter(q["question_type"] for q in questions)),
        "ground_truth": "metadata predicates verified against the corpus; canonical IDs are diverse citation examples",
        "deprecated": ["retrieval_questions.jsonl", "abstention_questions.jsonl"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(questions)} predicate-backed retrieval questions and {len(abstention)} abstention cases.")


if __name__ == "__main__":
    main()
