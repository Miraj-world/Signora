"""Checks for conservative parent-thread readiness decisions."""
from pathlib import Path
import sys


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_thread_context import audit


def test_missing_parent_ids_defers_expansion():
    result = audit([
        {"thread_id": "thread-1", "parent_id": None, "is_root_post": True, "source_type": "reddit_discussion"},
        {"thread_id": "thread-1", "parent_id": None, "is_root_post": False, "source_type": "reddit_discussion"},
    ])
    assert result["status"] == "defer"
    assert result["counts"]["items_with_parent_id"] == 0


def test_stable_parent_id_marks_corpus_ready_for_implementation():
    result = audit([
        {"thread_id": "thread-1", "parent_id": None, "is_root_post": True, "source_type": "reddit_discussion"},
        {"thread_id": "thread-1", "parent_id": "root-1", "is_root_post": False, "source_type": "reddit_discussion"},
    ])
    assert result["status"] == "ready"


if __name__ == "__main__":
    test_missing_parent_ids_defers_expansion()
    test_stable_parent_id_marks_corpus_ready_for_implementation()
    print("thread context audit checks passed")
