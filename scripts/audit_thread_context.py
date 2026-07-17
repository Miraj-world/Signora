"""Audit whether the corpus can support exact parent-thread expansion."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ITEMS_PATH = ROOT / "dataset" / "data" / "processed" / "feedback_items.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(items: list[dict]) -> dict:
    threaded = [item for item in items if item.get("thread_id")]
    thread_counts = Counter(item["thread_id"] for item in threaded)
    parent_count = sum(bool(item.get("parent_id")) for item in items)
    exact_parent_ready = parent_count > 0
    return {
        "status": "ready" if exact_parent_ready else "defer",
        "decision": (
            "Exact parent expansion is supported."
            if exact_parent_ready
            else "Do not enable parent expansion until parent_id is collected and a threaded-source evaluation set exists."
        ),
        "counts": {
            "total_items": len(items),
            "threaded_items": len(threaded),
            "unique_threads": len(thread_counts),
            "multi_item_threads": sum(count > 1 for count in thread_counts.values()),
            "root_items": sum(item.get("is_root_post") is True for item in threaded),
            "comment_items": sum(item.get("is_root_post") is False for item in threaded),
            "items_with_parent_id": parent_count,
        },
        "threaded_source_types": dict(Counter(item.get("source_type") for item in threaded)),
        "requirements_before_enablement": [
            "Collect stable parent_id values during threaded-source ingestion.",
            "Resolve each parent_id to a feedback item without guessing from thread order.",
            "Add held-out threaded questions where parent context changes answer quality.",
            "Compare citation entailment and unsupported-claim control with context off versus on.",
        ],
    }


def main() -> None:
    print(json.dumps(audit(read_jsonl(ITEMS_PATH)), indent=2))


if __name__ == "__main__":
    main()
