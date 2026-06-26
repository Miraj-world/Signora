import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_dataset_manifest_counts_match_files():
    manifest = json.loads((DATASET_ROOT / "data/manifests/dataset_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["record_counts"]
    with (DATASET_ROOT / "data/synthetic/structured/accounts.csv").open("r", encoding="utf-8", newline="") as f:
        assert sum(1 for _ in csv.DictReader(f)) == counts["accounts"]
    with (DATASET_ROOT / "data/synthetic/structured/customers.csv").open("r", encoding="utf-8", newline="") as f:
        assert sum(1 for _ in csv.DictReader(f)) == counts["customers"]
    with (DATASET_ROOT / "data/synthetic/structured/opportunities.csv").open("r", encoding="utf-8", newline="") as f:
        assert sum(1 for _ in csv.DictReader(f)) == counts["opportunities"]
    assert len(read_jsonl(DATASET_ROOT / "data/synthetic/raw/support_tickets.jsonl")) == counts["support_tickets"]
    assert len(read_jsonl(DATASET_ROOT / "data/synthetic/raw/sales_call_excerpts.jsonl")) == counts["sales_call_excerpts"]
    assert len(read_jsonl(DATASET_ROOT / "data/synthetic/raw/product_reviews.jsonl")) == counts["product_reviews"]
    assert len(read_jsonl(DATASET_ROOT / "data/synthetic/raw/churn_feedback.jsonl")) == counts["churn_feedback"]
    assert len(read_jsonl(DATASET_ROOT / "data/synthetic/raw/feature_requests.jsonl")) == counts["feature_requests"]
    assert len(read_jsonl(DATASET_ROOT / "data/synthetic/raw/customer_interviews.jsonl")) == counts["customer_interviews"]
    assert len(read_jsonl(DATASET_ROOT / "data/synthetic/raw/github_issues.jsonl")) == counts["github_issues"]
    assert len(read_jsonl(DATASET_ROOT / "data/synthetic/structured/product_usage_summaries.jsonl")) == counts["product_usage_summaries"]
    assert len(read_jsonl(DATASET_ROOT / "data/public/normalized/feedback_items.jsonl")) == counts["public_feedback_accepted"]
    assert len(read_jsonl(DATASET_ROOT / "data/public/quarantined/feedback_items.jsonl")) == counts["public_feedback_quarantined"]


def test_final_reference_scale_is_met():
    manifest = json.loads((DATASET_ROOT / "data/manifests/dataset_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["record_counts"]
    minimums = {
        "accounts": 500,
        "opportunities": 2000,
        "support_tickets": 3000,
        "sales_call_excerpts": 1000,
        "product_reviews": 1000,
        "churn_feedback": 300,
        "feature_requests": 500,
        "customer_interviews": 200,
        "github_issues": 500,
    }
    for key, minimum in minimums.items():
        assert counts[key] >= minimum


def test_evaluation_answer_keys_are_excluded_from_normal_index():
    include_manifest = json.loads((DATASET_ROOT / "data/manifests/vector_index_include_manifest.json").read_text(encoding="utf-8"))
    excluded = set(include_manifest["exclude"])
    assert "data/evaluation/**" in excluded
    assert "data/synthetic/reference/hidden_patterns.json" in excluded
    assert all(not path.startswith("data/evaluation") for path in include_manifest["include"])


def test_data_quality_has_no_critical_errors():
    metrics = json.loads((DATASET_ROOT / "reports/data_quality_metrics.json").read_text(encoding="utf-8"))
    assert metrics["critical_error_count"] == 0


if __name__ == "__main__":
    test_dataset_manifest_counts_match_files()
    test_evaluation_answer_keys_are_excluded_from_normal_index()
    test_data_quality_has_no_critical_errors()
    print("dataset integrity checks passed")
