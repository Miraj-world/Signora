"""HTTP contract checks for the Signora API."""
from pathlib import Path
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import signora_api


client = TestClient(signora_api.app)


def test_health_is_independent_of_model_readiness():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyst_app_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Ask customer evidence" in response.text


def test_answer_request_validation_rejects_empty_query():
    response = client.post("/v1/answers", json={"query": ""})
    assert response.status_code == 422


def test_answer_response_is_compact_by_default():
    fake_answer = {
            "status": "answered",
            "answer": "Supported answer. [atom_1]",
            "uncertainty": "Limited sample.",
            "recommendations": [],
            "citations": [{"atom_id": "atom_1"}],
            "generation_model": "test-model",
            "retrieval": {"should_abstain": False},
        }
    with patch.object(signora_api, "readiness", return_value={"status": "ready"}), \
            patch.object(signora_api, "answer_question", return_value=fake_answer):
        response = client.post("/v1/answers", json={"query": "What happened?"})
    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert "evidence" not in response.json()


def test_not_ready_returns_503():
    state = {"status": "not_ready", "missing_files": ["index_manifest.json"]}
    with patch.object(signora_api, "readiness", return_value=state):
        response = client.get("/ready")
    assert response.status_code == 503


if __name__ == "__main__":
    test_health_is_independent_of_model_readiness()
    test_analyst_app_is_served()
    test_answer_request_validation_rejects_empty_query()
    test_answer_response_is_compact_by_default()
    test_not_ready_returns_503()
    print("signora api checks passed")
