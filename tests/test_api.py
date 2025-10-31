# Testing the FastAPI endpoints using TestClient
from unittest.mock import patch

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_resolve_ticket_mocked():
    with patch("api.main.embedder") as mock_embedder, \
         patch("api.main.llm_client") as mock_llm:

        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_llm.generate_response.return_value = {
            "answer": "Mocked answer",
            "references": ["FAQ: Mocked"],
            "action_required": "none",
            "reasoning_trace": None
        }

        with TestClient(app) as client:
            response = client.post("/resolve-ticket", json={"ticket_text": "How do I transfer my domain?"})
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Mocked answer"

def test_resolve_ticket_short_text():
# Testing ticket resolution with too short input
    response = client.post(
        "/resolve-ticket",
        json={"ticket_text": "Help"}
    )
    assert response.status_code == 422  # Validation error

def test_resolve_ticket_invalid():
# Testing ticket resolution with invalid input (empty string)
    response = client.post(
        "/resolve-ticket",
        json={"ticket_text": ""}
    )
    assert response.status_code == 422  # Validation error

def test_resolve_ticket_missing_field():
# Testing ticket resolution with missing ticket_text field
    response = client.post(
        "/resolve-ticket",
        json={}
    )
    assert response.status_code == 422  # Validation error