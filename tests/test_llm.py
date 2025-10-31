# Unit testing for LLM integration and confidence calculation

import sys
from pathlib import Path
import json
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from unittest.mock import patch, MagicMock
from utils.confidence import calculate_confidence, should_escalate
from llm.ollama_client import TucowsSupportLLM

# Testing confidence calculation
def test_calculate_confidence_basic():
    similarity_scores = [0.8, 0.5, 0.3]
    llm_response = {"answer": "This is a test answer.", "references": ["FAQ 1", "FAQ 2"]}
    num_faqs_retrieved = 2

    confidence = calculate_confidence(similarity_scores, llm_response, num_faqs_retrieved)
    expected = 0.609
    assert confidence == expected

# Testing confidence calculation (edge case)
def test_calculate_confidence_edge_cases():
    similarity_scores = []
    llm_response = {"answer": "", "references": []}
    num_faqs_retrieved = 0

    confidence = calculate_confidence(similarity_scores, llm_response, num_faqs_retrieved)
    assert confidence == 0.0

# Testing escalation logic
def test_should_escalate_logic():
    # Low confidence → escalate to human review
    assert should_escalate(0.5, "none", threshold=0.6) == "needs_human_review"
    # High confidence → keep original action
    assert should_escalate(0.8, "none", threshold=0.6) == "none"
    # Non-none action → keep original action even if confidence is low
    assert should_escalate(0.3, "contact_provider", threshold=0.6) == "contact_provider"


# Testing TucowsSupportLLM with Ollama (successful response)
@patch("llm.ollama_client.ollama.Client")
def test_llm_successful_response(mock_ollama_client):
    # Mock successful Ollama response
    mock_client = MagicMock()
    mock_response = {
        "message": {
            "content": json.dumps({
                "answer": "To update nameservers, contact your domain provider.",
                "references": ["FAQ: Change my DNS nameservers"],
                "action_required": "contact_provider",
                "reasoning_trace": "Customer needs DNS update assistance"
            })
        }
    }
    mock_client.chat.return_value = mock_response
    mock_ollama_client.return_value = mock_client

    llm = TucowsSupportLLM(host="http://localhost:11434", model="llama3.2")
    response = llm.generate_response("How do I update my DNS?", [
        {"faq": {"question": "Change my DNS nameservers", "answer": "Contact your provider"}}
    ])

    assert response["action_required"] == "contact_provider"
    assert len(response["references"]) > 0
    assert "nameservers" in response["answer"].lower()


# Testing TucowsSupportLLM error handling (connection failure)
@patch("llm.ollama_client.ollama.Client")
def test_llm_connection_error(mock_ollama_client):
    # Simulate Ollama connection error
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("Connection refused")
    mock_ollama_client.return_value = mock_client

    llm = TucowsSupportLLM(host="http://localhost:11434", model="llama3.2")
    response = llm.generate_response("My domain is down", [])

    assert response["action_required"] == "needs_human_review"
    assert "LLM error" in response["reasoning_trace"]
    assert response["answer"].startswith("I'm experiencing technical difficulties")


# Testing TucowsSupportLLM invalid JSON response
@patch("llm.ollama_client.ollama.Client")
def test_llm_invalid_json_response(mock_ollama_client):
    # Mock Ollama returning invalid JSON
    mock_client = MagicMock()
    mock_response = {
        "message": {
            "content": "This is not valid JSON"
        }
    }
    mock_client.chat.return_value = mock_response
    mock_ollama_client.return_value = mock_client

    llm = TucowsSupportLLM(host="http://localhost:11434", model="llama3.2")
    response = llm.generate_response("Test query", [])

    assert response["action_required"] == "needs_human_review"
    assert "Invalid JSON" in response["reasoning_trace"] or "JSON decode error" in response["reasoning_trace"]


# Testing TucowsSupportLLM missing required fields
@patch("llm.ollama_client.ollama.Client")
def test_llm_missing_required_fields(mock_ollama_client):
    # Mock Ollama returning incomplete response
    mock_client = MagicMock()
    mock_response = {
        "message": {
            "content": json.dumps({
                "answer": "Partial response"
                # Missing: references, action_required
            })
        }
    }
    mock_client.chat.return_value = mock_response
    mock_ollama_client.return_value = mock_client

    llm = TucowsSupportLLM(host="http://localhost:11434", model="llama3.2")
    response = llm.generate_response("Test query", [])

    assert response["action_required"] == "needs_human_review"
    assert "Missing required keys" in response["reasoning_trace"]
