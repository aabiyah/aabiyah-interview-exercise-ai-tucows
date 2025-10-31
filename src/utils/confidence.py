# Confidence scoring for LLM responses.
from typing import Dict, List


def calculate_confidence(
        similarity_scores: List[float],
        llm_response: Dict,
        num_faqs_retrieved: int
) -> float:
    # Calculating confidence score based on multiple factors like similarity, references, length, and retrievals.

    # Top similarity score (40%)
    top_similarity = max(similarity_scores) if similarity_scores else 0.0
    similarity_component = top_similarity * 0.4

    # Reference count (30%)
    num_references = len(llm_response.get("references", []))
    reference_score = min(num_references / 3.0, 1.0)  # Normalize to max 3
    reference_component = reference_score * 0.3

    # Answer length (20%)
    answer_length = len(llm_response.get("answer", ""))
    length_score = min(answer_length / 200.0, 1.0)  # Normalize to 200 chars
    length_component = length_score * 0.2

    # Retrieval count (10%)
    retrieval_score = min(num_faqs_retrieved / 3.0, 1.0)
    retrieval_component = retrieval_score * 0.1

    # Total confidence (heuristic formula)
    confidence = (
            similarity_component +
            reference_component +
            length_component +
            retrieval_component
    )

    return round(confidence, 3)


def should_escalate(confidence: float, action_required: str, threshold: float = 0.6) -> str:
    # Determining if escalation to human review is needed based on confidence (override to human review if confidence is low
    if confidence < threshold and action_required == "none":
        return "needs_human_review"

    return action_required
