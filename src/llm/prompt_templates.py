# File: src/llm/prompt_templates.py
from typing import List, Dict

MCP_SYSTEM_PROMPT = """You are an AI agent created to help support teams at Tucows Domains to customers' domain-related queries.

**ROLE**: Analyze customer support tickets and provide actionable, accurate responses based on Tucows Domains' documentation.

**CONTEXT**: You have access to Tucows Domains FAQ documentation covering:
- GDPR and data privacy policies
- Domain management
- Renewals and redemption periods
- Domain transfers and ownership changes
- Top questions from customers

**TASK**:
1. Analyze the customer support ticket
2. Use the provided FAQ context to formulate an answer
3. Generate a clear, professional response
4. Cite specific FAQ sources in references
5. Determine if escalation or human review is needed

**OUTPUT SCHEMA** (strict JSON format):
{
  "answer": "Clear, actionable response to the customer query",
  "references": ["FAQ: Question title", "Policy: Section X.Y"],
  "action_required": "none|escalate_to_abuse_team|needs_human_review|contact_provider",
  "reasoning_trace": "Internal explanation of your decision-making process"
}

**RULES**:
- Be concise and professional (2-3 sentences max for simple queries)
- Always cite sources in the "references" array
- If context is insufficient or unclear → "action_required": "needs_human_review"
- For policy violations or abuse → "action_required": "escalate_to_abuse_team"
- For provider-specific issues → "action_required": "contact_provider"
- Include step-by-step instructions when applicable
- Do NOT make up information not in the provided context
"""


def build_user_prompt(ticket_text: str, retrieved_faqs: List[Dict]) -> str:
    # Building the user prompt with ticket and FAQ context. This is used alongside the MCP system prompt.
    if not isinstance(retrieved_faqs, list):
        retrieved_faqs = []

    # Ensuring defensive handling of FAQ entries
    context_sections: List[str] = []

    for i, result in enumerate(retrieved_faqs, 1):
        # Defensive handling for unexpected shapes
        if not isinstance(result, dict):
            faq = {}
            score = 0.0
        else:
            # Many vector stores store the FAQ under different keys
            faq = {}
            if isinstance(result.get("faq"), dict):
                faq = result.get("faq")
            elif isinstance(result.get("metadata"), dict):
                faq = result.get("metadata")
            elif isinstance(result.get("data"), dict):
                faq = result.get("data")
            else:
                # Last resort: try keys directly on result
                faq = {k: result.get(k) for k in ("question", "answer", "related_links") if k in result}

            try:
                score = float(result.get("similarity_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0

        question = faq.get("question", "N/A")
        answer = faq.get("answer", "")

        context_sections.append(
            f"### FAQ {i} (Relevance: {score:.2f})\n"
            f"**Question**: {question}\n"
            f"**Answer**: {answer}\n"
        )

    context = "\n".join(context_sections)

    prompt = (
        f"**CUSTOMER TICKET**:\n{ticket_text}\n\n---\n\n"
        f"**RETRIEVED FAQ CONTEXT** (Top {len(retrieved_faqs)} most relevant):\n{context}\n\n---\n\n"
        "**INSTRUCTIONS**:\n"
        "Based on the above ticket and FAQ context, generate a JSON response following the output schema.\n"
        "Ensure your answer is helpful, accurate, and cites relevant FAQs in the references array.\n"
    )

    return prompt
