"""Ollama LLM integration with structured output."""
import json
from typing import Dict, List
import ollama
from config import OLLAMA_HOST, OLLAMA_MODEL
from llm.prompt_templates import MCP_SYSTEM_PROMPT, build_user_prompt


class TucowsSupportLLM:
    """Ollama-powered LLM with structured output."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        # Use default timeout (no custom timeout) to match CLI behavior
        self.client = ollama.Client(host=host)
        self.model = model
        print(f"✓ Ollama client initialized (model: {model}, host: {host})")

    def generate_response(self, ticket_text: str, retrieved_faqs: List[Dict]) -> Dict:
        user_prompt = build_user_prompt(ticket_text, retrieved_faqs)

        try:
            print(f"→ Sending request to Ollama...")
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": MCP_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",
                options={
                    "temperature": 0.7,
                    "num_predict": 512,
                }
            )
            print(f"✓ Received response from Ollama")

            content = response["message"]["content"]
            result = json.loads(content)

            required_keys = ["answer", "references", "action_required"]
            if not isinstance(result, dict) or not all(k in result for k in required_keys):
                return self._fallback_response("Missing required keys in LLM response")

            result.setdefault("reasoning_trace", None)
            return result

        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error: {e}")
            return self._fallback_response(f"Invalid JSON from LLM: {str(e)}")
        except Exception as e:
            print(f"✗ Ollama error: {e}")
            return self._fallback_response(str(e))

    def _fallback_response(self, error_msg: str) -> Dict:
        return {
            "answer": "I'm experiencing technical difficulties. A support agent will assist you shortly.",
            "references": [],
            "action_required": "needs_human_review",
            "reasoning_trace": f"LLM error: {error_msg}"
        }
