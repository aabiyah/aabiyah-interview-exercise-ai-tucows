# Ollama LLM integration with structured output.
import json
from typing import Dict, List
import ollama
from config import OLLAMA_HOST, OLLAMA_MODEL
from llm.prompt_templates import MCP_SYSTEM_PROMPT, build_user_prompt


class TucowsSupportLLM:

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        print(f"[INIT] Initializing Ollama client...")
        print(f"[INIT] Host: {host}")
        print(f"[INIT] Model: {model}")
        try:
            self.client = ollama.Client(host=host)
            self.model = model
            print("[INIT] Ollama client initialized successfully!")
        except Exception as e:
            print(f"[INIT] Failed to initialize Ollama client: {e}")
            raise e

    def generate_response(self, ticket_text: str, retrieved_faqs: List[Dict]) -> Dict:
        print(f"\n[LLM] Generating response for ticket: {ticket_text[:60]}...")
        print(f"[LLM] Retrieved {len(retrieved_faqs)} FAQs")

        user_prompt = build_user_prompt(ticket_text, retrieved_faqs)
        print(f"[LLM] Prompt built successfully (length: {len(user_prompt)} chars)")

        try:
            print("[LLM] Sending request to Ollama model...")
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": MCP_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",
                options={
                    "temperature": 0.3,
                    "num_predict": 800
                }
            )
            print("[LLM] Received response from Ollama")

            # Log response for debugging
            print(f"[LLM] Raw response: {response}")

            content = response["message"]["content"]
            print(f"[LLM] Content received (length: {len(content)} chars)")

            result = json.loads(content)
            print("[LLM] Parsed JSON successfully")

            required_keys = ["answer", "references", "action_required"]
            if not isinstance(result, dict) or not all(k in result for k in required_keys):
                print(f"[LLM] Missing keys in response: {result}")
                raise ValueError(f"Missing required keys in LLM response: {result}")

            result.setdefault("reasoning_trace", None)
            print(f"[LLM] Final structured response ready")
            return result

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error: {e}")
            return self._fallback_response("Invalid JSON from LLM")
        except Exception as e:
            print(f"[ERROR] Ollama error: {e}")
            return self._fallback_response(str(e))

    def _fallback_response(self, error_msg: str) -> Dict:
        print(f"[FALLBACK] Returning fallback response due to: {error_msg}")
        return {
            "answer": "I'm experiencing technical difficulties. A support agent will assist you shortly.",
            "references": [],
            "action_required": "needs_human_review",
            "reasoning_trace": f"LLM error: {error_msg}"
        }
