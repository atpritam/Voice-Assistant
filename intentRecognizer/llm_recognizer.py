"""
LLM Intent Recognizer with Response Generation
Supports both OpenAI API and Local Ollama models
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from .intent_recognizer import DEFAULT_MIN_CONFIDENCE, IntentRecognizerUtils

DEFAULT_MODEL = "gpt-5-nano"
INVALID_INTENT_CONFIDENCE_PENALTY = 0.7
FALLBACK_CONFIDENCE = 0.4
ERROR_FALLBACK_CONFIDENCE = 0.3

@dataclass
class LLMResult:
    """Result from LLM recognition and response generation"""
    intent: str
    confidence: float
    confidence_level: str
    explanation: str
    response: str
    matched_pattern: str = "LLM Classification"
    processing_method: str = "llm"
    score_breakdown: Dict = None
    error: bool = False


class LLMRecognizer:
    """LLM-based intent recognition and response generation - supports OpenAI and Ollama"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        enable_logging: bool = False,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        res_info: Optional[Dict] = None,
        res_info_file: str = None,
        test_mode: bool = False,
        use_local_llm: bool = False,
        ollama_base_url: str = "http://localhost:11434"
    ):
        load_dotenv()

        self.use_local_llm = use_local_llm
        self.model = model
        self.min_confidence = min_confidence
        self.enable_logging = enable_logging
        self.test_mode = test_mode
        self.ollama_base_url = ollama_base_url

        if not test_mode:
            if res_info:
                self.res_info = res_info
            else:
                self.res_info = self._load_res_info(res_info_file)

        if enable_logging:
            self.logger = logging.getLogger(__name__)

        self._initialize_client()

        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "intent_distribution": {},
            "avg_confidence": [],
            "total_tokens_used": 0,
            "total_api_calls": 0,
            "llm_provider": "ollama" if use_local_llm else "openai"
        }

    def _load_res_info(self, res_info_file: str = None) -> Dict:
        """Load restaurant information from JSON file"""
        if res_info_file is None:
            utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
            res_info_file = os.path.join(utils_dir, 'res_info.json')

        try:
            with open(res_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            if self.enable_logging:
                self.logger.error(f"Info file not found: {res_info_file}")
            return {}
        except json.JSONDecodeError as e:
            if self.enable_logging:
                self.logger.error(f"Invalid JSON in info file: {e}")
            return {}

    def _initialize_client(self):
        """Initialize appropriate LLM client"""
        if self.use_local_llm:
            try:
                import requests
                self.requests = requests
                response = requests.get(f"{self.ollama_base_url}/api/tags")
                if response.status_code != 200:
                    raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_base_url}")
                if self.enable_logging:
                    self.logger.info(f"Connected to local Ollama at {self.ollama_base_url}")
                    self.logger.info(f"Using model: {self.model}")
            except ImportError:
                raise ImportError("requests library required for Ollama. Install: pip install requests")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Ollama: {e}")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            if self.enable_logging:
                self.logger.info(f"Using OpenAI API with model: {self.model}")

    def recognize(self, query: str, intent_patterns: Dict,
                  conversation_history: Optional[List[Dict]] = None,
                  recognized_intent: Optional[str] = None) -> LLMResult:
        """Recognize intent and generate response using LLM"""
        self.stats["total_queries"] += 1
        self.stats["total_api_calls"] += 1

        try:
            response = self._call_llm_api(query, intent_patterns, conversation_history, recognized_intent)
            result = self._process_api_response(response, intent_patterns, recognized_intent)

            if not result.response or result.response.strip() == "":
                if self.enable_logging:
                    self.logger.error(f"[LLM] Empty response generated for query: '{query}'")
                result.response = "I apologize, but I'm having trouble processing your request right now. Please try again or call us directly."

            self.stats["successful_queries"] += 1
            self.stats["intent_distribution"][result.intent] = self.stats["intent_distribution"].get(result.intent, 0) + 1
            self.stats["avg_confidence"].append(result.confidence)

            if self.enable_logging:
                provider = "OLLAMA" if self.use_local_llm else "LLM"
                self.logger.info(f"[{provider}] Intent: {result.intent} (confidence: {result.confidence:.3f}, level: {result.confidence_level})")
                if result.response:
                    preview = result.response[:80] + "..." if len(result.response) > 80 else result.response
                    self.logger.debug(f"[{provider}] Response: {preview}")

            return result

        except json.JSONDecodeError as e:
            if self.enable_logging:
                self.logger.error(f"[LLM] JSON parsing error: {e}")
            self.stats["failed_queries"] += 1
            return self._get_fallback_result(f"JSON parsing failed: {str(e)}")
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"[LLM] API error: {e}")
            self.stats["failed_queries"] += 1
            return self._get_fallback_result(f"API error: {str(e)}")

    def _call_llm_api(self, query: str, intent_patterns: Dict,
                     conversation_history: Optional[List[Dict]] = None,
                     recognized_intent: Optional[str] = None):
        """Call LLM API (Ollama or OpenAI)"""
        system_prompt = self._get_system_prompt(intent_patterns, recognized_intent)
        user_prompt = self._build_user_prompt(query, conversation_history)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if self.use_local_llm:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json"
            }
            response = self.requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        else:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                response_format={"type": "json_object"},  # type: ignore
                verbosity="low"
            )

    def _build_user_prompt(self, query: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """Build user prompt with conversation context"""
        if not conversation_history:
            return f'CURRENT CUSTOMER QUERY: "{query}"'

        prompt_parts = ["CONVERSATION HISTORY (recent messages):"]
        recent_history = conversation_history[-12:] if len(conversation_history) > 12 else conversation_history

        for entry in recent_history:
            role = "Customer" if entry['type'] == 'user' else "Assistant"
            prompt_parts.append(f"{role}: {entry['message']}")

        prompt_parts.extend([
            "",
            "CONTEXT AWARENESS:",
            "- Understand what the customer is referring to",
            "- Do NOT repeat information already mentioned",
            "- Track the order state and what has been collected",
            "- Respond naturally based on what is still needed",
            "",
            f'CURRENT CUSTOMER QUERY: "{query}"'
        ])

        return "\n".join(prompt_parts)

    def _get_system_prompt(self, intent_patterns: Dict, recognized_intent: Optional[str] = None) -> str:
        """Generate system prompt for intent classification and response generation"""
        valid_intents = [name for name in intent_patterns.keys() if name != "unknown"]
        intent_descriptions = """Intent descriptions:
- order: User wants to place an order
- complaint: User has a problem, or issue with their order, wants refund or escalate
- hours_location: User asks about business hours, location
- menu_inquiry: User asks about menu items, toppings, prices, or options
- delivery: User asks about delivery status, tracking, fees, or timing
- general: Greetings, thanks, confirmations"""

        if self.test_mode:
            prompt = f"Intent classification for pizza restaurant. Classify into: {', '.join(valid_intents)}\n{intent_descriptions}"
            if recognized_intent:
                prompt += f"\nPrevious layer suggested: {recognized_intent}, override if incorrect."
            prompt += '\n\nRespond ONLY with valid JSON, no markdown formatting:\n{{"intent": "intent_name", "confidence": 0.85}}'
            return prompt

        tts_rules = """ 1. Keep responses SHORT (1-3 sentences max under 50 words preferably)
2. NO special formatting: no $, %, parentheses, brackets, colons, semicolons
3. Spell out prices (e.g., 'twelve dollars' instead of $12)
4. No multiple questions in one response"""

        prompt = f"""You are a helpful voice customer support assistant for {self.res_info.get("name", "Business")}.
Generate ONE natural, conversational response that directly addresses the query.
INFO: {json.dumps(self.res_info, indent=2)}
Order flow:
1) Customer chooses something
2) You offer sides or drinks
3) You ask if they'd like to pick up or get the order delivered
4) get their name
5) ask address if its a delivery
6) confirm and close.
AVAILABLE INTENTS:
{intent_descriptions}

{tts_rules}"""

        prompt += '\n\nRespond with ONLY valid JSON (no markdown code blocks):\n{{"intent": "intent_name", "confidence": 0.85, "response": "Your natural, helpful response here"}}'
        return prompt

    def _process_api_response(self, response, intent_patterns: Dict, recognized_intent: Optional[str] = None) -> LLMResult:
        """Parse and validate API response, creating LLMResult"""
        if self.use_local_llm:
            result_text = response.get("message", {}).get("content", "{}")
            if "eval_count" in response:
                self.stats["total_tokens_used"] += response.get("eval_count", 0)
        else:
            result_text = response.choices[0].message.content
            if hasattr(response, "usage"):
                self.stats["total_tokens_used"] += response.usage.total_tokens

        result_text = re.sub(r'^```(?:json)?\s*|\s*```$', '', result_text.strip(), flags=re.MULTILINE).strip()
        result = json.loads(result_text)

        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.5)
        explanation = result.get("explanation", "LLM classification")
        generated_response = result.get("response", "")

        if not self.test_mode and not generated_response.strip():
            generated_response = None
            if self.enable_logging:
                self.logger.warning("[LLM] Generated empty response")

        valid_intents = [name for name in intent_patterns.keys() if name != "unknown"]
        if intent not in valid_intents:
            if recognized_intent and recognized_intent in valid_intents:
                intent = recognized_intent
                explanation = f"Using intent from previous layer: {recognized_intent}. {explanation}"
            else:
                intent = "unknown"
                confidence = max(FALLBACK_CONFIDENCE, confidence * INVALID_INTENT_CONFIDENCE_PENALTY)
                explanation = f"Invalid intent detected. {explanation}"

        confidence_level = IntentRecognizerUtils.determine_confidence_level(confidence)

        return LLMResult(
            intent=intent,
            confidence=confidence,
            confidence_level=confidence_level,
            explanation=explanation,
            response=generated_response,
            matched_pattern="LLM Classification",
            processing_method="llm",
            score_breakdown={"response_generated": True},
            error=False,
        )

    @staticmethod
    def _get_fallback_result(error_msg: str) -> LLMResult:
        """Return fallback result when LLM API fails"""
        return LLMResult(
            intent="unknown",
            confidence=ERROR_FALLBACK_CONFIDENCE,
            confidence_level="low",
            explanation=f"LLM API failed: {error_msg}",
            response="I apologize, but I'm having trouble processing your request right now. Please try again or call us directly.",
            matched_pattern="Error",
            processing_method="llm_fallback",
            score_breakdown={"error": error_msg},
            error=True,
        )

    def get_statistics(self) -> dict:
        """Get LLM recognizer statistics with calculated metrics."""
        avg_conf = sum(self.stats["avg_confidence"]) / len(self.stats["avg_confidence"]) if self.stats.get("avg_confidence") else 0.0
        success_rate = self.stats["successful_queries"] / self.stats["total_queries"] if self.stats.get("total_queries", 0) > 0 else 0.0

        return {
            "llm_provider": self.stats.get("llm_provider", "unknown"),
            "total_queries_processed": self.stats.get("total_queries", 0),
            "successful_queries": self.stats.get("successful_queries", 0),
            "failed_queries": self.stats.get("failed_queries", 0),
            "success_rate": success_rate,
            "intent_distribution": self.stats.get("intent_distribution", {}),
            "average_confidence": avg_conf,
            "total_tokens_used": self.stats.get("total_tokens_used", 0),
            "total_api_calls": self.stats.get("total_api_calls", 0),
        }