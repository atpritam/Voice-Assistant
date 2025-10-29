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
    generated_response: str
    response: str
    matched_pattern: str = "LLM Classification"
    processing_method: str = "llm"
    score_breakdown: Dict = None
    error: bool = False
    override: bool = False


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
                self.logger.error(f" Info file not found: {res_info_file}")
            return {}
        except json.JSONDecodeError as e:
            if self.enable_logging:
                self.logger.error(f" Invalid JSON in info file: {e}")
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
                  recognized_intent: Optional[str] = None,
                  original_conf: Optional[float] = None,
                  override_thres: Optional[float] = None) -> LLMResult:
        """Recognize intent and generate response using LLM

        Args:
            recognized_intent: Intent Recognized by previous layer
            original_conf: Original Confidence Score of recognized_intent by previous layer
            override_thres: Previous Layer Intent Override Threshold for LLM
        """

        self.stats["total_queries"] += 1
        self.stats["total_api_calls"] += 1

        try:
            response = self._call_llm_api(query, intent_patterns, conversation_history, recognized_intent, original_conf)
            result = self._process_api_response(response, intent_patterns, recognized_intent)

            if not self.test_mode:
                if not result.response or result.response.strip() == "":
                    if self.enable_logging:
                        self.logger.error(f" Empty response generated for query: '{query}'")
                    result.response = "I apologize, but I'm having trouble processing your request right now. Please try again or call us directly."

            self.stats["successful_queries"] += 1
            self.stats["intent_distribution"][result.intent] = self.stats["intent_distribution"].get(result.intent, 0) + 1
            self.stats["avg_confidence"].append(result.confidence)

            if self.enable_logging:
                provider = "Ollama" if self.use_local_llm else "OpenAI"
                if original_conf is None:
                    self.logger.info(
                        f"[{provider}] {result.intent} ({result.confidence:.3f}, {result.confidence_level})")
                elif not self.test_mode and original_conf is not None:
                    if (result.intent != recognized_intent and
                        result.confidence >= override_thres and
                        result.confidence > original_conf and
                        result.intent != "unknown"
                        ):

                        result.override = True
                        self.logger.info(
                            f"[{provider}] OVERRIDE: '{recognized_intent}' "
                            f"({original_conf:.3f}) -> '{result.intent}' ({result.confidence:.3f}, {result.confidence_level})"
                        )

                if not self.test_mode and result.generated_response:
                    preview = result.generated_response[:50] + "..." if len(
                        result.generated_response) > 50 else result.generated_response
                    self.logger.info(f"Response: '{preview}'")

            return result

        except json.JSONDecodeError as e:
            if self.enable_logging:
                self.logger.error(f" JSON parsing error: {e}")
            self.stats["failed_queries"] += 1
            return self._get_fallback_result(f"JSON parsing failed: {str(e)}")
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f" API error: {e}")
            self.stats["failed_queries"] += 1
            return self._get_fallback_result(f"API error: {str(e)}")

    def _call_llm_api(self, query: str, intent_patterns: Dict,
                     conversation_history: Optional[List[Dict]] = None,
                     recognized_intent: Optional[str] = None,
                      original_conf: Optional[float] = None):
        """Call LLM API (Ollama or OpenAI)"""
        system_prompt = self._get_system_prompt(intent_patterns, recognized_intent, original_conf)
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

        prompt_parts = ["CONVERSATION HISTORY:"]
        recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history

        for entry in recent_history:
            role = "Customer" if entry['type'] == 'user' else "Assistant"
            prompt_parts.append(f"{role}: {entry['message']}")

        prompt_parts.extend([
            "",
            "- Do NOT repeat information already mentioned",
            "- Track the order state and what has been collected",
            "- Respond naturally based on what is still needed",
            "- Do not ask multiple questions in one response"
            "- Current Query could be a continuation of previous query intent"
            "",
            f'CURRENT QUERY: "{query}"'
        ])

        return "\n".join(prompt_parts)

    def _get_selective_business_info(self, intent: str) -> str:
        """Return only business info relevant to the intent to reduce token usage"""

        if intent == "order" or intent == "delivery":
            info = {
                "menu_highlights": self.res_info.get("menu_highlights"),
                "delivery": self.res_info.get("delivery"),
            }

        elif intent == "complaint":
            info = {
                "location": self.res_info.get("location"),
                "delivery": self.res_info.get("delivery")
            }

        elif intent == "hours_location":
            info = {
                "hours": self.res_info.get("hours"),
                "location": self.res_info.get("location")
            }

        elif intent == "menu_inquiry":
            info = {
                "menu_highlights": self.res_info.get("menu_highlights")
            }

        elif intent == "general":
            info = {
                "business_type": self.res_info.get("business_type"),
            }

        else:
            info = {}

        return json.dumps(info, indent=2)

    def _get_system_prompt(self, intent_patterns: Dict, recognized_intent: Optional[str] = None, original_conf: Optional[float] = None) -> str:
        """Generate system prompt for intent classification and response generation"""
        valid_intents = [name for name in intent_patterns.keys() if name != "unknown"]
        intent_descriptions = """- general: Greetings, thanks, confirmations, chitchat
        - order: Placing NEW orders, pickup, menu item customization, returning customers wanting order like last time
        - delivery: Order status checks, tracking, "where is my order", delivery related questions, "when will it arrive?"
        - menu_inquiry: Asking about prices, options, recommendations, deals, "what do you have", deciding items
        - hours_location: Store hours, location, address
        - complaint: Problems with order/food issues, wrong items/size, sarcastic complaints, service/delivery issues"""

        if self.test_mode:
            prompt = f"""Intent classification for Pizza Restaurant. Classify into: {', '.join(valid_intents)}\n{intent_descriptions}"""
            prompt += '\nIdentify Primary intent in multi-intent query.'
            if recognized_intent in valid_intents:
                prompt += f"\nPrevious layer suggested: {recognized_intent}, override if you think it's incorrect."
            prompt += '\n\nRespond ONLY with valid JSON, no markdown formatting:\n{{"intent": "intent_name", "confidence": 0.85}}'
            return prompt

        if recognized_intent in valid_intents and original_conf > 0.7:
            info = self._get_selective_business_info(recognized_intent)
        else:
            info = json.dumps(self.res_info, indent=2)

        prompt = f"""You are a helpful voice customer support assistant for {self.res_info.get("name", "Business")}.
                 Use INFO to reply in under 40 words.
                 INFO: {info}
                 
    Order flow:
    1. User picks a specific pizza (ask size if missing)
    2. Offer sides/drinks
    3. Ask pickup or delivery
    4. Get name
    5. If delivery, ask user's address
    6. Confirm and close

    INTENTS: {', '.join(valid_intents)}
    {intent_descriptions}

    Confidence:
    >=0.8 high | >=0.6 medium | <0.6 low
    """

        if recognized_intent in valid_intents and original_conf > 0.5:
            prompt += f"""
    Previous layer suggested: {recognized_intent} ({original_conf:.2f})
    You may override intent if wrong. Previous layer had no chat history.
    """

        prompt += (
            "\nInput could be a continuation of the previous query’s intent.\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"intent": "intent_name", "confidence": 0.83, "response": "Short helpful reply"}'
        )

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

        if self.test_mode:
            generated_response = ""
            response = ""
        else:
            generated_response = result.get("response", "")
            if not generated_response.strip():
                generated_response = "I apologize, but I'm having trouble processing your request right now."
                if self.enable_logging:
                    self.logger.warning("[LLM] Generated empty response")
            response = self._sanitize_response(generated_response)

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
            generated_response=generated_response,
            response=response,
            matched_pattern="LLM Classification",
            processing_method="llm",
            score_breakdown={"response_generated": True},
            error=False,
            override=False
        )

    def _sanitize_response(self, text: str) -> str:
        """Remove TTS-unfriendly characters from response."""

        if not text:
            return text

        def format_currency(match):
            dollars = int(match.group(1))
            cents = int(match.group(2))
            dollar_word = 'dollar' if dollars == 1 else 'dollars'
            cent_word = 'cent' if cents == 1 else 'cents'
            return f"{dollars} {dollar_word}" if cents == 0 else f"{dollars} {dollar_word} and {cents} {cent_word}"

        text = re.sub(r'\$(\d+)\.(\d{2})', format_currency, text)
        text = re.sub(r'(\d+)\.(\d{2})(?=\s|,|and|for|$)', format_currency, text)
        text = re.sub(r'\$(\d+)', lambda m: f"{m.group(1)} {'dollar' if int(m.group(1)) == 1 else 'dollars'}", text)
        text = re.sub(r'\b(dollars?)\.(?=\d)', r'\1', text)
        text = (text.replace('%', ' percent')
                .replace(':', ',')
                .replace(';', ','))

        text = re.sub(r'[\[\]()]', '', text)

        return text.strip()

    @staticmethod
    def _get_fallback_result(error_msg: str) -> LLMResult:
        """Return fallback result when LLM API fails"""
        return LLMResult(
            intent="unknown",
            confidence=ERROR_FALLBACK_CONFIDENCE,
            confidence_level="low",
            explanation=f"LLM API failed: {error_msg}",
            generated_response="",
            response="I apologize, but I'm having trouble processing your request right now. Please try again or call us directly.",
            matched_pattern="Error",
            processing_method="llm_fallback",
            score_breakdown={"error": error_msg},
            error=True,
            override=False
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