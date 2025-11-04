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

from .intent_recognizer import DEFAULT_MIN_CONFIDENCE, IntentRecognizerUtils, StatisticsHelper, ConditionalLogger

DEFAULT_MODEL = "gpt-5-nano"
INVALID_INTENT_CONFIDENCE_PENALTY = 0.7
FALLBACK_CONFIDENCE = 0.4
ERROR_FALLBACK_CONFIDENCE = 0.3
RESPONSE_MODE_THRESHOLD = 0.65

# System Prompt Constants
INTENT_DESCRIPTIONS = """- general: Greetings, thanks, confirmations, chitchat, general business questions
- order: Placing NEW orders, pickup, menu item customization, returning customers wanting order like last time
- delivery: Order status checks, tracking, "where is my order", delivery related questions, "when will it arrive?"
- menu_inquiry: Asking about prices, menu options, recommendations, deals, "what do you have", deciding items
- hours_location: Store hours, location, address
- complaint: Problems with order/food issues, wrong items/size, sarcastic complaints, service/delivery issues"""

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
    mode: str = "classifier"


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
        ollama_base_url: str = "http://localhost:11434",
        response_generation_threshold: float = RESPONSE_MODE_THRESHOLD
    ):
        load_dotenv()

        self.use_local_llm = use_local_llm
        self.model = model
        self.min_confidence = min_confidence
        self.enable_logging = enable_logging
        self.test_mode = test_mode
        self.ollama_base_url = ollama_base_url
        self.response_generation_threshold = response_generation_threshold
        self.logger = ConditionalLogger(logging.getLogger(__name__), enable_logging)

        if not test_mode:
            if res_info:
                self.res_info = res_info
            else:
                self.res_info = self._load_res_info(res_info_file)

        self._initialize_client()

        self.stats = StatisticsHelper.init_base_stats(
            successful_queries=0,
            failed_queries=0,
            total_tokens_used=0,
            total_api_calls=0,
            llm_provider="ollama" if use_local_llm else "openai",
            response_generation_count=0,
            classification_count=0
        )

    def _load_res_info(self, res_info_file: str = None) -> Dict:
        """Load restaurant information from JSON file"""
        if res_info_file is None:
            utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
            res_info_file = os.path.join(utils_dir, 'res_info.json')

        try:
            with open(res_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f" Info file not found: {res_info_file}")
            return {}
        except json.JSONDecodeError as e:
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
            self.logger.info(f"Using OpenAI API with model: {self.model}")

    def recognize(
        self,
        query: str,
        intent_patterns: Dict,
        conversation_history: Optional[List[Dict]] = None,
        recognized_intent: Optional[str] = None,
        original_conf: Optional[float] = None
    ) -> LLMResult:
        """Recognize intent and generate response using LLM

        - If previous layer confidence >= threshold: Use LLM for response generation only
        - If previous layer confidence < threshold: Use LLM for full classification and response generation

        Args:
            query: User query
            intent_patterns: Available intent patterns
            conversation_history: Previous conversation context
            recognized_intent: Intent recognized by previous layer
            original_conf: Confidence score from previous layer
        """

        self.stats["total_queries"] += 1
        self.stats["total_api_calls"] += 1

        valid_intents = [name for name in intent_patterns.keys() if name != "unknown"]

        # Determine LLM mode based on previous layer confidence
        use_response_mode = (
            recognized_intent is not None and
            recognized_intent in valid_intents and
            original_conf is not None and
            original_conf >= self.response_generation_threshold
        )

        if use_response_mode:
            self.stats["response_generation_count"] += 1
        else:
            self.stats["classification_count"] += 1

        try:
            response = self._call_llm_api(
                query, valid_intents, conversation_history,
                recognized_intent, original_conf, use_response_mode
            )
            result = self._process_api_response(
                response, valid_intents, recognized_intent,
                original_conf, use_response_mode
            )

            if not self.test_mode:
                if not result.response or result.response.strip() == "":
                    self.logger.error(f" Empty response generated for query: '{query}'")
                    result.response = "I apologize, but I'm having trouble processing your request right now. Please try again or call us directly."

            self.stats["successful_queries"] += 1
            self.stats["intent_distribution"][result.intent] = self.stats["intent_distribution"].get(result.intent, 0) + 1
            self.stats["avg_confidence"].append(result.confidence)

            provider = "Ollama" if self.use_local_llm else "OpenAI"
            mode_str = result.mode.upper()

            if not use_response_mode:
                self.logger.info(
                    f"[{provider}] [{mode_str}] Classified as '{result.intent}' "
                    f"({result.confidence:.3f}, {result.confidence_level})"
                )

            if not self.test_mode and result.generated_response:
                preview = result.generated_response[:50] + "..." if len(
                    result.generated_response) > 50 else result.generated_response
                self.logger.info(f"Response: '{preview}'")

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f" JSON parsing error: {e}")
            self.stats["failed_queries"] += 1
            return self._get_fallback_result(f"JSON parsing failed: {str(e)}", recognized_intent, original_conf)
        except Exception as e:
            self.logger.error(f" API error: {e}")
            self.stats["failed_queries"] += 1
            return self._get_fallback_result(f"API error: {str(e)}", recognized_intent, original_conf)

    def _call_llm_api(
        self,
        query: str,
        valid_intents: List[str],
        conversation_history: Optional[List[Dict]] = None,
        recognized_intent: Optional[str] = None,
        original_conf: Optional[float] = None,
        use_response_mode: bool = False
    ):
        """Call LLM API (Ollama or OpenAI)"""
        system_prompt = self._get_system_prompt(
            valid_intents, recognized_intent, original_conf, use_response_mode
        )
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
            intent_str = f" [intent: {entry['intent']}]" if entry.get('intent') else ""
            prompt_parts.append(f"{role}: {entry['message']}{intent_str}")

        prompt_parts.extend([
            "",
            "- Do NOT repeat information already mentioned",
            "- Track the order state and what has been collected",
            "- Respond naturally based on what is still needed",
            "- Do not ask multiple questions in one response",
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

    def _get_system_prompt(
        self,
        valid_intents: List[str],
        recognized_intent: Optional[str] = None,
        original_conf: Optional[float] = None,
        use_response_mode: bool = False
    ) -> str:
        """Generate system prompt for intent classification or response generation"""

        # Test mode: Classification only, no response generation
        if self.test_mode:
            prompt = f"""Intent classification for Pizza Restaurant. Classify into: {', '.join(valid_intents)}
{INTENT_DESCRIPTIONS}

Identify Primary intent in multi-intent query.
Respond ONLY with valid JSON, no markdown formatting:
{{"intent": "intent_name", "confidence": 0.85}}"""
            return prompt

        order_flow = """Order flow:
        1. First let User picks a SPECIFIC pizza
        2. (ask size if missing)
        3. Offer sides/drinks
        4. Ask pickup or delivery
        5. Get name
        6. If delivery, ask user's address
        7. Confirm and close"""

        # Response Generation Mode: Intent already known
        if use_response_mode:
            info = self._get_selective_business_info(recognized_intent)

            prompt = f"""You are a helpful voice customer support assistant for {self.res_info.get("name", "Business")}.
Customer Query Intent Classified as: {recognized_intent} ({original_conf:.2f})

Use INFO to reply in under 40 words.
INFO: {info}

{order_flow}

Return ONLY valid JSON (no markdown):
{{"response": "Short helpful reply"}}"""
            return prompt

        # Classification Mode: Intent unknown or low confidence
        info = json.dumps(self.res_info, indent=2)

        prompt = f"""You are a helpful voice customer support assistant for {self.res_info.get("name", "Business")}, a {self.res_info.get("business_type", "Business")}.
        
        Classify INTENT into: {', '.join(valid_intents)}
        {INTENT_DESCRIPTIONS}
        
        {"Previous layer Suggested (you can correct it if seems incorrect): " + f"{recognized_intent} ({original_conf:.2f})" if (recognized_intent in valid_intents and original_conf is not None) else ""}

        Input could be a continuation of the previous query's intent. Previous Layer does not have Conversation History.

        Use INFO to generate RESPONSE in under 40 words.
        INFO: {info}

        {order_flow}

        Return ONLY valid JSON (no markdown):
        {{"intent": "intent_name", "confidence": 0.83, "response": "Short helpful reply"}}

        CONFIDENCE:
        >=0.8 high | >=0.6 medium | <0.6 low"""

        return prompt

    def _process_api_response(
        self,
        response,
        valid_intents: List[str],
        recognized_intent: Optional[str] = None,
        original_conf: Optional[float] = None,
        use_response_mode: bool = False
    ) -> LLMResult:
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

        # Response Generation Mode: Use previous layer's intent
        if use_response_mode:
            intent = recognized_intent
            confidence = original_conf
            confidence_level = IntentRecognizerUtils.determine_confidence_level(confidence)
            explanation = f"Response generated for intent '{recognized_intent}' from previous layer"
            mode = "response_generator"
        # Classification Mode: LLM provides intent
        else:
            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.5)
            explanation = result.get("explanation", "LLM classification")
            mode = "classifier"

            if intent not in valid_intents:
                intent = "unknown"
                confidence = max(FALLBACK_CONFIDENCE, confidence * INVALID_INTENT_CONFIDENCE_PENALTY)
                explanation = f"Invalid intent detected, defaulting to unknown. {explanation}"

            confidence_level = IntentRecognizerUtils.determine_confidence_level(confidence)

        # Handle response generation
        if self.test_mode:
            generated_response = ""
            response_text = ""
        else:
            generated_response = result.get("response", "")
            if not generated_response.strip():
                generated_response = "I apologize, but I'm having trouble processing your request right now."
                self.logger.warning("[LLM] Generated empty response")
            response_text = self._sanitize_response(generated_response)

        return LLMResult(
            intent=intent,
            confidence=confidence,
            confidence_level=confidence_level,
            explanation=explanation,
            generated_response=generated_response,
            response=response_text,
            matched_pattern="LLM Classification" if mode == "classifier" else f"LLM Response for {recognized_intent}",
            processing_method="llm",
            score_breakdown={"mode": mode, "response_generated": bool(generated_response)},
            error=False,
            mode=mode
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

    def _get_fallback_result(
        self,
        error_msg: str,
        recognized_intent: Optional[str] = None,
        original_conf: Optional[float] = None
    ) -> LLMResult:
        """Return fallback result when LLM API fails"""
        valid_intents = ["general", "order", "delivery", "menu_inquiry", "hours_location", "complaint"]

        if recognized_intent and recognized_intent in valid_intents and original_conf is not None:
            return LLMResult(
                intent=recognized_intent,
                confidence=original_conf,
                confidence_level=IntentRecognizerUtils.determine_confidence_level(original_conf),
                explanation=f"LLM API failed, using previous layer intent: {error_msg}",
                generated_response="",
                response="I apologize, but I'm having trouble processing your request right now. Please try again or call us directly.",
                matched_pattern="Error - Fallback to Previous Layer",
                processing_method="llm_fallback",
                score_breakdown={"error": error_msg, "fallback_to_previous": True},
                error=True,
                mode="fallback"
            )

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
            mode="fallback"
        )

    def get_statistics(self) -> dict:
        """Get LLM recognizer statistics with calculated metrics."""
        avg_conf = StatisticsHelper.calculate_average(self.stats["avg_confidence"])
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
            "response_generation_count": self.stats.get("response_generation_count", 0),
            "classification_count": self.stats.get("classification_count", 0),
        }
