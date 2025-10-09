"""
LLM Intent Recognizer with Response Generation
Supports both OpenAI API and Local Ollama models
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from .intent_recognizer import DEFAULT_MIN_CONFIDENCE, IntentRecognizerUtils
from utils.res_info import RESTAURANT_INFO

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
        restaurant_info: Optional[Dict] = None,
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
        self.restaurant_info = restaurant_info or RESTAURANT_INFO
        self.ollama_base_url = ollama_base_url

        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            self.logger = logging.getLogger(__name__)

        # Initialize appropriate client
        if self.use_local_llm:
            try:
                import requests
                self.requests = requests
                # Test Ollama connection
                response = requests.get(f"{ollama_base_url}/api/tags")
                if response.status_code != 200:
                    raise ConnectionError(f"Cannot connect to Ollama at {ollama_base_url}")
                if enable_logging:
                    self.logger.info(f"Connected to local Ollama at {ollama_base_url}")
                    self.logger.info(f"Using model: {model}")
            except ImportError:
                raise ImportError("requests library required for Ollama. Install: pip install requests")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Ollama: {e}")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                )
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            if enable_logging:
                self.logger.info(f"Using OpenAI API with model: {model}")

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

    def recognize(self, query: str, intent_patterns: Dict,
                  conversation_history: Optional[List[Dict]] = None,
                  recognized_intent: Optional[str] = None) -> LLMResult:
        """
        Recognize intent and generate response using LLM

        Args:
            query: User's current query
            intent_patterns: Available intent patterns for classification
            conversation_history: Previous conversation messages
            recognized_intent: Intent recognized by previous layers (if any)
        """
        self.stats["total_queries"] += 1
        self.stats["total_api_calls"] += 1

        try:
            if self.use_local_llm:
                response = self._call_ollama_api(
                    query, intent_patterns, conversation_history, recognized_intent
                )
            else:
                response = self._call_openai_api(
                    query, intent_patterns, conversation_history, recognized_intent
                )

            result = self._process_api_response(response, intent_patterns, recognized_intent)

            # Validate response content
            if not result.response or result.response.strip() == "":
                if self.enable_logging:
                    self.logger.error(f"[LLM] Empty response generated for query: '{query}'")
                result.response = "I apologize, but I'm having trouble processing your request right now. Please try again or call us directly."

            self.stats["successful_queries"] += 1
            self.stats["intent_distribution"][result.intent] = \
                self.stats["intent_distribution"].get(result.intent, 0) + 1
            self.stats["avg_confidence"].append(result.confidence)

            if self.enable_logging:
                provider = "OLLAMA" if self.use_local_llm else "LLM"
                self.logger.info(
                    f"[{provider}] Intent: {result.intent} "
                    f"(confidence: {result.confidence:.3f}, level: {result.confidence_level})"
                )
                if result.response and len(result.response) > 0:
                    preview = result.response[:80] + "..." if len(result.response) > 80 else result.response
                    self.logger.debug(f"[{provider}] Response: {preview}")
                else:
                    self.logger.warning(f"[{provider}] No response generated")

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

    def _get_intent_description(self) -> str:
        """Get description of intent from intent patterns"""
        return (
            """Intent descriptions:
            - order: User wants to place an order, buy pizza, or start ordering process
            - complaint: User has a problem, complaint, or issue with their order, wants refund or escalate
            - hours_location: User asks about business hours, location, or address
            - menu_inquiry: User asks about menu items, toppings, prices, or options
            - delivery: User asks about delivery status, tracking, fees, or timing
            - general: Greetings, thanks, confirmations (hello, thanks, ok, sure, etc.)"""
        )

    def _call_ollama_api(self, query: str, intent_patterns: Dict,
                        conversation_history: Optional[List[Dict]] = None,
                        recognized_intent: Optional[str] = None):
        """Call local Ollama API for intent classification and response generation"""
        system_prompt = self._get_system_prompt(intent_patterns, recognized_intent)
        user_prompt = self._build_user_prompt(query, conversation_history)

        # Build messages for Ollama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json"
        }

        response = self.requests.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def _call_openai_api(self, query: str, intent_patterns: Dict,
                        conversation_history: Optional[List[Dict]] = None,
                        recognized_intent: Optional[str] = None):
        """Call OpenAI API for intent classification and response generation"""
        system_prompt = self._get_system_prompt(intent_patterns, recognized_intent)
        user_prompt = self._build_user_prompt(query, conversation_history)

        return self.client.chat.completions.create(
            model=self.model,
            messages=[ # type: ignore
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"}, # type: ignore
            verbosity="low"
        )

    def _build_user_prompt(self, query: str,
                          conversation_history: Optional[List[Dict]] = None) -> str:
        """Build user prompt with conversation context"""
        prompt_parts = []

        if conversation_history and len(conversation_history) > 0:
            prompt_parts.append("CONVERSATION HISTORY (recent messages):")
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history

            for entry in recent_history:
                role = "Customer" if entry['type'] == 'user' else "Assistant"
                prompt_parts.append(f"{role}: {entry['message']}")

            prompt_parts.append("")
            prompt_parts.append("CONTEXT AWARENESS:")
            prompt_parts.append("- Understand what the customer is referring to")
            prompt_parts.append("- Do NOT repeat information already mentioned")
            prompt_parts.append("- Track the order state and what has been collected")
            prompt_parts.append("- Respond naturally based on what is still needed")
            prompt_parts.append("")

        prompt_parts.append(f'CURRENT CUSTOMER QUERY: "{query}"')

        return "\n".join(prompt_parts)

    def _process_api_response(self, response, intent_patterns: Dict,
                              recognized_intent: Optional[str] = None) -> LLMResult:
        """Parse and validate API response, creating LLMResult"""
        # Extract response text based on provider
        if self.use_local_llm:
            result_text = response.get("message", {}).get("content", "{}")
            # Track tokens if available
            if "eval_count" in response:
                self.stats["total_tokens_used"] += response.get("eval_count", 0)
        else:
            result_text = response.choices[0].message.content
            if hasattr(response, "usage"):
                self.stats["total_tokens_used"] += response.usage.total_tokens

        # Clean up result_text
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

        result = json.loads(result_text)

        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.5)
        explanation = result.get("explanation", "LLM classification")
        generated_response = result.get("response", "")

        if not self.test_mode and (not generated_response or generated_response.strip() == ""):
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
            score_breakdown={
                "llm_explanation": explanation,
                "response_generated": True
            },
            error=False,
        )

    def _get_system_prompt(self, intent_patterns: Dict,
                          recognized_intent: Optional[str] = None) -> str:
        """Generate system prompt for intent classification and response generation"""
        valid_intents = [name for name in intent_patterns.keys() if name != "unknown"]

        # Test Mode - Only intent recognition, no response generation
        if self.test_mode:
            base_prompt = f"""Intent classification for pizza restaurant. Classify into: {', '.join(valid_intents)} 
                              {self._get_intent_description()}"""

            if recognized_intent:
                base_prompt += f"Previous layer suggested: {recognized_intent}, override if incorrect."

            base_prompt += f"""Respond ONLY with valid JSON, no markdown formatting:
                                {{
                                    "intent": "intent_name",
                                    "confidence": 0.85,
                                }}"""
            return base_prompt

        # Intent recognition and response generation
        base_prompt = f"""You are a helpful voice customer support assistant for {self.restaurant_info['name']}, a pizza restaurant.
                          Generate ONE natural, conversational response that directly addresses the customer's query.
                          This response will be used for TTS so make the response adhere to common TTS text rules.
                          RESTAURANT INFO: {json.dumps(self.restaurant_info, indent=2)}
                          Offers Pick up and Delivery.
                          AVAILABLE INTENTS: {', '.join(valid_intents)}
                          {self._get_intent_description()}

                        Some of CRITICAL RESPONSE RULES for TTS:
                        1. Keep responses SHORT (1-3 sentences max under 40 words preferably)
                        2. NO special formatting: no $, %, parentheses, brackets, colons, semicolons
                        3. Spell out prices (e.g., 'twelve dollars' instead of $12)
                        4. No multiple questions in one response
                        5. Do not repeat information already mentioned unless user asks to do so"""

        if recognized_intent:
            base_prompt += f"""Previous intent classifier layer suggested: {recognized_intent} intent"
                            The previous layer does not have full conversational context, if you are highly confident that the intent is incorrect, provide the correct intent with a confidence 0 to 1 based on context."""


        base_prompt += f""" Respond with ONLY valid JSON (no markdown code blocks):
                            {{
                                "intent": "intent_name",
                                "confidence": 0.85,
                                "response": "Your natural, helpful response here"
                            }}"""
        return base_prompt

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
        avg_conf = (
            sum(self.stats["avg_confidence"]) / len(self.stats["avg_confidence"])
            if self.stats.get("avg_confidence") else 0.0
        )

        success_rate = (
            self.stats["successful_queries"] / self.stats["total_queries"]
            if self.stats.get("total_queries", 0) > 0 else 0.0
        )

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