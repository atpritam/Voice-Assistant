"""
LLM Intent Recognizer
Handles LLM-based intent recognition using OpenAI API
"""

import os
import json
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_MIN_CONFIDENCE = 0.5

# Confidence level thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.6

# Intent validation
INVALID_INTENT_CONFIDENCE_PENALTY = 0.7
FALLBACK_CONFIDENCE = 0.4
ERROR_FALLBACK_CONFIDENCE = 0.3

@dataclass
class LLMResult:
    """Result from LLM recognition"""
    intent: str
    confidence: float
    confidence_level: str  # 'high', 'medium', 'low'
    explanation: str
    matched_pattern: str = "LLM Classification"
    processing_method: str = "llm"
    score_breakdown: Dict = None
    error: bool = False

class LLMRecognizer:
    """LLM-based intent recognition using OpenAI API"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        enable_logging: bool = False,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        """
        Initialize LLM Recognizer
        """
        load_dotenv()

        # API configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.min_confidence = min_confidence
        self.client = OpenAI(api_key=self.api_key)

        # Logging setup
        self.enable_logging = enable_logging
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            self.logger = logging.getLogger(__name__)

        # Statistics tracking
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "intent_distribution": {},
            "avg_confidence": [],
            "total_tokens_used": 0,
            "total_api_calls": 0,
        }

    def recognize(self, query: str, intent_patterns: Dict) -> LLMResult:
        """
        Recognize intent using LLM
        """
        self.stats["total_queries"] += 1
        self.stats["total_api_calls"] += 1

        try:
            response = self._call_openai_api(query, intent_patterns)
            result = self._parse_api_response(response)
            validated_result = self._validate_intent(result, intent_patterns)
            llm_result = self._create_llm_result(validated_result)

            # Update statistics
            self.stats["successful_queries"] += 1
            self.stats["intent_distribution"][llm_result.intent] = (
                self.stats["intent_distribution"].get(llm_result.intent, 0) + 1
            )
            self.stats["avg_confidence"].append(llm_result.confidence)

            # Logging
            if self.enable_logging:
                self.logger.info(
                    f"[LLM] Intent: {llm_result.intent} "
                    f"(confidence: {llm_result.confidence:.3f}, level: {llm_result.confidence_level})"
                )
                self.logger.debug(f"[LLM] Explanation: {llm_result.explanation}")

            return llm_result

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

    def _call_openai_api(self, query: str, intent_patterns: Dict):
        """
        Call OpenAI API for intent classification
        """
        system_prompt = self._get_system_prompt(intent_patterns)
        user_prompt = f'Classify this user query: "{query}"'

        return self.client.chat.completions.create(
            model=self.model,
            messages=[ # type: ignore
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"}, # type: ignore
            verbosity="low"
        )

    def _parse_api_response(self, response) -> Dict:
        """
        Parse API response and extract result
        """
        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        if hasattr(response, "usage"):
            self.stats["total_tokens_used"] += response.usage.total_tokens

        return result

    @staticmethod
    def _validate_intent(result: Dict, intent_patterns: Dict) -> Dict:
        """
        Validate intent and adjust if needed
        """
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.5)
        explanation = result.get("explanation", "LLM classification")

        valid_intents = [name for name in intent_patterns.keys() if name != "unknown"]

        if intent not in valid_intents:
            intent = "unknown"
            confidence = max(FALLBACK_CONFIDENCE, confidence * INVALID_INTENT_CONFIDENCE_PENALTY)
            explanation = f"Invalid intent detected, defaulting to unknown. {explanation}"

        return {
            "intent": intent,
            "confidence": confidence,
            "explanation": explanation
        }

    @staticmethod
    def _create_llm_result(validated_result: Dict) -> LLMResult:
        """
        Create LLMResult object from validated data
        """
        intent = validated_result["intent"]
        confidence = validated_result["confidence"]
        explanation = validated_result["explanation"]
        confidence_level = "low"

        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "high"
        elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            confidence_level = "medium"

        return LLMResult(
            intent=intent,
            confidence=confidence,
            confidence_level=confidence_level,
            explanation=explanation,
            matched_pattern="LLM Classification",
            processing_method="llm",
            score_breakdown={"llm_explanation": explanation},
            error=False,
        )

    @staticmethod
    def _get_system_prompt(intent_patterns: Dict) -> str:
        """
        Generate system prompt for intent classification
        """
        valid_intents = [name for name in intent_patterns.keys() if name != "unknown"]

        system_prompt = f"""You are an intent classification assistant for a pizza restaurant chatbot.

Your task:
1. Classify the user's query into ONE of these intents: {', '.join(valid_intents)}
2. If the query doesn't match any intent well, classify it as "unknown"
3. Provide a confidence score (0.0 to 1.0)
4. Provide a brief explanation

Intent descriptions:
- order: User wants to place an order, buy pizza, or start ordering process
- complaint: User has a problem, complaint, or issue with their order
- hours_location: User asks about business hours, location, or address
- menu_inquiry: User asks about menu items, toppings, prices, or options
- delivery: User asks about delivery status, tracking, fees, or timing
- general: Greetings, thanks, confirmations (hello, thanks, ok, sure, etc.)

Respond ONLY with valid JSON in this exact format:
{{
    "intent": "intent_name",
    "confidence": 0.85,
    "explanation": "Brief explanation of why this intent was chosen"
}}
"""
        return system_prompt

    @staticmethod
    def _get_fallback_result(error_msg: str) -> LLMResult:
        """
        Return fallback result when LLM API fails
        """
        return LLMResult(
            intent="unknown",
            confidence=ERROR_FALLBACK_CONFIDENCE,
            confidence_level="low",
            explanation=f"LLM API failed: {error_msg}",
            matched_pattern="Error",
            processing_method="llm_fallback",
            score_breakdown={"error": error_msg},
            error=True,
        )

    def get_statistics(self) -> dict:
        """Get LLM recognizer statistics with calculated metrics."""
        avg_conf = (
            sum(self.stats["avg_confidence"]) / len(self.stats["avg_confidence"])
            if self.stats.get("avg_confidence")
            else 0.0
        )

        success_rate = (
            self.stats["successful_queries"] / self.stats["total_queries"]
            if self.stats.get("total_queries", 0) > 0
            else 0.0
        )

        return {
            "total_queries_processed": self.stats.get("total_queries", 0),
            "successful_queries": self.stats.get("successful_queries", 0),
            "failed_queries": self.stats.get("failed_queries", 0),
            "success_rate": success_rate,
            "intent_distribution": self.stats.get("intent_distribution", {}),
            "average_confidence": avg_conf,
            "total_tokens_used": self.stats.get("total_tokens_used", 0),
            "total_api_calls": self.stats.get("total_api_calls", 0),
        }