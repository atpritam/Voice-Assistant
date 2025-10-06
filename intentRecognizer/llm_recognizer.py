"""
LLM Intent Recognizer
Handles LLM-based intent recognition using OpenAI API
Includes integrated LLM service functionality
"""

import os
import json
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv


@dataclass
class LLMResult:
    """Result from LLM recognition"""

    intent: str
    confidence: float
    confidence_level: str  # 'high', 'medium', 'low'
    explanation: str
    processing_method: str = "llm"
    error: bool = False


class LLMRecognizer:
    """LLM-based intent recognition using OpenAI API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
        enable_logging: bool = False,
        min_confidence: float = 0.5,
    ):
        """
        Initialize LLM Recognizer

        Args:
            api_key: OpenAI API key (if None, loads from environment)
            model: OpenAI model to use (default: gpt-4o-mini)
            enable_logging: Enable detailed logging
            min_confidence: Minimum confidence threshold
        """

        load_dotenv()

        # API configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
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

        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "intent_distribution": {},
            "avg_confidence": [],
            "total_tokens_used": 0,
            "total_api_calls": 0,
        }

    def _get_system_prompt(self, intent_patterns: Dict) -> str:
        """
        Generate system prompt for intent classification

        Args:
            intent_patterns: Dictionary of available intents and their patterns

        Returns:
            System prompt string
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

    def recognize(self, query: str, intent_patterns: Dict) -> LLMResult:
        """
        Recognize intent using LLM

        Args:
            query: User query string
            intent_patterns: Dictionary of available intents

        Returns:
            LLMResult object with classification
        """
        # Update statistics
        self.stats["total_queries"] += 1
        self.stats["total_api_calls"] += 1

        try:
            # Create system and user prompts
            system_prompt = self._get_system_prompt(intent_patterns)
            user_prompt = f'Classify this user query: "{query}"'

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[  # type: ignore
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},  # type: ignore
                temperature=0.3,
            )

            # Parse response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            # Track token usage
            if hasattr(response, "usage"):
                self.stats["total_tokens_used"] += response.usage.total_tokens

            # Validate and extract results
            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.5)
            explanation = result.get("explanation", "LLM classification")

            valid_intents = [
                name for name in intent_patterns.keys() if name != "unknown"
            ]

            # Ensure intent is valid
            if intent not in valid_intents:
                intent = "unknown"
                confidence = max(0.4, confidence * 0.7)
                explanation = (
                    f"Invalid intent detected, defaulting to unknown. {explanation}"
                )

            # Determine confidence level
            if confidence >= 0.8:
                confidence_level = "high"
            elif confidence >= 0.6:
                confidence_level = "medium"
            else:
                confidence_level = "low"

            # Update statistics
            self.stats["successful_queries"] += 1
            self.stats["intent_distribution"][intent] = (
                self.stats["intent_distribution"].get(intent, 0) + 1
            )
            self.stats["avg_confidence"].append(confidence)

            llm_result = LLMResult(
                intent=intent,
                confidence=confidence,
                confidence_level=confidence_level,
                explanation=explanation,
                processing_method="llm",
                error=False,
            )

            # Logging
            if self.enable_logging:
                self.logger.info(
                    f"[LLM] Query: '{query}' → Intent: {intent} "
                    f"(confidence: {confidence:.3f}, level: {confidence_level})"
                )
                self.logger.debug(f"[LLM] Explanation: {explanation}")

            return llm_result

        except json.JSONDecodeError as e:
            if self.enable_logging:
                self.logger.error(f"[LLM] JSON parsing error: {e}")
            self.stats["failed_queries"] += 1
            return self._get_fallback_result(query, f"JSON parsing failed: {str(e)}")

        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"[LLM] API error: {e}")
            self.stats["failed_queries"] += 1
            return self._get_fallback_result(query, f"API error: {str(e)}")

    def _get_fallback_result(self, query: str, error_msg: str) -> LLMResult:
        """
        Return fallback result when LLM API fails

        Args:
            query: Original query
            error_msg: Error message

        Returns:
            Fallback LLMResult
        """
        return LLMResult(
            intent="unknown",
            confidence=0.3,
            confidence_level="low",
            explanation=f"LLM API failed: {error_msg}",
            processing_method="llm_fallback",
            error=True,
        )

    def get_statistics(self) -> Dict:
        """Get LLM recognizer statistics"""
        avg_conf = (
            sum(self.stats["avg_confidence"]) / len(self.stats["avg_confidence"])
            if self.stats["avg_confidence"]
            else 0.0
        )

        success_rate = (
            self.stats["successful_queries"] / self.stats["total_queries"]
            if self.stats["total_queries"] > 0
            else 0.0
        )

        return {
            "total_queries_processed": self.stats["total_queries"],
            "successful_queries": self.stats["successful_queries"],
            "failed_queries": self.stats["failed_queries"],
            "success_rate": success_rate,
            "intent_distribution": self.stats["intent_distribution"],
            "average_confidence": avg_conf,
            "total_tokens_used": self.stats["total_tokens_used"],
            "total_api_calls": self.stats["total_api_calls"],
        }
