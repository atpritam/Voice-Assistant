"""
LLM Intent Recognizer with Response Generation
Supports both Local and Cloud Ollama models
"""

import os
import sys
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.logger import ConditionalLogger
from utils.statistics import StatisticsHelper

from ..intent_recognizer import DEFAULT_MIN_CONFIDENCE, IntentRecognizerUtils
from . import templates

DEFAULT_MODEL = "llama3.2:3b-instruct-q4_K_M"
RESPONSE_MODE_THRESHOLD = 0.65


def sanitize_response(text: str) -> str:
    """Remove TTS-unfriendly characters from response.

    Converts currency symbols, percentages, and punctuation to speech-friendly formats.

    Args:
        text: Raw response text from LLM

    Returns:
        Sanitized text suitable for text-to-speech
    """
    def format_currency(match):
        dollars = int(match.group(1))
        cents = int(match.group(2))
        dollar_word = 'dollar' if dollars == 1 else 'dollars'
        cent_word = 'cent' if cents == 1 else 'cents'
        return f"{dollars} {dollar_word}" if cents == 0 else f"{dollars} {dollar_word} and {cents} {cent_word}"

    # Convert currency formats
    text = re.sub(r'\$(\d+)\.(\d{2})', format_currency, text)
    text = re.sub(r'(\d+)\.(\d{2})(?=\s|,|and|for|$)', format_currency, text)
    text = re.sub(r'\$(\d+)', lambda m: f"{m.group(1)} {'dollar' if int(m.group(1)) == 1 else 'dollars'}", text)
    text = re.sub(r'\b(dollars?)\.(?=\d)', r'\1', text)

    # Convert symbols to words
    text = (text.replace('%', ' percent')
            .replace(':', ',')
            .replace(';', ','))

    # Remove brackets and parentheses
    text = re.sub(r'[\[\]()]', '', text)

    return text.strip()


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
    """LLM-based intent recognition and response generation - supports local and cloud Ollama models"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        enable_logging: bool = False,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        res_info: Optional[Dict] = None,
        res_info_file: str = None,
        test_mode: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        response_generation_threshold: float = RESPONSE_MODE_THRESHOLD
    ):
        load_dotenv()

        self.model = model
        self.min_confidence = min_confidence
        self.enable_logging = enable_logging
        self.test_mode = test_mode
        self.ollama_base_url = ollama_base_url
        self.response_generation_threshold = response_generation_threshold
        self.logger = ConditionalLogger(__name__, enable_logging)

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
            llm_provider="ollama",
            response_generation_count=0,
            classification_count=0
        )

    def _load_res_info(self, res_info_file: str = None) -> Dict:
        """Load restaurant information from JSON file"""
        if res_info_file is None:
            utils_dir = os.path.join(os.path.dirname(__file__), '../..', 'utils')
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
        try:
            from ollama import Client
            self.client = Client(host=self.ollama_base_url)
            models_response = self.client.list()
            available_models = [model.model for model in models_response.models]

            if not available_models:
                raise ConnectionError(f"No models found in Ollama at {self.ollama_base_url}")

            if self.model not in available_models:
                self.logger.warning(f"Model '{self.model}' not found in Ollama.")
                self.logger.warning(f"Available models: {available_models}")
                raise ValueError(f"Model '{self.model}' not available.")

            is_cloud_model = self.model.endswith('-cloud')
            model_type = "Cloud Model" if is_cloud_model else f"local Ollama model at {self.ollama_base_url}"
            self.logger.info(f"Connected to {model_type}.")
            self.logger.info(f"Using model: {self.model}")

        except ImportError:
            raise ImportError("ollama library required for Ollama. Install: pip install ollama")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

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

            provider = "Ollama"
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
        """Call LLM API"""
        system_prompt = self._get_system_prompt(
            valid_intents, recognized_intent, original_conf, use_response_mode
        )
        user_prompt = templates.build_user_prompt(query, conversation_history)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat(
            model=self.model,
            messages=messages,
            format="json",
        )
        return response

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
            return templates.get_test_mode_prompt(valid_intents)

        # Response Generation Mode: Intent already known
        if use_response_mode:
            selective_info = templates.get_selective_business_info(self.res_info, recognized_intent)
            return templates.get_response_generation_prompt(
                self.res_info.get("name", "Business"),
                recognized_intent,
                original_conf,
                selective_info
            )

        # Classification Mode: Intent unknown or low confidence
        business_info = json.dumps(self.res_info, indent=2)
        return templates.get_classification_prompt(
            self.res_info.get("name", "Business"),
            self.res_info.get("business_type", "Business"),
            valid_intents,
            business_info,
            recognized_intent,
            original_conf
        )

    def _process_api_response(
        self,
        response,
        valid_intents: List[str],
        recognized_intent: Optional[str] = None,
        original_conf: Optional[float] = None,
        use_response_mode: bool = False
    ) -> LLMResult:
        """Parse and validate API response, creating LLMResult"""
        result_text = response.get("message", {}).get("content", "{}")
        if "eval_count" in response:
            self.stats["total_tokens_used"] += response.get("eval_count", 0)
        if "prompt_eval_count" in response:
            self.stats["total_tokens_used"] += response.get("prompt_eval_count", 0)

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
                confidence = confidence * 0.7
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
            response_text = sanitize_response(generated_response)

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
            confidence=0.0,
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
        return StatisticsHelper.build_stats_response(
            self.stats,
            average_confidence=StatisticsHelper.calculate_average(self.stats["avg_confidence"]),
            success_rate=StatisticsHelper.calculate_success_rate(
                self.stats.get("successful_queries", 0),
                self.stats.get("total_queries", 0)
            )
        )