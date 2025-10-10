"""
LLM Intent Recognizer with Streaming Response Generation
Supports both OpenAI API and Local Ollama models with streaming
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Callable
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
    """LLM-based intent recognition and response generation - supports OpenAI and Ollama with streaming"""

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
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
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
                  recognized_intent: Optional[str] = None,
                  stream_callback: Optional[Callable] = None) -> LLMResult:
        """Recognize intent and generate response using LLM with streaming support"""
        self.stats["total_queries"] += 1
        self.stats["total_api_calls"] += 1

        try:
            if stream_callback:
                response, full_response = self._call_llm_api_streaming(
                    query, intent_patterns, conversation_history,
                    recognized_intent, stream_callback
                )
            else:
                response = self._call_llm_api(query, intent_patterns,
                                             conversation_history, recognized_intent)
                full_response = None

            result = self._process_api_response(response, intent_patterns,
                                               recognized_intent, full_response)

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

    def _call_llm_api_streaming(self, query: str, intent_patterns: Dict,
                               conversation_history: Optional[List[Dict]] = None,
                               recognized_intent: Optional[str] = None,
                               stream_callback: Optional[Callable] = None):
        """Call LLM API with streaming (Ollama or OpenAI)"""
        system_prompt = self._get_system_prompt(intent_patterns, recognized_intent)
        user_prompt = self._build_user_prompt(query, conversation_history)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        accumulated_text = ""
        json_started = False
        in_response_field = False
        response_buffer = ""

        if self.enable_logging:
            self.logger.info("[LLM] Starting streaming response...")
            if stream_callback:
                self.logger.info("[LLM] Stream callback is set")
            else:
                self.logger.warning("[LLM] Stream callback is None!")

        if self.use_local_llm:
            # Ollama streaming
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "format": "json"
            }

            response = self.requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=30
            )
            response.raise_for_status()

            chunk_count = 0
            for line in response.iter_lines():
                if line:
                    chunk_count += 1
                    chunk_data = json.loads(line)
                    if "message" in chunk_data and "content" in chunk_data["message"]:
                        chunk_text = chunk_data["message"]["content"]
                        accumulated_text += chunk_text

                        if self.enable_logging and chunk_count <= 5:
                            self.logger.info(f"[LLM Stream] Chunk {chunk_count}: '{chunk_text}'")

                        # Parse and stream response field character by character
                        for char in chunk_text:
                            response_buffer += char

                            # Detect JSON start
                            if not json_started and char == '{':
                                json_started = True
                                if self.enable_logging:
                                    self.logger.info("[LLM Stream] JSON started")
                                continue

                            if json_started and not in_response_field:
                                # Look for "response": (with flexible whitespace)
                                if '"response"' in response_buffer:
                                    # Look for colon after response
                                    idx = response_buffer.find('"response"')
                                    after = response_buffer[idx + len('"response"'):]
                                    # Check for : and " pattern (allowing whitespace)
                                    colon_idx = after.find(':')
                                    if colon_idx != -1:
                                        after_colon = after[colon_idx + 1:].lstrip()
                                        if after_colon.startswith('"'):
                                            in_response_field = True
                                            if self.enable_logging:
                                                self.logger.info("[LLM Stream] Entered response field")
                                            # Get text after opening quote
                                            text_after_quote = after_colon[1:]
                                            if text_after_quote and text_after_quote[0] != '"':
                                                if stream_callback:
                                                    if self.enable_logging:
                                                        self.logger.info(f"[LLM Stream] Initial text: '{text_after_quote}'")
                                                    stream_callback("response_chunk", text_after_quote)
                                            response_buffer = ""
                            elif in_response_field:
                                # Stream each character until closing quote
                                if char == '"' and (len(response_buffer) <= 1 or response_buffer[-2] != '\\'):
                                    in_response_field = False
                                    if self.enable_logging:
                                        self.logger.info("[LLM Stream] Exited response field")
                                    response_buffer = ""
                                elif char != '\\':
                                    if stream_callback:
                                        stream_callback("response_chunk", char)
                                    response_buffer = ""

            if self.enable_logging:
                self.logger.info(f"[LLM Stream] Total chunks received: {chunk_count}")

            # Return complete response for parsing
            return {"message": {"content": accumulated_text}}, accumulated_text

        else:
            # OpenAI streaming
            stream = self.client.chat.completions.create( # type: ignore
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},  # type: ignore
                stream=True
            )

            chunk_count = 0
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_count += 1
                    chunk_text = chunk.choices[0].delta.content
                    accumulated_text += chunk_text

                    if self.enable_logging and chunk_count <= 5:
                        self.logger.info(f"[LLM Stream] Chunk {chunk_count}: '{chunk_text}'")

                    # Parse and stream response field character by character
                    for char in chunk_text:
                        response_buffer += char

                        # Detect JSON start
                        if not json_started and char == '{':
                            json_started = True
                            if self.enable_logging:
                                self.logger.info("[LLM Stream] JSON started")
                            continue

                        if json_started and not in_response_field:
                            # Look for "response": (with flexible whitespace)
                            if '"response"' in response_buffer:
                                # Look for colon after response
                                idx = response_buffer.find('"response"')
                                after = response_buffer[idx + len('"response"'):]
                                # Check for : and " pattern (allowing whitespace)
                                colon_idx = after.find(':')
                                if colon_idx != -1:
                                    after_colon = after[colon_idx + 1:].lstrip()
                                    if after_colon.startswith('"'):
                                        in_response_field = True
                                        if self.enable_logging:
                                            self.logger.info("[LLM Stream] Entered response field")
                                        # Get text after opening quote
                                        text_after_quote = after_colon[1:]
                                        if text_after_quote and text_after_quote[0] != '"':
                                            if stream_callback:
                                                if self.enable_logging:
                                                    self.logger.info(f"[LLM Stream] Initial text: '{text_after_quote}'")
                                                stream_callback("response_chunk", text_after_quote)
                                        response_buffer = ""
                        elif in_response_field:
                            # Stream each character until closing quote
                            if char == '"' and (len(response_buffer) <= 1 or response_buffer[-2] != '\\'):
                                in_response_field = False
                                if self.enable_logging:
                                    self.logger.info("[LLM Stream] Exited response field")
                                response_buffer = ""
                            elif char != '\\':
                                if stream_callback:
                                    stream_callback("response_chunk", char)
                                response_buffer = ""

            if self.enable_logging:
                self.logger.info(f"[LLM Stream] Total chunks received: {chunk_count}")

            # Create mock response object for parsing
            class MockResponse:
                def __init__(self, content):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {'content': content})()
                    })()]

            if self.enable_logging:
                self.logger.info("[LLM] Streaming complete")

            return MockResponse(accumulated_text), accumulated_text

    def _extract_response_chunk(self, chunk_text: str, json_started: bool,
                               in_response_field: bool, response_buffer: str) -> Optional[Dict]:
        """Extract response text from JSON stream - simplified for character-by-character streaming"""
        result = {
            "json_started": json_started,
            "in_response_field": in_response_field,
            "buffer": response_buffer,
            "text": ""
        }

        for char in chunk_text:
            response_buffer += char

            # Detect JSON start
            if not json_started and char == '{':
                json_started = True
                result["json_started"] = True
                continue

            if json_started:
                # Look for "response": " pattern
                if not in_response_field:
                    # Check if we've found the response field opening
                    if '"response"' in response_buffer:
                        # Look for the colon and opening quote
                        idx = response_buffer.rfind('"response"')
                        after_response = response_buffer[idx + len('"response"'):]

                        # Check if we have ": " pattern
                        colon_quote_match = after_response.find(':"')
                        if colon_quote_match != -1:
                            in_response_field = True
                            result["in_response_field"] = True

                            # Get any text after the opening quote
                            text_start = colon_quote_match + 2
                            if text_start < len(after_response):
                                initial_text = after_response[text_start:]
                                if initial_text and initial_text[0] != '"':
                                    result["text"] = initial_text

                            response_buffer = ""
                else:
                    # We're inside response field - stream each character
                    # Check for unescaped closing quote
                    if char == '"' and (len(response_buffer) == 0 or response_buffer[-1] != '\\'):
                        # End of response field
                        in_response_field = False
                        result["in_response_field"] = False
                        response_buffer = ""
                    else:
                        # Stream this character immediately
                        if char != '\\' or (len(response_buffer) > 1 and response_buffer[-2] == '\\'):
                            result["text"] = char
                        response_buffer = ""

        result["buffer"] = response_buffer

        return result if result["text"] else None

    def _call_llm_api(self, query: str, intent_patterns: Dict,
                     conversation_history: Optional[List[Dict]] = None,
                     recognized_intent: Optional[str] = None):
        """Call LLM API without streaming (Ollama or OpenAI)"""
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
            )

    def _build_user_prompt(self, query: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """Build user prompt with conversation context"""
        if not conversation_history:
            return f'CURRENT CUSTOMER QUERY: "{query}"'

        prompt_parts = ["CONVERSATION HISTORY (recent messages):"]
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history

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
- order: User wants to place an order, buy pizza, or start ordering process
- complaint: User has a problem, complaint, or issue with their order, wants refund or escalate
- hours_location: User asks about business hours, location, or address
- menu_inquiry: User asks about menu items, toppings, prices, or options
- delivery: User asks about delivery status, tracking, fees, or timing
- general: Greetings, thanks, confirmations (hello, thanks, ok, sure, etc.)"""

        if self.test_mode:
            prompt = f"Intent classification for pizza restaurant. Classify into: {', '.join(valid_intents)}\n{intent_descriptions}"
            if recognized_intent:
                prompt += f"\nPrevious layer suggested: {recognized_intent}, override if incorrect."
            prompt += '\n\nRespond ONLY with valid JSON, no markdown formatting:\n{{"intent": "intent_name", "confidence": 0.85}}'
            return prompt

        tts_rules = """Some of CRITICAL RESPONSE RULES for TTS:
1. Keep responses SHORT (1-3 sentences max under 40 words preferably)
2. NO special formatting: no $, %, parentheses, brackets, colons, semicolons
3. Spell out prices (e.g., 'twelve dollars' instead of $12)
4. No multiple questions in one response
5. Do not repeat information already mentioned unless user asks to do so"""

        prompt = f"""You are a helpful voice customer support assistant for {self.res_info['name']}, a pizza restaurant.
Generate ONE natural, conversational response that directly addresses the customer's query.
This response will be used for TTS so make the response adhere to common TTS text rules.
RESTAURANT INFO: {json.dumps(self.res_info, indent=2)}
Offers Pick up and Delivery.
AVAILABLE INTENTS: {', '.join(valid_intents)}
{intent_descriptions}

{tts_rules}"""

        if recognized_intent:
            prompt += f"""\n\nPrevious intent classifier layer suggested: {recognized_intent} intent
The previous layer does not have full conversational context, if you are highly confident that the intent is incorrect, provide the correct intent with a confidence 0 to 1 based on context."""

        prompt += '\n\nRespond with ONLY valid JSON (no markdown code blocks):\n{{"intent": "intent_name", "confidence": 0.85, "response": "Your natural, helpful response here"}}'
        return prompt

    def _process_api_response(self, response, intent_patterns: Dict,
                              recognized_intent: Optional[str] = None,
                              full_response_text: Optional[str] = None) -> LLMResult:
        """Parse and validate API response, creating LLMResult"""
        if full_response_text:
            result_text = full_response_text
        elif self.use_local_llm:
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
            score_breakdown={"llm_explanation": explanation, "response_generated": True},
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