"""
Main Intent Recognition System - Configurable Pipeline Architecture
Pipeline: Algorithmic → Semantic → LLM (default)
Each layer is tried when the previous layer fails or has the below threshold confidence
"""

import os
import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_ALGORITHMIC_THRESHOLD = 0.6
DEFAULT_SEMANTIC_THRESHOLD = 0.5
DEFAULT_LLM_MODEL = "gpt-5-nano"
DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"
LLM_OVERRIDE_THRESHOLD = 0.8 # active when test mode is off


@dataclass
class RecognitionResult:
    """Unified result from intent recognition"""
    intent: str
    confidence: float
    confidence_level: str
    matched_pattern: str
    processing_method: str
    layer_used: str
    response: str = ""
    llm_explanation: str = ""
    score_breakdown: Dict = None


class IntentRecognizerUtils:
    """Shared utilities for all recognizer layers"""

    @staticmethod
    def determine_confidence_level(confidence: float) -> str:
        """Determine confidence level based on thresholds"""
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return 'high'
        elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            return 'medium'
        return 'low'

    @staticmethod
    def load_patterns_from_file(patterns_file: str, enable_logging: bool = False) -> Dict:
        """Load intent patterns from JSON file"""
        try:
            with open(patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('intents', data) if 'intents' in data else data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            if enable_logging:
                logger = logging.getLogger(__name__)
                logger.error(f"Error loading patterns: {e}")
            return {}

    @staticmethod
    def get_default_patterns_file() -> str:
        """Get default path to patterns file"""
        utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
        return os.path.join(utils_dir, 'intent_patterns.json')


from .algorithmic_recognizer import AlgorithmicRecognizer
from .semantic_recognizer import SemanticRecognizer
from .llm_recognizer import LLMRecognizer


class IntentRecognizer:
    """
    Configurable Intent Recognition Pipeline

    Each layer can be independently enabled/disabled:
    - Layer 1: Algorithmic (Keyword pattern Matching + Levenshtein)
    - Layer 2: Semantic (Sentence Transformers)
    - Layer 3: LLM (OpenAI API - Fallback)
    """

    def __init__(
            self,
            patterns_file: str = None,
            enable_logging: bool = False,
            min_confidence: float = DEFAULT_MIN_CONFIDENCE,
            enable_algorithmic: bool = True,
            enable_semantic: bool = True,
            enable_llm: bool = True,
            algorithmic_threshold: float = DEFAULT_ALGORITHMIC_THRESHOLD,
            semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
            semantic_model: str = DEFAULT_SEMANTIC_MODEL,
            llm_model: str = DEFAULT_LLM_MODEL,
            test_mode: bool = False,
            use_local_llm: bool = False,
            ollama_base_url: str = None,
    ):
        if not (enable_algorithmic or enable_semantic or enable_llm):
            raise ValueError("Invalid configuration: At least one layer must be enabled.")

        self.patterns_file = patterns_file or IntentRecognizerUtils.get_default_patterns_file()
        self.min_confidence = min_confidence
        self.enable_algorithmic = enable_algorithmic
        self.enable_semantic = enable_semantic
        self.enable_llm = enable_llm
        self.test_mode = test_mode
        self.algorithmic_threshold = algorithmic_threshold if (enable_semantic or enable_llm) else 0
        self.semantic_threshold = semantic_threshold if enable_llm else 0
        self.enable_logging = enable_logging
        self.use_local_llm = use_local_llm
        self.ollama_base_url = ollama_base_url

        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)

        self.stats = {
            'total_queries': 0,
            'algorithmic_used': 0,
            'semantic_used': 0,
            'llm_used': 0,
            'llm_overrides': 0,
            'intent_distribution': {},
            'avg_confidence': [],
            'layer_configuration': {
                'algorithmic': enable_algorithmic,
                'semantic': enable_semantic,
                'llm': enable_llm
            }
        }

        self.patterns = IntentRecognizerUtils.load_patterns_from_file(
            self.patterns_file, self.enable_logging
        )

        self.algorithmic_recognizer = None
        self.semantic_recognizer = None
        self.llm_recognizer = None

        self._initialize_layers(semantic_model, llm_model)

    def _initialize_layers(self, semantic_model: str, llm_model: str):
        """Initialize all enabled layers"""
        if self.enable_algorithmic:
            self.algorithmic_recognizer = AlgorithmicRecognizer(
                patterns_file=self.patterns_file,
                enable_logging=self.enable_logging,
                min_confidence=self.min_confidence
            )
            if self.enable_logging:
                self.logger.info(" Algorithmic layer initialized")

        if self.enable_semantic:
            try:
                self.semantic_recognizer = SemanticRecognizer(
                    patterns_file=self.patterns_file,
                    model_name=semantic_model,
                    enable_logging=self.enable_logging,
                    min_confidence=self.min_confidence,
                    use_cache=True
                )
                if self.enable_logging:
                    self.logger.info(f" Semantic layer initialized (model: {semantic_model})")
            except ImportError as e:
                if self.enable_logging:
                    self.logger.error(
                        f"Semantic layer failed - missing dependencies: {e}\n"
                        "  Install with: pip install sentence-transformers scikit-learn"
                    )
                raise RuntimeError(
                    "Cannot initialize semantic layer - missing dependencies. "
                    "Install with: pip install sentence-transformers scikit-learn"
                )
            except Exception as e:
                if self.enable_logging:
                    self.logger.error(f" Semantic layer initialization failed: {e}")
                raise

        if self.enable_llm:
            try:
                self.llm_recognizer = LLMRecognizer(
                    model=llm_model,
                    enable_logging=self.enable_logging,
                    min_confidence=self.min_confidence,
                    use_local_llm=self.use_local_llm,
                    ollama_base_url=self.ollama_base_url,
                )
                provider_type = "Ollama" if self.use_local_llm else "OpenAI"
                if self.enable_logging:
                    self.logger.info(f" LLM layer initialized ({provider_type}, model: {llm_model})")
            except Exception as e:
                if self.enable_logging:
                    self.logger.error(f" LLM layer initialization failed: {e}")
                raise

    def recognize_intent(self, query: str,
                        conversation_history: Optional[List[Dict]] = None) -> RecognitionResult:
        """
        Main recognition method - routes through enabled layers

        Pipeline Flow:
        1. Try first enabled layer
        2. If confidence below threshold, try next enabled layer
        3. Return best result with generated response
        """
        self.stats['total_queries'] += 1

        if self.enable_logging:
            self.logger.info("-"*50)
            self.logger.info(f"Processing Query: '{query}'")
            self.logger.info("-"*50)

        recognized_intent = None

        if self.enable_algorithmic:
            result = self._try_algorithmic_layer(query, conversation_history)
            if result is not None:
                return result
            if self.algorithmic_recognizer:
                algo_result = self.algorithmic_recognizer.recognize(query)
                if algo_result.intent != "unknown":
                    recognized_intent = algo_result.intent

        if self.enable_semantic:
            result = self._try_semantic_layer(query, conversation_history)
            if result is not None:
                return result
            if self.semantic_recognizer:
                semantic_result = self.semantic_recognizer.recognize(query)
                if semantic_result.intent != "unknown":
                    recognized_intent = semantic_result.intent

        if self.enable_llm:
            result = self._try_llm_layer(query, conversation_history, recognized_intent)
            if result is not None:
                return result

        return self._create_unknown_result("No layers produced a result")

    def _try_algorithmic_layer(self, query: str,
                               conversation_history: Optional[List[Dict]] = None) -> Optional[RecognitionResult]:
        """Try algorithmic pattern matching layer"""
        algo_result = self.algorithmic_recognizer.recognize(query)

        if algo_result.intent == "unknown" or algo_result.confidence < self.algorithmic_threshold:
            if self.enable_logging:
                if self.enable_semantic:
                    self.logger.info("  - Proceeding to Semantic layer")
                elif self.enable_llm:
                    self.logger.info("  - Proceeding to LLM layer")
            return None

        self.stats['algorithmic_used'] += 1
        return self._create_result(
            query, algo_result, layer='algorithmic',
            conversation_history=conversation_history,
            original_confidence=algo_result.confidence
        )

    def _try_semantic_layer(self, query: str,
                           conversation_history: Optional[List[Dict]] = None) -> Optional[RecognitionResult]:
        """Try semantic similarity layer"""
        semantic_result = self.semantic_recognizer.recognize(query)

        if self.enable_llm and (
            semantic_result.intent == "unknown" or
            semantic_result.confidence < self.semantic_threshold
        ):
            if self.enable_logging:
                self.logger.info("  - Proceeding to LLM layer")
            return None

        self.stats['semantic_used'] += 1
        return self._create_result(
            query, semantic_result, layer='semantic',
            conversation_history=conversation_history,
            original_confidence=semantic_result.confidence
        )

    def _try_llm_layer(self, query: str,
                      conversation_history: Optional[List[Dict]] = None,
                      recognized_intent: Optional[str] = None) -> RecognitionResult:
        """Try LLM fallback layer"""
        llm_result = self.llm_recognizer.recognize(
            query, self.patterns, conversation_history, recognized_intent
        )
        self.stats['llm_used'] += 1
        return self._create_result(
            query, llm_result, layer='llm',
            conversation_history=conversation_history
        )

    def _create_result(self, query, result, layer: str,
                       conversation_history: Optional[List[Dict]] = None,
                       original_confidence: float = None) -> RecognitionResult:
        """Create unified RecognitionResult from any layer with LLM override capability"""

        # Store original intent and confidence for potential override
        original_intent = result.intent
        original_conf = original_confidence if original_confidence is not None else result.confidence

        response = getattr(result, 'response', '')

        if self.test_mode:
            # TEST MODE: Skip all response generation for faster intent testing
            response = ""
        elif not response and self.enable_llm and layer != 'llm':
            # Use LLM for response generation
            if self.enable_logging:
                self.logger.info(f"  - Using LLM for response generation")

            try:
                llm_result = self.llm_recognizer.recognize(
                    query, self.patterns, conversation_history, result.intent
                )

                # Check if LLM should override the intent
                if (llm_result.confidence >= LLM_OVERRIDE_THRESHOLD and
                    llm_result.confidence > original_conf and
                    llm_result.intent != original_intent and
                    llm_result.intent != "unknown"):

                    if self.enable_logging:
                        self.logger.info(
                            f"  - LLM OVERRIDE: Changing intent from '{original_intent}' "
                            f"({original_conf:.3f}) to '{llm_result.intent}' ({llm_result.confidence:.3f})"
                        )

                    # Update intent and confidence to LLM's classification
                    result.intent = llm_result.intent
                    result.confidence = llm_result.confidence
                    result.confidence_level = llm_result.confidence_level
                    self.stats['llm_overrides'] += 1

                response = llm_result.response

                # Check if LLM response is valid
                if not response or response.strip() == "":
                    if self.enable_logging:
                        self.logger.warning(f"LLM returned empty response, using fallback")
                    response = self._generate_simple_response(result.intent, query, conversation_history)
                else:
                    if self.enable_logging:
                        self.logger.debug(f"[LLM Response] Generated: {response[:100]}...")
            except json.JSONDecodeError as e:
                if self.enable_logging:
                    self.logger.warning(f"LLM response JSON parse error: {e}, using fallback")
                response = self._generate_simple_response(result.intent)
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"LLM response generation failed: {e}, using fallback")
                response = self._generate_simple_response(result.intent)
        elif not response and not self.test_mode:
            # Fallback to simple response if LLM not configured or not available
            response = self._generate_simple_response(result.intent)

        # Update stats
        self.stats['intent_distribution'][result.intent] = \
            self.stats['intent_distribution'].get(result.intent, 0) + 1
        self.stats['avg_confidence'].append(result.confidence)

        if self.enable_logging:
            self.logger.info(
                f" Accepted from {layer.upper()} layer: "
                f"Query: '{query}' -> "
                f"{result.intent} (confidence: {result.confidence:.3f}, "
                f"level: {result.confidence_level})"
            )

        return RecognitionResult(
            intent=result.intent,
            confidence=result.confidence,
            confidence_level=result.confidence_level,
            matched_pattern=getattr(result, 'matched_pattern', 'ML Classification'),
            processing_method=getattr(result, 'processing_method', layer),
            layer_used=layer,
            response=response,
            llm_explanation=getattr(result, 'explanation', ''),
            score_breakdown=getattr(result, 'score_breakdown', {})
        )

    def _generate_simple_response(self, intent: str) -> str:
        """Generate simple response for non-LLM layers"""
        responses = {
            'order': "I'd be happy to help you place an order! What size pizza would you like?",
            'complaint': "I apologize for any inconvenience. Can you please tell me more about the issue?",
            'hours_location': "We're open Monday to Friday eleven AM to ten PM, Saturday eleven AM to eleven PM, Sunday twelve PM to nine PM. Located at one twenty three Main Street, Downtown.",
            'menu_inquiry': "Our menu features various pizzas in Small, Medium, Large, and Family sizes. We have thin, regular, thick, and gluten-free crusts. What would you like to know more about?",
            'delivery': "I can help you with delivery information. Our delivery fee is three dollars, and typical delivery time is thirty to forty five minutes.",
            'general': "Hello! How can I help you today?",
            'unknown': "I'm here to help! Could you please clarify what you'd like to know about our pizza restaurant?"
        }
        return responses.get(intent, responses['unknown'])

    def _create_unknown_result(self, reason: str) -> RecognitionResult:
        """Create unknown result with given reason"""
        return RecognitionResult(
            intent='unknown',
            confidence=0.0,
            confidence_level='low',
            matched_pattern='',
            processing_method='error',
            layer_used='none',
            response="I'm here to help! Could you please clarify what you'd like to know?",
            llm_explanation=reason,
            score_breakdown={'error': reason}
        )

    def generate_response(self, intent_info: RecognitionResult, message: str) -> str:
        """
        Generate a response based on the recognized intent
        (Deprecated - response is now generated during recognition)
        """
        return intent_info.response if intent_info.response else intent_info.intent

    def evaluate(self, test_data: list) -> Dict:
        """Evaluate recognizer accuracy on test data"""
        results = []
        correct = 0
        total = len(test_data)

        for query, expected_intent in test_data:
            result = self.recognize_intent(query)
            is_correct = result.intent == expected_intent
            if is_correct:
                correct += 1

            results.append({
                'query': query,
                'expected': expected_intent,
                'predicted': result.intent,
                'confidence': result.confidence,
                'layer_used': result.layer_used,
                'correct': is_correct
            })

        accuracy = correct / total if total > 0 else 0.0

        high_conf = [r for r in results if r['confidence'] >= HIGH_CONFIDENCE_THRESHOLD]
        medium_conf = [r for r in results if MEDIUM_CONFIDENCE_THRESHOLD <= r['confidence'] < HIGH_CONFIDENCE_THRESHOLD]
        low_conf = [r for r in results if r['confidence'] < MEDIUM_CONFIDENCE_THRESHOLD]

        llm_results = [r for r in results if r['layer_used'] == 'llm']
        semantic_results = [r for r in results if r['layer_used'] == 'semantic']
        algo_results = [r for r in results if r['layer_used'] == 'algorithmic']

        return {
            'accuracy': accuracy,
            'total_queries': total,
            'correct': correct,
            'incorrect': total - correct,
            'high_confidence_count': len(high_conf),
            'medium_confidence_count': len(medium_conf),
            'low_confidence_count': len(low_conf),
            'high_confidence_accuracy': sum(r['correct'] for r in high_conf) / len(high_conf) if high_conf else 0,
            'llm_used_count': len(llm_results),
            'semantic_used_count': len(semantic_results),
            'algo_used_count': len(algo_results),
            'llm_accuracy': sum(r['correct'] for r in llm_results) / len(llm_results) if llm_results else 0,
            'semantic_accuracy': sum(r['correct'] for r in semantic_results) / len(semantic_results) if semantic_results else 0,
            'algo_accuracy': sum(r['correct'] for r in algo_results) / len(algo_results) if algo_results else 0,
            'detailed_results': results
        }

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics from the pipeline"""
        avg_conf = sum(self.stats['avg_confidence']) / len(self.stats['avg_confidence']) \
            if self.stats['avg_confidence'] else 0.0

        total = self.stats['total_queries']

        stats_dict = {
            'pipeline_configuration': {
                'algorithmic_enabled': self.enable_algorithmic,
                'semantic_enabled': self.enable_semantic,
                'llm_enabled': self.enable_llm,
                'algorithmic_threshold': self.algorithmic_threshold if self.enable_algorithmic else None,
                'semantic_threshold': self.semantic_threshold if self.enable_semantic else None,
            },
            'total_queries_processed': total,
            'layer_usage': {
                'algorithmic': {
                    'count': self.stats['algorithmic_used'],
                    'percentage': (self.stats['algorithmic_used'] / total * 100) if total > 0 else 0
                },
                'semantic': {
                    'count': self.stats['semantic_used'],
                    'percentage': (self.stats['semantic_used'] / total * 100) if total > 0 else 0
                },
                'llm': {
                    'count': self.stats['llm_used'],
                    'percentage': (self.stats['llm_used'] / total * 100) if total > 0 else 0
                },
                'llm_overrides': self.stats.get('llm_overrides', 0)
            },
            'intent_distribution': self.stats['intent_distribution'],
            'average_confidence': avg_conf
        }

        if self.enable_algorithmic and self.algorithmic_recognizer:
            stats_dict['algorithmic_layer'] = self.algorithmic_recognizer.get_statistics()

        if self.enable_semantic and self.semantic_recognizer:
            try:
                stats_dict['semantic_layer'] = self.semantic_recognizer.get_statistics()
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Could not get semantic statistics: {e}")

        if self.enable_llm and self.llm_recognizer:
            try:
                stats_dict['llm_layer'] = self.llm_recognizer.get_statistics()
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Could not get LLM statistics: {e}")

        return stats_dict