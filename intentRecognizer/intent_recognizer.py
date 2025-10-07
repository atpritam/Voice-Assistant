"""
Main Intent Recognition System - Configurable Pipeline Architecture
Pipeline: Algorithmic → Semantic → LLM (default)
Each layer is tried when previous layer fails or has below threshold confidence
"""

import os
import json
import logging
from typing import Dict, Optional
from dataclasses import dataclass

# Shared Constants
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_ALGORITHMIC_THRESHOLD = 0.6
DEFAULT_SEMANTIC_THRESHOLD = 0.5
DEFAULT_LLM_MODEL = "gpt-5-nano"
DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"


@dataclass
class RecognitionResult:
    """Unified result from intent recognition"""
    intent: str
    confidence: float
    confidence_level: str  # 'high', 'medium', 'low'
    matched_pattern: str
    processing_method: str
    layer_used: str  # 'algorithmic', 'semantic', or 'llm'
    llm_explanation: str = ""
    score_breakdown: Dict = None


class IntentRecognizerUtils:
    """Shared utilities for all recognizer layers"""

    @staticmethod
    def determine_confidence_level(confidence: float) -> str:
        """
        Determine confidence level based on thresholds

        Args:
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            'high', 'medium', or 'low'
        """
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return 'high'
        elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            return 'medium'
        return 'low'

    @staticmethod
    def load_patterns_from_file(patterns_file: str, enable_logging: bool = False) -> Dict:
        """
        Load intent patterns from JSON file

        Args:
            patterns_file: Path to JSON file
            enable_logging: Whether to log errors

        Returns:
            Dictionary of intent patterns
        """
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
        """
        Get default path to patterns file

        Returns:
            Path to intent_patterns.json
        """
        utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
        return os.path.join(utils_dir, 'intent_patterns.json')


from intentRecognizer.algorithmic_recognizer import AlgorithmicRecognizer
from intentRecognizer.semantic_recognizer import SemanticRecognizer
from intentRecognizer.llm_recognizer import LLMRecognizer


class IntentRecognizer:
    """
    Configurable Intent Recognition Pipeline

    Each layer can be independently enabled/disabled:
    - Layer 1: Algorithmic (Keyword pattern Matching + Levenshtein)
    - Layer 2: Semantic (Sentence Transformers)
    - Layer 3: LLM (OpenAI API - Fallback)

    Configuration Examples:
    1. Full Pipeline: Algorithm → Semantic → LLM (default)
    2. Algorithm + LLM: Skip semantic layer
    3. Semantic + LLM: Skip algorithmic layer
    4. LLM Only: Single layer processing
    """

    def __init__(
            self,
            patterns_file: str = None,
            enable_logging: bool = False,
            min_confidence: float = DEFAULT_MIN_CONFIDENCE,

            # Layer Enable/Disable Switches
            enable_algorithmic: bool = True,
            enable_semantic: bool = True,
            enable_llm: bool = True,

            # Layer Thresholds
            algorithmic_threshold: float = DEFAULT_ALGORITHMIC_THRESHOLD,
            semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,

            # Model Configuration
            semantic_model: str = DEFAULT_SEMANTIC_MODEL,
            llm_model: str = DEFAULT_LLM_MODEL,
    ):
        """
        Args:
            patterns_file: Path to JSON file with intent patterns
            enable_logging: Enable detailed logging for analysis
            min_confidence: Minimum confidence threshold

            enable_algorithmic: Enable algorithmic pattern matching layer
            enable_semantic: Enable semantic similarity layer
            enable_llm: Enable LLM fallback layer

            algorithmic_threshold: Confidence below which next layer is tried
            semantic_threshold: Confidence below which LLM layer is tried

            semantic_model: Sentence transformer model name
            llm_model: OpenAI model to use
        """

        if not (enable_algorithmic or enable_semantic or enable_llm):
            raise ValueError(
                "Invalid configuration: At least one layer must be enabled. "
            )

        if patterns_file is None:
            patterns_file = IntentRecognizerUtils.get_default_patterns_file()

        self.patterns_file = patterns_file
        self.min_confidence = min_confidence

        self.enable_algorithmic = enable_algorithmic
        self.enable_semantic = enable_semantic
        self.enable_llm = enable_llm

        self.algorithmic_threshold = algorithmic_threshold if (self.enable_semantic or self.enable_llm) else 0
        self.semantic_threshold = semantic_threshold if self.enable_llm else 0

        self.stats = {
            'total_queries': 0,
            'algorithmic_used': 0,
            'semantic_used': 0,
            'llm_used': 0,
            'intent_distribution': {},
            'avg_confidence': [],
            'layer_configuration': {
                'algorithmic': self.enable_algorithmic,
                'semantic': self.enable_semantic,
                'llm': self.enable_llm
            }
        }

        # Setup logging
        self.enable_logging = enable_logging
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)

        # Initialize enabled layers
        self.algorithmic_recognizer = None
        self.semantic_recognizer = None
        self.llm_recognizer = None

        if self.enable_algorithmic:
            self.algorithmic_recognizer = self._initialize_algorithmic_recognizer()

        if self.enable_semantic:
            self.semantic_recognizer = self._initialize_semantic_recognizer(semantic_model)

        if self.enable_llm:
            self.llm_recognizer = self._initialize_llm_recognizer(llm_model)

        self.patterns = IntentRecognizerUtils.load_patterns_from_file(
            self.patterns_file,
            self.enable_logging
        )

    def _initialize_algorithmic_recognizer(self) -> AlgorithmicRecognizer:
        """Initialize algorithmic keyword pattern matching layer"""
        recognizer = AlgorithmicRecognizer(
            patterns_file=self.patterns_file,
            enable_logging=self.enable_logging,
            min_confidence=self.min_confidence
        )
        if self.enable_logging:
            self.logger.info(" Algorithmic layer initialized")
        return recognizer

    def _initialize_semantic_recognizer(self, model: str) -> Optional[SemanticRecognizer]:
        """Initialize semantic similarity layer"""
        try:
            recognizer = SemanticRecognizer(
                patterns_file=self.patterns_file,
                model_name=model,
                enable_logging=self.enable_logging,
                min_confidence=self.min_confidence,
                use_cache=True
            )
            if self.enable_logging:
                self.logger.info(f" Semantic layer initialized (model: {model})")
            return recognizer

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

    def _initialize_llm_recognizer(self, model: str) -> Optional[LLMRecognizer]:
        """Initialize LLM fallback layer"""
        try:
            recognizer = LLMRecognizer(
                model=model,
                enable_logging=self.enable_logging,
                min_confidence=self.min_confidence
            )
            if self.enable_logging:
                self.logger.info(f" LLM layer initialized (model: {model})")
            return recognizer

        except Exception as e:
            if self.enable_logging:
                self.logger.error(f" LLM layer initialization failed: {e}")
            raise

    def recognize_intent(self, query: str) -> RecognitionResult:
        """
        Main recognition method - routes through enabled layers

        Pipeline Flow:
        1. Try first enabled layer
        2. If confidence below threshold, try next enabled layer
        3. Return best result

        Args:
            query: User input string

        Returns:
            RecognitionResult object with intent information
        """
        self.stats['total_queries'] += 1

        if self.enable_logging:
            self.logger.info("-"*50)
            self.logger.info(f"Processing Query: '{query}'")
            self.logger.info("-"*50)

        if self.enable_algorithmic:
            result = self._try_algorithmic_layer(query)
            if result is not None:
                return result

        if self.enable_semantic:
            result = self._try_semantic_layer(query)
            if result is not None:
                return result

        if self.enable_llm:
            result = self._try_llm_layer(query)
            if result is not None:
                return result

        # Technically, it should never reach this point but for absolute worst
        return self._create_unknown_result("No layers produced a result")


    def _try_algorithmic_layer(self, query: str) -> Optional[RecognitionResult]:
        """
        Try algorithmic pattern matching layer

        Returns:
            RecognitionResult if confident enough, None to try next layer
        """

        algo_result = self.algorithmic_recognizer.recognize(query)

        # Check if we should proceed to next layer
        if algo_result.intent == "unknown" or algo_result.confidence < self.algorithmic_threshold:
            if self.enable_logging:
                if self.enable_semantic:
                    self.logger.info("  - Proceeding to Semantic layer")
                elif self.enable_llm:
                    self.logger.info("  - Proceeding to LLM layer")
            return None

        # Use algorithmic result
        self.stats['algorithmic_used'] += 1
        return self._create_result(query, algo_result, layer='algorithmic')

    def _try_semantic_layer(self, query: str) -> Optional[RecognitionResult]:
        """
        Try semantic similarity layer

        Returns:
            RecognitionResult if confident enough, None to try next layer
        """

        semantic_result = self.semantic_recognizer.recognize(query)

        # Check if we should proceed to LLM layer
        if self.enable_llm and (
            semantic_result.intent == "unknown" or
            semantic_result.confidence < self.semantic_threshold
        ):
            if self.enable_logging:
                self.logger.info("  - Proceeding to LLM layer")
            return None

        # Use semantic result
        self.stats['semantic_used'] += 1
        return self._create_result(query, semantic_result, layer='semantic')

    def _try_llm_layer(self, query: str) -> RecognitionResult:
        """
        Try LLM fallback layer (always returns a result)

        Returns:
            RecognitionResult from LLM
        """

        llm_result = self.llm_recognizer.recognize(query, self.patterns)

        self.stats['llm_used'] += 1
        return self._create_result(query, llm_result, layer='llm')

    def _create_result(self, query, result, layer: str) -> RecognitionResult:
        """
        Create unified RecognitionResult from any layer

        Args:
            result: AlgorithmicResult, SemanticResult, or LLMResult
            layer: Layer name ('algorithmic', 'semantic', or 'llm')

        Returns:
            RecognitionResult object
        """
        # Update statistics
        self.stats['intent_distribution'][result.intent] = \
            self.stats['intent_distribution'].get(result.intent, 0) + 1
        self.stats['avg_confidence'].append(result.confidence)

        if self.enable_logging:
            self.logger.info(
                f" Accepted from {layer.upper()} layer: "
                f"Query: '{query}' → "
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
            llm_explanation=getattr(result, 'explanation', ''),
            score_breakdown=getattr(result, 'score_breakdown', {})
        )

    def _create_unknown_result(self, reason: str) -> RecognitionResult:
        """Create unknown result with given reason"""
        return RecognitionResult(
            intent='unknown',
            confidence=0.0,
            confidence_level='low',
            matched_pattern='',
            processing_method='error',
            layer_used='none',
            llm_explanation=reason,
            score_breakdown={'error': reason}
        )

    def generate_response(self, intent_info: RecognitionResult, message: str) -> str:
        """
        Generate a response based on the recognized intent

        Args:
            intent_info: RecognitionResult object from recognize_intent
            message: The original user message

        Returns:
            Generated response (currently returns intent name)
        """
        # Simple response - returns intent name
        # In production, this would generate contextual responses
        return intent_info.intent

    def evaluate(self, test_data: list) -> Dict:
        """
        Evaluate recognizer accuracy on test data

        Args:
            test_data: List of (query, expected_intent) tuples

        Returns:
            Dictionary with evaluation metrics
        """
        results = []
        correct = 0
        total = len(test_data)

        # Run evaluation on all test cases
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

        # Calculate overall accuracy
        accuracy = correct / total if total > 0 else 0.0

        # Calculate metrics by confidence level
        high_conf = [r for r in results if r['confidence'] >= HIGH_CONFIDENCE_THRESHOLD]
        medium_conf = [r for r in results if MEDIUM_CONFIDENCE_THRESHOLD <= r['confidence'] < HIGH_CONFIDENCE_THRESHOLD]
        low_conf = [r for r in results if r['confidence'] < MEDIUM_CONFIDENCE_THRESHOLD]

        # Calculate metrics by layer
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

        # Base statistics
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
                }
            },
            'intent_distribution': self.stats['intent_distribution'],
            'average_confidence': avg_conf
        }

        # Add individual layer statistics if enabled
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