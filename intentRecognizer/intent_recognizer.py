"""
Main Intent Recognition System - Cascade Architecture
Combines algorithmic and LLM recognizers with LLM as fallback
"""

import os
import logging
from typing import Dict, Optional
from dataclasses import dataclass

from intentRecognizer.algorithmic_recognizer import AlgorithmicRecognizer
from intentRecognizer.llm_recognizer import LLMRecognizer


@dataclass
class RecognitionResult:
    """Unified result from intent recognition"""
    intent: str
    confidence: float
    confidence_level: str  # 'high', 'medium', 'low'
    matched_pattern: str
    processing_method: str  # 'keyword', 'levenshtein', 'combined', 'llm'
    used_llm: bool = False
    llm_explanation: str = ""
    score_breakdown: Dict = None


class IntentRecognizer:
    """
    Main Intent Recognizer with Cascade Architecture

    Pipeline:
    1. Algorithmic Recognizer (Pattern Matching + Levenshtein)
    2. LLM Recognizer (Fallback if confidence too low or unknown)
    """

    def __init__(
            self,
            patterns_file: str = None,
            enable_logging: bool = False,
            min_confidence: float = 0.5,
            enable_llm_fallback: bool = True,
            llm_fallback_threshold: float = 0.45,
            api_key: Optional[str] = None,
            model: str = "gpt-4o-mini"
    ):
        """
        Initialize the cascade intent recognizer

        Args:
            patterns_file: Path to JSON file with intent patterns
            enable_logging: Enable detailed logging for analysis
            min_confidence: Minimum confidence threshold (default: 0.5)
            enable_llm_fallback: Enable LLM fallback for unmatched queries
            llm_fallback_threshold: Confidence below which LLM is used (default: 0.45)
            api_key: OpenAI API key (optional, will load from env)
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        # Configuration
        if patterns_file is None:
            utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
            patterns_file = os.path.join(utils_dir, 'intent_patterns.json')


        self.patterns_file = patterns_file
        self.min_confidence = min_confidence
        self.enable_llm_fallback = enable_llm_fallback
        self.llm_fallback_threshold = llm_fallback_threshold

        # Setup logging
        self.enable_logging = enable_logging
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)

        # Initialize Algorithmic Recognizer (Primary)
        self.algorithmic_recognizer = AlgorithmicRecognizer(
            patterns_file=patterns_file,
            enable_logging=enable_logging,
            min_confidence=min_confidence
        )

        # Initialize LLM Recognizer (Fallback)
        self.llm_recognizer = None
        if self.enable_llm_fallback:
            try:
                self.llm_recognizer = LLMRecognizer(
                    api_key=api_key,
                    model=model,
                    enable_logging=enable_logging,
                    min_confidence=min_confidence
                )
                if self.enable_logging:
                    self.logger.info("LLM fallback enabled in cascade architecture")
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Failed to initialize LLM recognizer: {e}")
                self.enable_llm_fallback = False

        # Load patterns for LLM
        self.patterns = self.algorithmic_recognizer.patterns

        # Unified statistics tracking
        self.stats = {
            'total_queries': 0,
            'algorithmic_success': 0,
            'llm_fallback_used': 0,
            'intent_distribution': {},
            'avg_confidence': []
        }

    def recognize_intent(self, query: str) -> RecognitionResult:
        """
        Main method: Recognize intent using cascade architecture

        Pipeline:
        1. Try algorithmic recognition first
        2. If confidence too low or unknown, use LLM fallback

        Args:
            query: User input string (from ASR)

        Returns:
            RecognitionResult object with intent information
        """
        # Update statistics
        self.stats['total_queries'] += 1

        # STEP 1: Try Algorithmic Recognition (Primary)
        algo_result = self.algorithmic_recognizer.recognize(query)

        # Check if algorithmic result is confident enough
        use_llm_fallback = (
            self.enable_llm_fallback and
            self.llm_recognizer and
            (algo_result.intent == "unknown" or algo_result.confidence < self.llm_fallback_threshold)
        )

        # STEP 2: LLM Fallback (if needed)
        if use_llm_fallback:
            if self.enable_logging:
                self.logger.info(
                    f"Algorithmic confidence too low ({algo_result.confidence:.3f}), "
                    f"using LLM fallback for: '{query}'"
                )

            try:
                llm_result = self.llm_recognizer.recognize(query, self.patterns)

                # Use LLM result if it has higher confidence and no error
                if not llm_result.error and llm_result.confidence > algo_result.confidence:
                    # Update statistics
                    self.stats['llm_fallback_used'] += 1
                    self.stats['intent_distribution'][llm_result.intent] = \
                        self.stats['intent_distribution'].get(llm_result.intent, 0) + 1
                    self.stats['avg_confidence'].append(llm_result.confidence)

                    # Create unified result from LLM
                    result = RecognitionResult(
                        intent=llm_result.intent,
                        confidence=llm_result.confidence,
                        confidence_level=llm_result.confidence_level,
                        matched_pattern="LLM Classification",
                        processing_method='llm',
                        used_llm=True,
                        llm_explanation=llm_result.explanation,
                        score_breakdown={}
                    )

                    if self.enable_logging:
                        self.logger.info(
                            f"[CASCADE] Using LLM result: {llm_result.intent} "
                            f"(confidence: {llm_result.confidence:.3f})"
                        )

                    return result

            except Exception as e:
                if self.enable_logging:
                    self.logger.error(f"LLM fallback failed: {e}")

        # STEP 3: Use Algorithmic Result (Default)
        self.stats['algorithmic_success'] += 1
        self.stats['intent_distribution'][algo_result.intent] = \
            self.stats['intent_distribution'].get(algo_result.intent, 0) + 1
        self.stats['avg_confidence'].append(algo_result.confidence)

        # Create unified result from algorithmic
        result = RecognitionResult(
            intent=algo_result.intent,
            confidence=algo_result.confidence,
            confidence_level=algo_result.confidence_level,
            matched_pattern=algo_result.matched_pattern,
            processing_method=algo_result.processing_method,
            used_llm=False,
            llm_explanation="",
            score_breakdown=algo_result.score_breakdown
        )

        if self.enable_logging:
            self.logger.info(
                f"[CASCADE] Using algorithmic result: {algo_result.intent} "
                f"(confidence: {algo_result.confidence:.3f})"
            )

        return result

    def generate_response(self, intent_info: RecognitionResult, message: str) -> str:
        """
        Generate a response based on the recognized intent.

        Args:
            intent_info: RecognitionResult object from recognize_intent
            message: The original user message

        Returns:
            str: Generated response (currently returns intent name)
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
                'used_llm': result.used_llm,
                'correct': is_correct
            })

        accuracy = correct / total if total > 0 else 0.0

        # Calculate metrics by confidence level
        high_conf = [r for r in results if r['confidence'] >= 0.8]
        medium_conf = [r for r in results if 0.6 <= r['confidence'] < 0.8]
        low_conf = [r for r in results if r['confidence'] < 0.6]

        # Calculate metrics by processing method
        llm_results = [r for r in results if r['used_llm']]
        algo_results = [r for r in results if not r['used_llm']]

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
            'algo_used_count': len(algo_results),
            'llm_accuracy': sum(r['correct'] for r in llm_results) / len(llm_results) if llm_results else 0,
            'algo_accuracy': sum(r['correct'] for r in algo_results) / len(algo_results) if algo_results else 0,
            'detailed_results': results
        }

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics from cascade system"""
        avg_conf = sum(self.stats['avg_confidence']) / len(self.stats['avg_confidence']) \
            if self.stats['avg_confidence'] else 0.0

        # Base statistics
        stats_dict = {
            'cascade_architecture': True,
            'total_queries_processed': self.stats['total_queries'],
            'algorithmic_success_count': self.stats['algorithmic_success'],
            'llm_fallback_used_count': self.stats['llm_fallback_used'],
            'intent_distribution': self.stats['intent_distribution'],
            'average_confidence': avg_conf,
            'llm_fallback_rate': (
                self.stats['llm_fallback_used'] / self.stats['total_queries']
                if self.stats['total_queries'] > 0 else 0.0
            )
        }

        # Add algorithmic recognizer statistics
        algo_stats = self.algorithmic_recognizer.get_statistics()
        stats_dict['algorithmic_recognizer'] = algo_stats

        # Add LLM recognizer statistics if enabled
        if self.enable_llm_fallback and self.llm_recognizer:
            try:
                llm_stats = self.llm_recognizer.get_statistics()
                stats_dict['llm_recognizer'] = llm_stats
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Could not get LLM statistics: {e}")

        return stats_dict