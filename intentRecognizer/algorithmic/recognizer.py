"""
Algorithmic Intent Recognizer
Handles pattern matching, keyword analysis, and Levenshtein distance-based recognition
"""

import os
import sys
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.logger import ConditionalLogger
from utils.statistics import StatisticsHelper
from utils.text_processor import TextProcessor

from ..intent_recognizer import IntentRecognizerUtils
from .boostEngine import BoostRuleEngine
from .resource_loader import LinguisticResourceLoader
from .tfidf import InvertedIndex
from .similarity import SimilarityCalculator

# EARLY EXIT THRESHOLDS
INTENT_HIGH_SIMILARITY_EXIT = 0.85          # Stops checking other intents
HIGH_SIMILARITY_IN_INTENT_EXIT = 0.85       # Stops checking other patterns within intent


@dataclass
class AlgorithmicResult:
    """Result from algorithmic recognition"""
    intent: str
    confidence: float
    confidence_level: str
    matched_pattern: str
    processing_method: str
    score_breakdown: Dict


@dataclass
class IntentEvaluation:
    """Result of evaluating a single intent"""
    similarity: float
    pattern: str
    breakdown: Dict


class AlgorithmicRecognizer:
    """Pattern-based intent recognition using keywords and string similarity"""

    def __init__(self, patterns_file: str = None, enable_logging: bool = False,
                 min_confidence: float = 0.5, linguistic_resources_file: str = None, use_boost_engine: bool = True,
                 algorithmic_threshold: float = 0.65):
        """Initialize the algorithmic recognizer

        Args:
            patterns_file: Path to intent patterns JSON
            enable_logging: Enable detailed logging
            min_confidence: Minimum confidence threshold
            linguistic_resources_file: Path to linguistic resources JSON
            use_boost_engine: Enable domain-specific contextual boost rules
            algorithmic_threshold: System-level threshold for algorithmic layer
        """
        self.patterns_file = patterns_file or IntentRecognizerUtils.get_default_patterns_file()
        self.min_confidence = min_confidence
        self.algorithmic_threshold = algorithmic_threshold
        self.enable_logging = enable_logging
        self.logger = ConditionalLogger(__name__, enable_logging)

        self.patterns = IntentRecognizerUtils.load_patterns_from_file(self.patterns_file, enable_logging)

        resources = LinguisticResourceLoader.load_resources(linguistic_resources_file)
        self.synonyms = resources['synonyms']
        self.filler_words = resources['filler_words']
        self.intent_critical_keywords = resources['intent_critical_keywords']
        self.synonym_lookup = LinguisticResourceLoader.build_synonym_lookup(self.synonyms)

        self.text_processor = TextProcessor(self.filler_words)
        self.similarity_calculator = SimilarityCalculator(self.text_processor, self.synonym_lookup)
        self.use_boost_engine = use_boost_engine
        if self.use_boost_engine:
            self.boost_engine = BoostRuleEngine(self.intent_critical_keywords, self.synonyms, enable_logging)

        self.inverted_index = InvertedIndex(self.patterns, self.text_processor)

        self._intent_keywords_cache = {}
        self._normalized_patterns_cache = {}
        self._preprocess_patterns()

        self.stats = StatisticsHelper.init_base_stats(
            intents_evaluated=[],
            patterns_evaluated=[]
        )

    def _preprocess_patterns(self):
        """Preprocess all patterns for faster matching"""
        for intent_name, intent_data in self.patterns.items():
            if intent_name == "unknown":
                continue

            keywords = set()
            normalized_patterns = []

            for pattern in intent_data.get("patterns", []):
                normalized = self.text_processor.normalize(pattern)
                keywords.update(normalized.split())
                normalized_patterns.append(normalized)

            self._intent_keywords_cache[intent_name] = keywords
            self._normalized_patterns_cache[intent_name] = normalized_patterns

    def _preprocess_but_clause(self, query: str) -> str:
        """
        Handle 'but' clause in long queries without other separators

        Strategy: In complex queries with 'but', the clause after 'but' typically
        contains the primary intent while the first part provides context.
        """
        query_lower = query.lower()
        words = query.split()

        if ("but" in query_lower and
            query_lower.count("but") == 1 and
            len(words) >= 8 and
            "and" not in query_lower):

            parts = query.split("but", 1)
            second_part = parts[1].strip()

            if len(second_part.split()) >= 3:
                return second_part

        return query

    def _filter_patterns_by_length(self, query: str, patterns: List[str],
                                            normalized_patterns: List[str]) -> List[Tuple[int, str, str]]:
        """Filter patterns by length difference"""
        query_len = len(query)

        if query_len < 20:
            max_diff = 40
        elif query_len < 60:
            max_diff = 30
        else:
            max_diff = min(120, int(query_len * 0.75))

        return [(i, p, norm) for i, (p, norm) in enumerate(zip(patterns, normalized_patterns))
                if abs(query_len - len(p)) <= max_diff]

    def _evaluate_single_intent(self, intent_name: str, intent_data: Dict,
                                         query: str, query_words: Set[str]) -> Optional[IntentEvaluation]:
        """Intent evaluation with conservative filtering

        Args:
            intent_name: Name of the intent to evaluate
            intent_data: Intent configuration data
            query: User query
            query_words: Set of words from the query

        Returns:
            IntentEvaluation with best match or None if no patterns
        """
        patterns = intent_data.get("patterns", [])
        normalized_patterns = self._normalized_patterns_cache.get(intent_name, [])

        if not patterns:
            return None

        filtered = [(i, p, normalized_patterns[i] if i < len(normalized_patterns)
                        else self.text_processor.normalize(p)) for i, p in enumerate(patterns)]

        if not filtered:
            return None

        # Filter patterns with word overlap
        patterns_to_check = []
        for idx, pattern, pattern_norm in filtered:
            pattern_words = set(pattern.split())
            if query_words & pattern_words or len(pattern_words) <= 3:
                patterns_to_check.append((idx, pattern, pattern_norm))

        if not patterns_to_check:
            patterns_to_check = filtered

        max_similarity = 0.0
        best_pattern = ""
        best_breakdown = {}

        for idx, pattern, pattern_norm in patterns_to_check:
            similarity, metrics = self.similarity_calculator.calculate_similarity(
                query, pattern, intent_name, pattern_norm, self.intent_critical_keywords
            )

            if similarity > max_similarity:
                max_similarity = similarity
                best_pattern = pattern
                best_breakdown = metrics.to_dict()

            if similarity > HIGH_SIMILARITY_IN_INTENT_EXIT:
                break

        self.stats['patterns_evaluated'].append(len(patterns_to_check))

        return IntentEvaluation(max_similarity, best_pattern, best_breakdown)

    def _evaluate_all_intents(self, query: str, query_words: Set[str],
                                        early_exit_threshold: Optional[float] = None) -> Dict:
        """Intent evaluation using inverted index

        Args:
            query: User query
            query_words: Set of words from the query
            early_exit_threshold: Stop evaluation if this similarity is reached

        Returns:
            Dictionary mapping intent names to their evaluation results
        """

        candidate_intents = self.inverted_index.get_candidate_intents(query_words)

        if not candidate_intents:
            candidate_intents = [(intent, 0.0) for intent in self.patterns.keys() if intent != 'unknown']

        self.logger.debug(f"Candidate intents: {[intent for intent, _ in candidate_intents]}")

        intent_scores = {}
        intents_processed = 0

        for intent_name, _ in candidate_intents:
            intents_processed += 1

            intent_data = self.patterns.get(intent_name)
            if not intent_data:
                continue

            evaluation = self._evaluate_single_intent(intent_name, intent_data, query, query_words)
            if evaluation and evaluation.similarity > 0:
                intent_scores[intent_name] = {
                    'similarity': evaluation.similarity,
                    'pattern': evaluation.pattern,
                    'breakdown': evaluation.breakdown
                }

                if early_exit_threshold is not None and evaluation.similarity >= early_exit_threshold:
                    break

        self.stats['intents_evaluated'].append(intents_processed)

        return intent_scores

    def _select_best_intent(self, intent_scores: Dict) -> Tuple[str, float, str, Dict]:
        """Select best intent from scores and apply threshold

        Args:
            intent_scores: Dictionary of intent scores

        Returns:
            Tuple of (intent_name, similarity, pattern, breakdown)
        """
        if not intent_scores:
            return "unknown", 0.0, "", {}

        best_intent_name = max(intent_scores.items(), key=lambda x: x[1]['similarity'])[0]
        best_data = intent_scores[best_intent_name]
        best_similarity = best_data['similarity']
        best_pattern = best_data['pattern']
        best_breakdown = best_data['breakdown']

        threshold = self.patterns.get(best_intent_name, {}).get('similarity_threshold', self.min_confidence)

        if best_similarity < threshold:
            return "unknown", best_similarity, best_pattern, best_breakdown

        return best_intent_name, best_similarity, best_pattern, best_breakdown

    def find_best_match(self, query: str) -> Tuple[str, float, str, Dict]:
        """Find the best matching intent for a given query

        Args:
            query: User query

        Returns:
            Tuple of (intent_name, similarity, pattern, breakdown)
        """
        if not query or not self.patterns:
            return "unknown", 0.0, "", {}

        # but-clause preprocessing
        processed_query = self._preprocess_but_clause(query)
        but_clause_applied = processed_query != query

        query_words = set(processed_query.split())
        if not query_words:
            return "unknown", 0.0, "", {}

        early_exit_threshold = INTENT_HIGH_SIMILARITY_EXIT

        intent_scores = self._evaluate_all_intents(
            processed_query,
            query_words,
            early_exit_threshold=early_exit_threshold
        )
        if not intent_scores:
            return "unknown", 0.0, "", {}

        # pre-boost state for tracking boost impact on winner
        pre_boost_scores = {}
        pre_boost_winner = None
        if self.use_boost_engine:
            pre_boost_scores = {intent: data['similarity'] for intent, data in intent_scores.items()}
            pre_boost_winner = max(pre_boost_scores.items(), key=lambda x: x[1])[0] if pre_boost_scores else None
            intent_scores = self.boost_engine.apply_all_boosts(query_words, intent_scores, processed_query)

        intent_name, similarity, pattern, breakdown = self._select_best_intent(intent_scores)

        if self.use_boost_engine and pre_boost_winner and intent_name in pre_boost_scores:
            pre_boost_winner_score = pre_boost_scores.get(pre_boost_winner, 0.0)
            current_winner_pre_boost_score = pre_boost_scores.get(intent_name, 0.0)

            breakdown['pre_boost_winner'] = pre_boost_winner
            breakdown['pre_boost_winner_score'] = pre_boost_winner_score
            breakdown['current_winner_pre_boost_score'] = current_winner_pre_boost_score
            breakdown['boost_delta'] = similarity - current_winner_pre_boost_score
            breakdown['winner_changed'] = (pre_boost_winner != intent_name)

        # Fallback: If but-clause was used but resulted in general/unknown, retry with full query
        if but_clause_applied and (intent_name in ["general", "unknown"] or similarity < self.algorithmic_threshold):

            full_query_words = set(query.split())
            full_intent_scores = self._evaluate_all_intents(
                query,
                full_query_words,
                early_exit_threshold=early_exit_threshold
            )

            if full_intent_scores:
                if self.use_boost_engine:
                    full_intent_scores = self.boost_engine.apply_all_boosts(full_query_words, full_intent_scores, query)

                full_intent_name, full_similarity, full_pattern, full_breakdown = self._select_best_intent(full_intent_scores)
                return full_intent_name, full_similarity, full_pattern, full_breakdown

        return intent_name, similarity, pattern, breakdown

    def recognize(self, query: str) -> AlgorithmicResult:
        """Main recognition method

        Args:
            query: User query

        Returns:
            AlgorithmicResult with intent classification
        """
        self.stats['total_queries'] += 1
        query = self.text_processor.normalize(query)
        intent_name, similarity, matched_pattern, breakdown = self.find_best_match(query)
        confidence_level = IntentRecognizerUtils.determine_confidence_level(similarity)

        method = 'none'
        boost_info = None

        if breakdown:
            # Determine base processing method
            if breakdown.get('keyword_similarity', 0) > 0.7:
                method = 'keyword'
            elif breakdown.get('levenshtein_similarity', 0) > 0.7:
                method = 'levenshtein'
            else:
                method = 'keyword + levenshtein'

            if self.use_boost_engine and 'boost_delta' in breakdown:
                boost_delta = breakdown['boost_delta']
                winner_changed = breakdown.get('winner_changed', False)

                if winner_changed or abs(boost_delta) > 0.15:
                    method = f"{method} + boost"
                    boost_info = f"{boost_delta:+.2f}"

        self.stats['intent_distribution'][intent_name] += 1
        self.stats['avg_confidence'].append(similarity)

        if boost_info:
            self.logger.info(
                f"{intent_name} ({similarity:.3f}, {confidence_level}, {method} [{boost_info}])"
            )
        else:
            self.logger.info(
                f"{intent_name} ({similarity:.3f}, {confidence_level}, {method})"
            )

        return AlgorithmicResult(
            intent=intent_name,
            confidence=similarity,
            confidence_level=confidence_level,
            matched_pattern=matched_pattern,
            processing_method=method,
            score_breakdown=breakdown
        )

    def get_statistics(self) -> Dict:
        """Get recognizer statistics"""
        return StatisticsHelper.build_stats_response(
            self.stats,
            average_confidence=StatisticsHelper.calculate_average(self.stats['avg_confidence']),
            avg_intents_evaluated_per_query=StatisticsHelper.calculate_average(self.stats['intents_evaluated']),
            avg_patterns_evaluated_per_query=StatisticsHelper.calculate_average(self.stats['patterns_evaluated'])
        )