"""
Intent Recognition System - Algorithm Layer Only
Optimized for ASR (Automatic Speech Recognition) input
Uses keyword matching, Levenshtein distance, and intelligent scoring
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import Levenshtein


@dataclass
class RecognitionResult:
    """Structured result from intent recognition"""
    intent: str
    confidence: float
    confidence_level: str  # 'high', 'medium', 'low'
    matched_pattern: str
    processing_method: str  # 'keyword', 'levenshtein', 'combined'


class IntentRecognizer:

    def __init__(
            self,
            patterns_file: str = None,
            enable_logging: bool = False,
            min_confidence: float = 0.5
    ):
        """
        Initialize the intent recognizer

        Args:
            patterns_file: Path to JSON file with intent patterns
            enable_logging: Enable detailed logging for analysis
            min_confidence: Minimum confidence threshold (default: 0.5)
        """
        # Paths
        if patterns_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            patterns_file = os.path.join(current_dir, 'intent_patterns.json')

        self.patterns_file = patterns_file
        self.min_confidence = min_confidence

        # Setup logging
        self.enable_logging = enable_logging
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)

        self.patterns = self._load_patterns()

        # Performance optimization caches
        self._intent_keywords_cache = {}

        # Initialize linguistic resources
        self._initialize_linguistic_resources()
        self._preprocess_patterns()

        # Statistics tracking
        self.stats = {
            'total_queries': 0,
            'intent_distribution': {},
            'avg_confidence': [],
        }

    def _load_patterns(self) -> Dict:
        """Load intent patterns from JSON file"""
        try:
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Support both formats: direct intents or nested structure
            if 'intents' in data:
                return data['intents']
            return data

        except FileNotFoundError:
            if self.enable_logging:
                self.logger.error(f"Patterns file not found: {self.patterns_file}")
            return self._get_default_patterns()

        except json.JSONDecodeError as e:
            if self.enable_logging:
                self.logger.error(f"Error parsing patterns file: {e}")
            return self._get_default_patterns()

    def _get_default_patterns(self) -> Dict:
        """Fallback patterns if file not found"""
        return {
            "unknown": {
                "patterns": [],
                "similarity_threshold": 0.0,
                "description": "Fallback for unrecognized intents"
            }
        }

    def _initialize_linguistic_resources(self):
        """Initialize synonyms, filler words, and keyword mappings"""

        # Synonym groups for semantic expansion
        self.synonyms = {
            'order': {'order', 'buy', 'purchase', 'get', 'want', 'place'},
            'complaint': {'complaint', 'complain', 'wrong', 'issue', 'problem',
                          'bad', 'terrible', 'horrible', 'disappointed', 'unhappy', 'refund'},
            'delivery': {'delivery', 'deliver', 'delivered', 'delivering', 'shipment'},
            'menu': {'menu', 'have', 'sell', 'offer', 'available', 'serve'},
            'hours': {'hours', 'open', 'close', 'time', 'opening', 'closing', 'schedule'},
            'location': {'location', 'address', 'where', 'place', 'store', 'restaurant', 'shop'},
            'track': {'track', 'status', 'where', 'find', 'locate'},
            'price': {'price', 'cost', 'much', 'expensive', 'cheap', 'charge', 'fee'}
        }

        # Common filler words in speech (to be removed)
        self.filler_words = {
            'um', 'uh', 'umm', 'uhh', 'like', 'you know', 'basically', 'actually',
            'literally', 'just', 'really', 'very', 'so', 'well', 'i mean',
            'kind of', 'sort of', 'please'
        }

        # Critical keywords that strongly indicate specific intents
        self.intent_critical_keywords = {
            'order': {'order', 'buy', 'purchase', 'want', 'get', 'pizza', 'large', 'medium', 'small'},
            'complaint': {'wrong', 'cold', 'late', 'missing', 'burnt', 'complaint',
                          'problem', 'issue', 'refund', 'manager', 'terrible', 'horrible', 'bad'},
            'hours_location': {'hours', 'open', 'close', 'address', 'location',
                               'where', 'when', 'time'},
            'menu_inquiry': {'menu', 'toppings', 'sizes', 'price', 'cost',
                             'have', 'options', 'special', 'deal'},
            'delivery': {'delivery', 'deliver', 'track', 'status', 'eta',
                         'arrive', 'fee', 'charge', 'time'},
            'general': {'hello', 'hi', 'hey', 'thanks', 'thank', 'bye',
                        'goodbye', 'help', 'assist'}
        }

    def _preprocess_patterns(self):
        """Preprocess all patterns for faster matching"""
        for intent_name, intent_data in self.patterns.items():
            if intent_name == "unknown":
                continue

            # Extract keywords from all patterns for this intent
            keywords = set()

            for pattern in intent_data.get("patterns", []):
                normalized = self._normalize_text(pattern)
                words = normalized.split()
                keywords.update(words)

            self._intent_keywords_cache[intent_name] = keywords

        if self.enable_logging:
            self.logger.info(f"Preprocessed {len(self.patterns)} intent categories")

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison - optimized for ASR input

        Args:
            text: Input text string

        Returns:
            Normalized text string
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower().strip()

        # Remove common punctuation
        punctuation = '!?.,;:\'"()[]{}/'
        text = text.translate(str.maketrans('', '', punctuation))

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def _remove_filler_words(self, words: List[str]) -> List[str]:
        """
        Remove filler words common in speech

        Args:
            words: List of words

        Returns:
            Filtered list of words
        """
        return [w for w in words if w not in self.filler_words]

    def _expand_with_synonyms(self, words: Set[str]) -> Set[str]:
        """
        Expand word set with known synonyms

        Args:
            words: Set of words to expand

        Returns:
            Expanded set including synonyms
        """
        expanded = set(words)
        for word in words:
            for syn_group in self.synonyms.values():
                if word in syn_group:
                    expanded.update(syn_group)
                    break
        return expanded

    def _calculate_similarity(self, query: str, pattern: str) -> Tuple[float, Dict]:
        """
        Calculate multi-metric similarity score between query and pattern

        Args:
            query: User input string (from ASR)
            pattern: Pattern string to compare against

        Returns:
            Tuple of (final_similarity_score, score_breakdown_dict)
        """
        if not query or not pattern:
            return 0.0, {}

        # Normalize both strings
        query_norm = self._normalize_text(query)
        pattern_norm = self._normalize_text(pattern)

        if not query_norm or not pattern_norm:
            return 0.0, {}

        # Extract and filter words
        query_words = query_norm.split()
        pattern_words = pattern_norm.split()

        query_words_filtered = self._remove_filler_words(query_words)
        pattern_words_filtered = self._remove_filler_words(pattern_words)

        if not query_words_filtered or not pattern_words_filtered:
            return 0.0, {}

        # Convert to sets
        query_set = set(query_words_filtered)
        pattern_set = set(pattern_words_filtered)

        # METRIC 1: KEYWORD SIMILARITY
        # Exact word overlap
        exact_overlap = len(query_set.intersection(pattern_set))
        union_size = len(query_set.union(pattern_set))
        exact_similarity = exact_overlap / union_size if union_size > 0 else 0.0

        # Synonym-expanded overlap
        query_expanded = self._expand_with_synonyms(query_set)
        pattern_expanded = self._expand_with_synonyms(pattern_set)
        synonym_overlap = len(query_expanded.intersection(pattern_expanded))
        expanded_union_size = len(query_expanded.union(pattern_expanded))
        synonym_similarity = synonym_overlap / expanded_union_size if expanded_union_size > 0 else 0.0

        # Combined keyword score (70% exact, 30% synonyms)
        keyword_similarity = 0.7 * exact_similarity + 0.3 * synonym_similarity

        # METRIC 2: LEVENSHTEIN DISTANCE
        # Normalized Levenshtein ratio (0 to 1)
        levenshtein_similarity = Levenshtein.ratio(query_norm, pattern_norm)

        # METRIC 3: PHRASE MATCHING BONUS
        phrase_bonus = self._calculate_phrase_bonus(query_words_filtered, pattern_words_filtered)

        # METRIC 4: CRITICAL KEYWORD BONUS
        keyword_bonus = self._calculate_keyword_bonus(query_set, pattern_set)

        # FINAL WEIGHTED COMBINATION
        # Base score: 100% weighting
        base_similarity = (
                0.70 * keyword_similarity +  # Primary: semantic keyword matching
                0.30 * levenshtein_similarity  # Secondary: overall string similarity
        )

        # Add bonuses (capped at 1.0)
        final_similarity = min(1.0, base_similarity + phrase_bonus + keyword_bonus)

        breakdown = {
            'keyword_similarity': keyword_similarity,
            'exact_overlap': exact_similarity,
            'synonym_similarity': synonym_similarity,
            'levenshtein_similarity': levenshtein_similarity,
            'phrase_bonus': phrase_bonus,
            'keyword_bonus': keyword_bonus,
            'base_score': base_similarity,
            'final_score': final_similarity
        }

        return final_similarity, breakdown

    def _calculate_phrase_bonus(self, query_words: List[str], pattern_words: List[str]) -> float:
        """
        Calculate bonus for matching consecutive word phrases
        Preserves word order which is important in natural language

        Args:
            query_words: Filtered query words
            pattern_words: Filtered pattern words

        Returns:
            Bonus score (0.0 to 0.10)
        """
        if len(pattern_words) < 2:
            return 0.0

        max_phrase_length = 0
        query_text = ' '.join(query_words)

        # Check for 2-word and 3-word phrase matches
        for n in [2, 3]:
            if len(pattern_words) < n:
                continue

            for i in range(len(pattern_words) - n + 1):
                phrase = ' '.join(pattern_words[i:i + n])

                if phrase in query_text:
                    max_phrase_length = max(max_phrase_length, n)

        # Return scaled bonus
        if max_phrase_length >= 3:
            return 0.10  # 10% bonus for 3+ word phrases
        elif max_phrase_length == 2:
            return 0.05  # 5% bonus for 2-word phrases

        return 0.0

    def _calculate_keyword_bonus(self, query_set: Set[str], pattern_set: Set[str]) -> float:
        """
        Calculate bonus for matching critical intent keywords
        These are high-confidence indicators of specific intents

        Args:
            query_set: Set of query words
            pattern_set: Set of pattern words

        Returns:
            Bonus score (0.0 to 0.10)
        """
        max_bonus = 0.0

        for intent_name, critical_keywords in self.intent_critical_keywords.items():
            # Find critical keywords in both query and pattern
            query_critical = query_set.intersection(critical_keywords)
            pattern_critical = pattern_set.intersection(critical_keywords)

            # If both have critical keywords from same intent category
            if query_critical and pattern_critical:
                overlap = len(query_critical.intersection(pattern_critical))
                if overlap > 0:
                    # More matches = higher bonus (max 10%)
                    bonus = min(0.10, 0.03 * overlap)
                    max_bonus = max(max_bonus, bonus)

        return max_bonus

    def find_best_match(self, query: str) -> Tuple[str, float, str, Dict]:
        """
        Find the best matching intent for a given query

        Args:
            query: User input string (from ASR)

        Returns:
            Tuple of (intent_name, similarity_score, matched_pattern, score_breakdown)
        """
        if not query or not self.patterns:
            return "unknown", 0.0, "", {}

        # Normalize query once
        query_norm = self._normalize_text(query)
        query_words = set(self._remove_filler_words(query_norm.split()))

        if not query_words:
            return "unknown", 0.0, "", {}

        best_intent = "unknown"
        best_similarity = 0.0
        best_pattern = ""
        best_breakdown = {}

        # Evaluate each intent category
        for intent_name, intent_data in self.patterns.items():
            if intent_name == "unknown":
                continue

            # Quick pre-filter: check for any keyword overlap
            if intent_name in self._intent_keywords_cache:
                intent_keywords = self._intent_keywords_cache[intent_name]
                keyword_overlap = len(query_words.intersection(intent_keywords))

                # Skip if no keyword overlap (performance optimization)
                if keyword_overlap == 0:
                    continue

            patterns = intent_data.get("patterns", [])

            # Find best matching pattern in this category
            max_similarity_in_category = 0.0
            best_pattern_in_category = ""
            best_breakdown_in_category = {}

            for pattern in patterns:
                similarity, breakdown = self.calculate_similarity(query, pattern)

                if similarity > max_similarity_in_category:
                    max_similarity_in_category = similarity
                    best_pattern_in_category = pattern
                    best_breakdown_in_category = breakdown

                # Early exit for very high confidence
                if similarity > 0.95:
                    break

            # Update global best if this category scored higher
            if max_similarity_in_category > best_similarity:
                best_similarity = max_similarity_in_category
                best_intent = intent_name
                best_pattern = best_pattern_in_category
                best_breakdown = best_breakdown_in_category

        # Check if best match meets minimum threshold
        threshold = self.patterns.get(best_intent, {}).get('similarity_threshold', self.min_confidence)

        if best_similarity < threshold:
            return "unknown", best_similarity, best_pattern, best_breakdown

        return best_intent, best_similarity, best_pattern, best_breakdown

    def recognize_intent(self, query: str) -> RecognitionResult:
        """
        Main method: Recognize intent from user query

        Args:
            query: User input string (from ASR)

        Returns:
            RecognitionResult object with intent information
        """
        # Update statistics
        self.stats['total_queries'] += 1

        # Find best match
        intent_name, similarity, matched_pattern, breakdown = self.find_best_match(query)

        # Determine confidence level
        if similarity >= 0.8:
            confidence_level = 'high'
        elif similarity >= 0.6:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'

        # Determine primary matching method
        if breakdown:
            if breakdown['keyword_similarity'] > 0.7:
                method = 'keyword'
            elif breakdown['levenshtein_similarity'] > 0.7:
                method = 'levenshtein'
            else:
                method = 'combined'
        else:
            method = 'none'

        # Update statistics
        self.stats['intent_distribution'][intent_name] = \
            self.stats['intent_distribution'].get(intent_name, 0) + 1
        self.stats['avg_confidence'].append(similarity)

        # Create result
        result = RecognitionResult(
            intent=intent_name,
            confidence=similarity,
            confidence_level=confidence_level,
            matched_pattern=matched_pattern,
            processing_method=method
        )

        # Logging (if enabled)
        if self.enable_logging:
            self.logger.info(
                f"Query: '{query}' → Intent: {intent_name} "
                f"(confidence: {similarity:.3f}, level: {confidence_level}, method: {method})"
            )

        return result

    def evaluate(self, test_data: List[Tuple[str, str]]) -> Dict:
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
                'correct': is_correct
            })

        accuracy = correct / total if total > 0 else 0.0

        # Calculate metrics by confidence level
        high_conf = [r for r in results if r['confidence'] >= 0.8]
        medium_conf = [r for r in results if 0.6 <= r['confidence'] < 0.8]
        low_conf = [r for r in results if r['confidence'] < 0.6]

        return {
            'accuracy': accuracy,
            'total_queries': total,
            'correct': correct,
            'incorrect': total - correct,
            'high_confidence_count': len(high_conf),
            'medium_confidence_count': len(medium_conf),
            'low_confidence_count': len(low_conf),
            'high_confidence_accuracy': sum(r['correct'] for r in high_conf) / len(high_conf) if high_conf else 0,
            'detailed_results': results
        }

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        avg_conf = sum(self.stats['avg_confidence']) / len(self.stats['avg_confidence']) \
            if self.stats['avg_confidence'] else 0.0

        return {
            'total_queries_processed': self.stats['total_queries'],
            'intent_distribution': self.stats['intent_distribution'],
            'average_confidence': avg_conf,
        }

    def generate_response(self, intent_info: RecognitionResult, message: str) -> str:
        """
        Generate a response based on the recognized intent.
        Simply returns the intent name.

        Args:
            intent_info: RecognitionResult object from recognize_intent
            message: The original user message (unused but kept for compatibility)

        Returns:
            str: The recognized intent name
        """
        return intent_info.intent