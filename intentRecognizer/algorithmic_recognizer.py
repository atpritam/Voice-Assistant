"""
Algorithmic Intent Recognizer
Handles pattern matching, keyword analysis, and Levenshtein distance-based recognition
Optimized for ASR (Automatic Speech Recognition) input
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import Levenshtein

# Similarity calculation weights
KEYWORD_WEIGHT = 0.50
LEVENSHTEIN_WEIGHT = 0.50
EXACT_OVERLAP_WEIGHT = 0.7
SYNONYM_WEIGHT = 0.3

# Boost values
ORDER_ACTION_BOOST = 0.20
ORDER_DELIVERY_PENALTY = 0.15
NEGATIVE_SENTIMENT_BOOST = 0.20
PRICE_SIZE_BOOST = 0.25
TIME_LOCATION_BOOST = 0.20
ESCALATION_BOOST = 0.30

# Phrase matching bonuses
PHRASE_3_WORD_BONUS = 0.10
PHRASE_2_WORD_BONUS = 0.05

# Keyword bonuses
FIRST_KEYWORD_BONUS = 0.08
ADDITIONAL_KEYWORD_BONUS = 0.04
MAX_KEYWORD_BONUS = 0.20

# Length pre-filter thresholds
MAX_LENGTH_DIFF_LONG_STRINGS = 30
MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 2.5
LONG_STRING_THRESHOLD = 15

# Confidence levels
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
HIGH_SIMILARITY_EARLY_EXIT = 0.92

@dataclass
class SimilarityMetrics:
    """Container for all similarity calculation metrics"""
    keyword_similarity: float
    exact_overlap: float
    synonym_similarity: float
    levenshtein_similarity: float
    phrase_bonus: float
    keyword_bonus: float
    base_score: float
    final_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility"""
        return {
            'keyword_similarity': self.keyword_similarity,
            'exact_overlap': self.exact_overlap,
            'synonym_similarity': self.synonym_similarity,
            'levenshtein_similarity': self.levenshtein_similarity,
            'phrase_bonus': self.phrase_bonus,
            'keyword_bonus': self.keyword_bonus,
            'base_score': self.base_score,
            'final_score': self.final_score
        }


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

class TextNormalizer:
    """Handles all text normalization and preprocessing"""

    def __init__(self, filler_words: Set[str]):
        self.filler_words = filler_words

    def normalize(self, text: str) -> str:
        """
        Normalize text for comparison - optimized for ASR input

        Args:
            text: Input text string

        Returns:
            Normalized text string
        """
        if not text:
            return ""

        # Convert to lowercase and strip
        text = text.lower().strip()

        # Remove common punctuation
        punctuation = '!?.,;:\'"()[]{}/'
        text = text.translate(str.maketrans('', '', punctuation))

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def extract_words(self, text: str) -> List[str]:
        """Extract words from normalized text"""
        return text.split()

    def remove_filler_words(self, words: List[str]) -> List[str]:
        """Remove filler words common in speech"""
        return [w for w in words if w not in self.filler_words]

    def extract_filtered_words(self, text: str) -> List[str]:
        """Extract and filter words in one step"""
        normalized = self.normalize(text)
        words = self.extract_words(normalized)
        return self.remove_filler_words(words)


class SimilarityCalculator:
    """Handles all similarity metric calculations"""

    def __init__(self, text_normalizer: TextNormalizer, synonym_lookup: Dict):
        self.normalizer = text_normalizer
        self.synonym_lookup = synonym_lookup

    def passes_length_prefilter(self, query_norm: str, pattern_norm: str) -> bool:
        """Quick length-based filter to skip expensive calculations"""
        len_query = len(query_norm)
        len_pattern = len(pattern_norm)

        # Only apply filter for long strings
        if len_query <= LONG_STRING_THRESHOLD or len_pattern <= LONG_STRING_THRESHOLD:
            return True

        len_diff = abs(len_query - len_pattern)
        len_ratio = len_query / len_pattern if len_pattern > 0 else 0

        # Filter out if too different
        if len_diff > MAX_LENGTH_DIFF_LONG_STRINGS:
            return False

        if len_ratio < MIN_LENGTH_RATIO or len_ratio > MAX_LENGTH_RATIO:
            return False

        return True

    def calculate_keyword_similarity(self, query_set: Set[str], pattern_set: Set[str]) -> Tuple[float, float, float]:
        """
        Calculate keyword-based similarity metrics

        Returns:
            Tuple of (combined_keyword_similarity, exact_similarity, synonym_similarity)
        """
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

        # Combined keyword score
        keyword_similarity = EXACT_OVERLAP_WEIGHT * exact_similarity + SYNONYM_WEIGHT * synonym_similarity

        return keyword_similarity, exact_similarity, synonym_similarity

    def _expand_with_synonyms(self, words: Set[str]) -> Set[str]:
        """Expand word set with known synonyms"""
        expanded = set(words)
        for word in words:
            if word in self.synonym_lookup:
                expanded.update(self.synonym_lookup[word])
        return expanded

    def calculate_levenshtein_similarity(self, query_norm: str, pattern_norm: str) -> float:
        """Calculate normalized Levenshtein distance"""
        return Levenshtein.ratio(query_norm, pattern_norm)

    def calculate_phrase_bonus(self, query_words: List[str], pattern_words: List[str]) -> float:
        """
        Calculate bonus for matching consecutive word phrases

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
            return PHRASE_3_WORD_BONUS
        elif max_phrase_length == 2:
            return PHRASE_2_WORD_BONUS

        return 0.0

    def calculate_keyword_bonus(self, query_set: Set[str], intent_name: Optional[str],
                               intent_critical_keywords: Dict) -> float:
        """
        Calculate bonus for matching critical intent keywords

        Returns:
            Bonus score (0.0 to 0.20)
        """
        max_bonus = 0.0

        # If we know the intent, check directly
        if intent_name and intent_name in intent_critical_keywords:
            critical_keywords = intent_critical_keywords[intent_name]
            query_critical = query_set.intersection(critical_keywords)

            if query_critical:
                num_matches = len(query_critical)
                bonus = min(MAX_KEYWORD_BONUS,
                           FIRST_KEYWORD_BONUS + (num_matches - 1) * ADDITIONAL_KEYWORD_BONUS)
                return bonus

        # If intent unknown, find best matching category
        else:
            for category_name, critical_keywords in intent_critical_keywords.items():
                query_critical = query_set.intersection(critical_keywords)
                if query_critical:
                    num_matches = len(query_critical)
                    bonus = min(MAX_KEYWORD_BONUS,
                               FIRST_KEYWORD_BONUS + (num_matches - 1) * ADDITIONAL_KEYWORD_BONUS)
                    max_bonus = max(max_bonus, bonus)

        return max_bonus

    def calculate_similarity(self, query: str, pattern: str, intent_name: Optional[str],
                           pattern_norm: Optional[str], intent_critical_keywords: Dict) -> Tuple[float, SimilarityMetrics]:
        """
        Main similarity calculation coordinator

        Returns:
            Tuple of (final_score, metrics_object)
        """
        # Normalize texts
        query_norm = self.normalizer.normalize(query)
        if pattern_norm is None:
            pattern_norm = self.normalizer.normalize(pattern)

        if not query_norm or not pattern_norm:
            return 0.0, self._empty_metrics()

        # Length pre-filter
        if not self.passes_length_prefilter(query_norm, pattern_norm):
            return 0.0, self._empty_metrics()

        # Extract and filter words
        query_words = self.normalizer.extract_filtered_words(query)
        pattern_words = self.normalizer.extract_filtered_words(pattern)

        if not query_words or not pattern_words:
            return 0.0, self._empty_metrics()

        query_set = set(query_words)
        pattern_set = set(pattern_words)

        # Calculate all metrics
        keyword_sim, exact_sim, synonym_sim = self.calculate_keyword_similarity(query_set, pattern_set)
        levenshtein_sim = self.calculate_levenshtein_similarity(query_norm, pattern_norm)
        phrase_bonus = self.calculate_phrase_bonus(query_words, pattern_words)
        keyword_bonus = self.calculate_keyword_bonus(query_set, intent_name, intent_critical_keywords)

        # Combine scores
        base_score = KEYWORD_WEIGHT * keyword_sim + LEVENSHTEIN_WEIGHT * levenshtein_sim
        final_score = min(1.0, base_score + phrase_bonus + keyword_bonus)

        # Create metrics object
        metrics = SimilarityMetrics(
            keyword_similarity=keyword_sim,
            exact_overlap=exact_sim,
            synonym_similarity=synonym_sim,
            levenshtein_similarity=levenshtein_sim,
            phrase_bonus=phrase_bonus,
            keyword_bonus=keyword_bonus,
            base_score=base_score,
            final_score=final_score
        )

        return final_score, metrics

    def _empty_metrics(self) -> SimilarityMetrics:
        """Return empty metrics object"""
        return SimilarityMetrics(
            keyword_similarity=0.0,
            exact_overlap=0.0,
            synonym_similarity=0.0,
            levenshtein_similarity=0.0,
            phrase_bonus=0.0,
            keyword_bonus=0.0,
            base_score=0.0,
            final_score=0.0
        )


class BoostRuleEngine:
    """Applies contextual boost rules to intent scores"""

    def __init__(self, intent_critical_keywords: Dict, enable_logging: bool = False):
        self.intent_critical_keywords = intent_critical_keywords
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)

    def apply_all_boosts(self, query_words: Set[str], intent_scores: Dict) -> Dict:
        """Apply all boost rules to intent scores"""
        self._apply_order_action_boost(query_words, intent_scores)
        self._apply_negative_sentiment_boost(query_words, intent_scores)
        self._apply_price_size_boost(query_words, intent_scores)
        self._apply_time_location_boost(query_words, intent_scores)
        self._apply_escalation_boost(query_words, intent_scores)
        return intent_scores

    def _apply_order_action_boost(self, query_words: Set[str], intent_scores: Dict):
        """RULE 1 & 2: Order action verb handling"""
        has_order_action = bool(query_words.intersection(
            self.intent_critical_keywords.get('order', set())
        ))
        has_order_keyword = 'order' in query_words

        if has_order_action and has_order_keyword:
            # Boost order intent
            if 'order' in intent_scores:
                original = intent_scores['order']['similarity']
                intent_scores['order']['similarity'] = min(1.0, original + ORDER_ACTION_BOOST)

                if self.enable_logging:
                    self.logger.debug(
                        f"Order boost: {original:.3f} → {intent_scores['order']['similarity']:.3f}"
                    )

            # Penalize delivery unless tracking keywords present
            has_tracking = bool(query_words.intersection({'track', 'status', 'where', 'eta'}))
            if not has_tracking and 'delivery' in intent_scores:
                original = intent_scores['delivery']['similarity']
                intent_scores['delivery']['similarity'] = max(0.0, original - ORDER_DELIVERY_PENALTY)

                if self.enable_logging:
                    self.logger.debug(
                        f"Delivery penalty: {original:.3f} → {intent_scores['delivery']['similarity']:.3f}"
                    )

    def _apply_negative_sentiment_boost(self, query_words: Set[str], intent_scores: Dict):
        """RULE 3: Negative sentiment for complaints"""
        negative_words = {
            'wrong', 'bad', 'terrible', 'horrible', 'disappointed', 'complain',
            'unhappy', 'angry', 'upset', 'disgusted', 'awful', 'missing', 'cold', 'late', 'issue'
        }

        if query_words.intersection(negative_words) and 'complaint' in intent_scores:
            original = intent_scores['complaint']['similarity']
            intent_scores['complaint']['similarity'] = min(1.0, original + NEGATIVE_SENTIMENT_BOOST)

            if self.enable_logging:
                self.logger.debug(
                    f"Negative sentiment boost: {original:.3f} → {intent_scores['complaint']['similarity']:.3f}"
                )

    def _apply_price_size_boost(self, query_words: Set[str], intent_scores: Dict):
        """RULE 4: Price + size indicates menu inquiry"""
        has_price = bool(query_words.intersection({'price', 'prices', 'cost', 'much'}))
        has_size = bool(query_words.intersection({'small', 'medium', 'large'}))

        if has_price and has_size and 'menu_inquiry' in intent_scores:
            original = intent_scores['menu_inquiry']['similarity']
            intent_scores['menu_inquiry']['similarity'] = min(1.0, original + PRICE_SIZE_BOOST)

            if self.enable_logging:
                self.logger.debug(
                    f"Price+size boost: {original:.3f} → {intent_scores['menu_inquiry']['similarity']:.3f}"
                )

    def _apply_time_location_boost(self, query_words: Set[str], intent_scores: Dict):
        """RULE 5: Time/location questions boost hours_location"""
        time_location_context = {
            'when', 'what time', 'how long', 'until when', 'from when',
            'where', 'which', 'what address', 'how far'
        }
        hours_keywords = {'open', 'close', 'hours', 'location', 'address', 'store'}

        has_question = bool(query_words.intersection(time_location_context))
        has_hours = bool(query_words.intersection(hours_keywords))

        if has_question and has_hours and 'hours_location' in intent_scores:
            original = intent_scores['hours_location']['similarity']
            intent_scores['hours_location']['similarity'] = min(1.0, original + TIME_LOCATION_BOOST)

            if self.enable_logging:
                self.logger.debug(
                    f"Time/location boost: {original:.3f} → {intent_scores['hours_location']['similarity']:.3f}"
                )

    def _apply_escalation_boost(self, query_words: Set[str], intent_scores: Dict):
        """RULE 6: Escalation keywords strongly indicate complaint"""
        escalation_keywords = {
            'refund', 'manager', 'supervisor', 'speak to', 'talk to',
            'compensation', 'money back', 'unacceptable', 'ridiculous'
        }

        query_norm = ' '.join(query_words)
        has_escalation = any(keyword in query_norm for keyword in escalation_keywords)

        if has_escalation and 'complaint' in intent_scores:
            original = intent_scores['complaint']['similarity']
            intent_scores['complaint']['similarity'] = min(1.0, original + ESCALATION_BOOST)

            if self.enable_logging:
                self.logger.debug(
                    f"Escalation boost: {original:.3f} → {intent_scores['complaint']['similarity']:.3f}"
                )


# MAIN RECOGNIZER CLASS

class AlgorithmicRecognizer:
    """Pattern-based intent recognition using keywords and string similarity"""

    def __init__(self, patterns_file: str = None, enable_logging: bool = False,
                 min_confidence: float = 0.5):
        """Initialize the algorithmic recognizer"""
        # Configuration
        if patterns_file is None:
            utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
            patterns_file = os.path.join(utils_dir, 'intent_patterns.json')

        self.patterns_file = patterns_file
        self.min_confidence = min_confidence
        self.enable_logging = enable_logging

        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)

        # Load patterns
        self.patterns = self._load_patterns()

        # Initialize linguistic resources
        self._initialize_linguistic_resources()

        # Create helper components
        self.text_normalizer = TextNormalizer(self.filler_words)
        self.similarity_calculator = SimilarityCalculator(
            self.text_normalizer,
            self._synonym_lookup
        )
        self.boost_engine = BoostRuleEngine(
            self.intent_critical_keywords,
            enable_logging
        )

        # Preprocess patterns
        self._intent_keywords_cache = {}
        self._normalized_patterns_cache = {}
        self._preprocess_patterns()

        # Statistics
        self.stats = {
            'total_queries': 0,
            'intent_distribution': {},
            'avg_confidence': []
        }

    def _load_patterns(self) -> Dict:
        """Load intent patterns from JSON file"""
        try:
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('intents', data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            if self.enable_logging:
                self.logger.error(f"Error loading patterns: {e}")
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
        # Synonym groups
        self.synonyms = {
            'order': {'order', 'buy', 'purchase', 'get', 'want', 'place'},
            'complaint': {'complaint', 'complain', 'wrong', 'issue', 'problem',
                          'bad', 'terrible', 'horrible', 'disappointed', 'unhappy', 'refund'},
            'delivery': {'delivery', 'deliver', 'delivered', 'delivering', 'shipment'},
            'menu': {'menu', 'have', 'sell', 'offer', 'available', 'serve'},
            'hours': {'hours', 'open', 'close', 'time', 'opening', 'closing', 'schedule'},
            'location': {'location', 'address', 'where', 'place', 'store', 'restaurant', 'shop'},
            'track': {'track', 'status', 'where', 'find', 'locate'},
            'price': {'price', 'cost', 'much', 'expensive', 'cheap', 'charge', 'fee', 'prices'},
            'specialty': {'specialty', 'special', 'signature', 'featured', 'premium'}
        }

        # Build synonym lookup
        self._synonym_lookup = {}
        for syn_group in self.synonyms.values():
            for word in syn_group:
                self._synonym_lookup[word] = syn_group

        # Filler words
        self.filler_words = {
            'um', 'uh', 'umm', 'uhh', 'like', 'you know', 'basically', 'actually',
            'literally', 'just', 'really', 'very', 'so', 'well', 'i mean',
            'kind of', 'sort of', 'please'
        }

        # Critical keywords
        self.intent_critical_keywords = {
            'order': {'order', 'buy', 'purchase', 'want', 'get', 'place', 'make'},
            'complaint': {'wrong', 'cold', 'late', 'missing', 'burnt', 'complaint',
                          'problem', 'issue', 'refund', 'manager', 'terrible', 'horrible', 'bad'},
            'hours_location': {'hours', 'open', 'close', 'address', 'location',
                               'where', 'when', 'time'},
            'menu_inquiry': {'menu', 'toppings', 'sizes', 'price', 'cost',
                             'have', 'options', 'special', 'deal', 'signature'},
            'delivery': {'track', 'status', 'eta', 'arrive', 'fee', 'charge'},
            'general': {'hello', 'hi', 'hey', 'thanks', 'thank', 'bye',
                        'goodbye', 'help', 'assist'}
        }

    def _preprocess_patterns(self):
        """Preprocess all patterns for faster matching"""
        for intent_name, intent_data in self.patterns.items():
            if intent_name == "unknown":
                continue

            keywords = set()
            normalized_patterns = []

            for pattern in intent_data.get("patterns", []):
                normalized = self.text_normalizer.normalize(pattern)
                words = normalized.split()
                keywords.update(words)
                normalized_patterns.append(normalized)

            self._intent_keywords_cache[intent_name] = keywords
            self._normalized_patterns_cache[intent_name] = normalized_patterns

    def _evaluate_single_intent(self, intent_name: str, intent_data: Dict,
                               query: str, query_words: Set[str]) -> Optional[IntentEvaluation]:
        """Evaluate query against one intent's patterns"""
        # Quick keyword pre-filter
        if intent_name in self._intent_keywords_cache:
            intent_keywords = self._intent_keywords_cache[intent_name]
            if not query_words.intersection(intent_keywords):
                return None

        patterns = intent_data.get("patterns", [])
        normalized_patterns = self._normalized_patterns_cache.get(intent_name, [])

        max_similarity = 0.0
        best_pattern = ""
        best_breakdown = {}

        # Find best matching pattern
        for i, pattern in enumerate(patterns):
            pattern_norm = normalized_patterns[i] if i < len(normalized_patterns) else None
            similarity, metrics = self.similarity_calculator.calculate_similarity(
                query, pattern, intent_name, pattern_norm, self.intent_critical_keywords
            )

            if similarity > max_similarity:
                max_similarity = similarity
                best_pattern = pattern
                best_breakdown = metrics.to_dict()

            # Early exit for high confidence
            if similarity > HIGH_SIMILARITY_EARLY_EXIT:
                break

        return IntentEvaluation(max_similarity, best_pattern, best_breakdown)

    def _evaluate_all_intents(self, query: str, query_words: Set[str]) -> Dict:
        """Evaluate query against all intent patterns"""
        intent_scores = {}

        for intent_name, intent_data in self.patterns.items():
            if intent_name == "unknown":
                continue

            evaluation = self._evaluate_single_intent(intent_name, intent_data, query, query_words)
            if evaluation and evaluation.similarity > 0:
                intent_scores[intent_name] = {
                    'similarity': evaluation.similarity,
                    'pattern': evaluation.pattern,
                    'breakdown': evaluation.breakdown
                }

        return intent_scores

    def _select_best_intent(self, intent_scores: Dict) -> Tuple[str, float, str, Dict]:
        """Select best intent from scores and apply threshold"""
        if not intent_scores:
            return "unknown", 0.0, "", {}

        best_intent = max(intent_scores.items(), key=lambda x: x[1]['similarity'])
        best_intent_name = best_intent[0]
        best_similarity = best_intent[1]['similarity']
        best_pattern = best_intent[1]['pattern']
        best_breakdown = best_intent[1]['breakdown']

        # Check threshold
        threshold = self.patterns.get(best_intent_name, {}).get(
            'similarity_threshold', self.min_confidence
        )

        if best_similarity < threshold:
            return "unknown", best_similarity, best_pattern, best_breakdown

        return best_intent_name, best_similarity, best_pattern, best_breakdown

    def find_best_match(self, query: str) -> Tuple[str, float, str, Dict]:
        """Find the best matching intent for a given query"""
        if not query or not self.patterns:
            return "unknown", 0.0, "", {}

        # Preprocess query
        query_words = set(self.text_normalizer.extract_filtered_words(query))
        if not query_words:
            return "unknown", 0.0, "", {}

        # Evaluate all intents
        intent_scores = self._evaluate_all_intents(query, query_words)
        if not intent_scores:
            return "unknown", 0.0, "", {}

        # Apply boost rules
        intent_scores = self.boost_engine.apply_all_boosts(query_words, intent_scores)

        # Select best intent
        return self._select_best_intent(intent_scores)

    def recognize(self, query: str) -> AlgorithmicResult:
        """Main recognition method"""
        self.stats['total_queries'] += 1

        # Find best match
        intent_name, similarity, matched_pattern, breakdown = self.find_best_match(query)

        # Determine confidence level
        if similarity >= HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = 'high'
        elif similarity >= MEDIUM_CONFIDENCE_THRESHOLD:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'

        # Determine processing method
        if breakdown:
            if breakdown['keyword_similarity'] > 0.7:
                method = 'keyword'
            elif breakdown['levenshtein_similarity'] > 0.7:
                method = 'levenshtein'
            else:
                method = 'keyword + levenshtein'
        else:
            method = 'none'

        # Update statistics
        self.stats['intent_distribution'][intent_name] = \
            self.stats['intent_distribution'].get(intent_name, 0) + 1
        self.stats['avg_confidence'].append(similarity)

        # Logging
        if self.enable_logging:
            self.logger.info(
                f"[ALGORITHMIC] Intent: {intent_name} "
                f"(confidence: {similarity:.3f}, level: {confidence_level}, method: {method})"
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
        avg_conf = (sum(self.stats['avg_confidence']) / len(self.stats['avg_confidence'])
                   if self.stats['avg_confidence'] else 0.0)

        return {
            'total_queries_processed': self.stats['total_queries'],
            'intent_distribution': self.stats['intent_distribution'],
            'average_confidence': avg_conf
        }