"""
Optimized Algorithmic Intent Recognizer
Handles pattern matching, keyword analysis, and Levenshtein distance-based recognition
Optimized for ASR (Automatic Speech Recognition) input with multi-stage filtering
Enhanced with optional TF-IDF weighted inverted index
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import math
import Levenshtein

from .intent_recognizer import IntentRecognizerUtils
from .boostEngine import BoostRuleEngine

# SIMILARITY CALCULATION WEIGHTS
KEYWORD_WEIGHT = 0.50
LEVENSHTEIN_WEIGHT = 0.50
EXACT_OVERLAP_WEIGHT = 0.7
SYNONYM_WEIGHT = 0.3

# PHRASE MATCHING BONUSES
PHRASE_3_WORD_BONUS = 0.10
PHRASE_2_WORD_BONUS = 0.05

# KEYWORD BONUSES
FIRST_KEYWORD_BONUS = 0.08
ADDITIONAL_KEYWORD_BONUS = 0.04
MAX_KEYWORD_BONUS = 0.20

# LENGTH FILTERING THRESHOLDS
MAX_LENGTH_DIFF_LONG_STRINGS = 30
MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 2.5
LONG_STRING_THRESHOLD = 15

# EARLY EXIT THRESHOLDS
HIGH_SIMILARITY_EARLY_EXIT = 0.92

# OPTIMIZATION FLAGS
LEVENSHTEIN_SKIP_THRESHOLD = 0.17
LENGTH_DIFF_FILTER_ENABLED = True
INVERTED_INDEX_ENABLED = True

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
        return {k: v for k, v in self.__dict__.items()}


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


class LinguisticResourceLoader:
    """Loads linguistic resources from external JSON file"""

    @staticmethod
    def load_resources(resource_file: str = None) -> Dict:
        """Load linguistic resources from JSON file"""
        if resource_file is None:
            utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
            resource_file = os.path.join(utils_dir, 'linguistic_resources.json')

        try:
            with open(resource_file, 'r', encoding='utf-8') as f:
                resources = json.load(f)

            return {
                'synonyms': {k: set(v) for k, v in resources.get('synonyms', {}).items()},
                'filler_words': set(resources.get('filler_words', [])),
                'intent_critical_keywords': {k: set(v) for k, v in resources.get('intent_critical_keywords', {}).items()}
            }
        except FileNotFoundError:
            raise FileNotFoundError(f"Linguistic resources file not found: {resource_file}\nExpected file at: utils/linguistic_resources.json")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in linguistic resources file: {e}")

    @staticmethod
    def build_synonym_lookup(synonyms: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Build reverse lookup for efficient synonym matching"""
        lookup = {}
        for syn_group in synonyms.values():
            for word in syn_group:
                lookup[word] = syn_group
        return lookup


class InvertedIndex:
    """Inverted index with TF-IDF weighting for fast intent candidate selection"""

    def __init__(self, patterns: Dict, text_normalizer):
        self.normalizer = text_normalizer
        self.index = defaultdict(set)
        self.intent_keyword_scores = {}
        self.term_document_freq = defaultdict(int)
        self.idf_weights = {}
        self.total_intents = 0

        self._build_index(patterns)
        self._calculate_idf_weights()

    def _build_index(self, patterns: Dict):
        """Build inverted index with term frequency tracking"""
        intents_with_patterns = [name for name in patterns.keys()
                                if name != "unknown" and patterns[name].get("patterns")]
        self.total_intents = len(intents_with_patterns)

        for intent_name in intents_with_patterns:
            word_freq = defaultdict(int)
            total_words = 0
            unique_words = set()

            for pattern in patterns[intent_name].get("patterns", []):
                words = self.normalizer.extract_filtered_words(pattern)
                for word in words:
                    self.index[word].add(intent_name)
                    word_freq[word] += 1
                    total_words += 1
                    unique_words.add(word)

            for word in unique_words:
                self.term_document_freq[word] += 1

            if total_words > 0:
                self.intent_keyword_scores[intent_name] = {
                    word: count / total_words for word, count in word_freq.items()
                }

    def _calculate_idf_weights(self):
        """Calculate IDF (Inverse Document Frequency) weights for terms"""
        for term, doc_freq in self.term_document_freq.items():
            self.idf_weights[term] = math.log((self.total_intents + 1) / (doc_freq + 1))

    def get_candidate_intents(self, query_words: Set[str]) -> List[Tuple[str, float]]:
        """Get candidate intents ranked by TF-IDF weighted relevance"""
        if not query_words:
            return [(intent, 0.0) for intent in self.intent_keyword_scores.keys()]

        intent_scores = defaultdict(float)
        for word in query_words:
            if word in self.index:
                idf = self.idf_weights.get(word, 1.0)
                for intent_name in self.index[word]:
                    tf = self.intent_keyword_scores.get(intent_name, {}).get(word, 0.1)
                    intent_scores[intent_name] += tf * idf

        if not intent_scores:
            return [(intent, 0.0) for intent in self.intent_keyword_scores.keys()]

        return sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)


class TextNormalizer:
    """Handles all text normalization and preprocessing"""

    def __init__(self, filler_words: Set[str]):
        self.filler_words = filler_words

    def normalize(self, text: str) -> str:
        """Normalize text for comparison - optimized for ASR input"""
        if not text:
            return ""
        text = text.lower().strip()
        text = text.translate(str.maketrans('', '', '!?.,;:\'"()[]{}/@'))
        return ' '.join(text.split())

    def extract_filtered_words(self, text: str) -> List[str]:
        """Extract and filter words in one step"""
        words = self.normalize(text).split()
        return [w for w in words if w not in self.filler_words]


class SimilarityCalculator:
    """Handles all similarity metric calculations - OPTIMIZED"""

    def __init__(self, text_normalizer: TextNormalizer, synonym_lookup: Dict):
        self.normalizer = text_normalizer
        self.synonym_lookup = synonym_lookup

    def passes_length_prefilter(self, query_norm: str, pattern_norm: str) -> bool:
        """Quick length-based filter to skip expensive calculations"""
        len_query = len(query_norm)
        len_pattern = len(pattern_norm)

        if len_query <= LONG_STRING_THRESHOLD or len_pattern <= LONG_STRING_THRESHOLD:
            return True

        len_diff = abs(len_query - len_pattern)
        len_ratio = len_query / len_pattern if len_pattern > 0 else 0

        return len_diff <= MAX_LENGTH_DIFF_LONG_STRINGS and MIN_LENGTH_RATIO <= len_ratio <= MAX_LENGTH_RATIO

    def calculate_keyword_similarity(self, query_set: Set[str], pattern_set: Set[str]) -> Tuple[float, float, float]:
        """Calculate keyword-based similarity metrics"""
        exact_overlap = len(query_set & pattern_set)
        union_size = len(query_set | pattern_set)
        exact_similarity = exact_overlap / union_size if union_size > 0 else 0.0

        query_expanded = self._expand_with_synonyms(query_set)
        pattern_expanded = self._expand_with_synonyms(pattern_set)
        synonym_overlap = len(query_expanded & pattern_expanded)
        expanded_union_size = len(query_expanded | pattern_expanded)
        synonym_similarity = synonym_overlap / expanded_union_size if expanded_union_size > 0 else 0.0

        keyword_similarity = EXACT_OVERLAP_WEIGHT * exact_similarity + SYNONYM_WEIGHT * synonym_similarity
        return keyword_similarity, exact_similarity, synonym_similarity

    def _expand_with_synonyms(self, words: Set[str]) -> Set[str]:
        """Expand word set with known synonyms"""
        expanded = set(words)
        for word in words:
            if word in self.synonym_lookup:
                expanded.update(self.synonym_lookup[word])
        return expanded

    def calculate_phrase_bonus(self, query_words: List[str], pattern_words: List[str]) -> float:
        """Calculate bonus for matching consecutive word phrases"""
        if len(pattern_words) < 2:
            return 0.0

        query_text = ' '.join(query_words)
        for n in [3, 2]:
            if len(pattern_words) >= n:
                for i in range(len(pattern_words) - n + 1):
                    if ' '.join(pattern_words[i:i + n]) in query_text:
                        return PHRASE_3_WORD_BONUS if n == 3 else PHRASE_2_WORD_BONUS
        return 0.0

    def calculate_keyword_bonus(self, query_set: Set[str], intent_name: Optional[str],
                               intent_critical_keywords: Dict) -> float:
        """Calculate bonus for matching critical intent keywords"""
        if intent_name and intent_name in intent_critical_keywords:
            critical_keywords = intent_critical_keywords[intent_name]
            num_matches = len(query_set & critical_keywords)
            if num_matches:
                return min(MAX_KEYWORD_BONUS, FIRST_KEYWORD_BONUS + (num_matches - 1) * ADDITIONAL_KEYWORD_BONUS)

        max_bonus = 0.0
        for critical_keywords in intent_critical_keywords.values():
            num_matches = len(query_set & critical_keywords)
            if num_matches:
                bonus = min(MAX_KEYWORD_BONUS, FIRST_KEYWORD_BONUS + (num_matches - 1) * ADDITIONAL_KEYWORD_BONUS)
                max_bonus = max(max_bonus, bonus)
        return max_bonus

    def calculate_similarity(self, query: str, pattern: str, intent_name: Optional[str],
                           pattern_norm: Optional[str], intent_critical_keywords: Dict) -> Tuple[float, SimilarityMetrics]:
        """Main similarity calculation coordinator - OPTIMIZED"""
        query_norm = self.normalizer.normalize(query)
        pattern_norm = pattern_norm or self.normalizer.normalize(pattern)

        if not query_norm or not pattern_norm or not self.passes_length_prefilter(query_norm, pattern_norm):
            return 0.0, SimilarityMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        query_words = self.normalizer.extract_filtered_words(query)
        pattern_words = self.normalizer.extract_filtered_words(pattern)

        if not query_words or not pattern_words:
            return 0.0, SimilarityMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        query_set = set(query_words)
        pattern_set = set(pattern_words)

        keyword_sim, exact_sim, synonym_sim = self.calculate_keyword_similarity(query_set, pattern_set)

        if keyword_sim < LEVENSHTEIN_SKIP_THRESHOLD:
            metrics = SimilarityMetrics(keyword_sim, exact_sim, synonym_sim, 0.0, 0.0, 0.0, keyword_sim, keyword_sim)
            return keyword_sim, metrics

        levenshtein_sim = Levenshtein.ratio(query_norm, pattern_norm)
        phrase_bonus = self.calculate_phrase_bonus(query_words, pattern_words)
        keyword_bonus = self.calculate_keyword_bonus(query_set, intent_name, intent_critical_keywords)

        base_score = KEYWORD_WEIGHT * keyword_sim + LEVENSHTEIN_WEIGHT * levenshtein_sim
        final_score = min(1.0, base_score + phrase_bonus + keyword_bonus)

        metrics = SimilarityMetrics(keyword_sim, exact_sim, synonym_sim, levenshtein_sim,
                                   phrase_bonus, keyword_bonus, base_score, final_score)
        return final_score, metrics


class AlgorithmicRecognizer:
    """Optimized pattern-based intent recognition using keywords and string similarity"""

    def __init__(self, patterns_file: str = None, enable_logging: bool = False,
                 min_confidence: float = 0.5, linguistic_resources_file: str = None, use_boost_engine: bool = True):
        """Initialize the optimized algorithmic recognizer
s
        Args:
            patterns_file: Path to intent patterns JSON
            enable_logging: Enable detailed logging
            min_confidence: Minimum confidence threshold
            linguistic_resources_file: Path to linguistic resources JSON
            use_boost_engine: Enable domain-specific contextual boost rules
        """
        self.patterns_file = patterns_file or IntentRecognizerUtils.get_default_patterns_file()
        self.min_confidence = min_confidence
        self.enable_logging = enable_logging

        if enable_logging:
            self.logger = logging.getLogger(__name__)

        self.patterns = IntentRecognizerUtils.load_patterns_from_file(self.patterns_file, enable_logging)

        resources = LinguisticResourceLoader.load_resources(linguistic_resources_file)
        self.synonyms = resources['synonyms']
        self.filler_words = resources['filler_words']
        self.intent_critical_keywords = resources['intent_critical_keywords']
        self.synonym_lookup = LinguisticResourceLoader.build_synonym_lookup(self.synonyms)

        self.text_normalizer = TextNormalizer(self.filler_words)
        self.similarity_calculator = SimilarityCalculator(self.text_normalizer, self.synonym_lookup)
        self.use_boost_engine = use_boost_engine
        if self.use_boost_engine:
            self.boost_engine = BoostRuleEngine(self.intent_critical_keywords, enable_logging)

        self.inverted_index = InvertedIndex(self.patterns, self.text_normalizer) if INVERTED_INDEX_ENABLED else None

        self._intent_keywords_cache = {}
        self._normalized_patterns_cache = {}
        self._preprocess_patterns()

        self.stats = {
            'total_queries': 0,
            'intent_distribution': defaultdict(int),
            'avg_confidence': [],
            'intents_evaluated': [],
            'patterns_evaluated': [],
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
                keywords.update(normalized.split())
                normalized_patterns.append(normalized)

            self._intent_keywords_cache[intent_name] = keywords
            self._normalized_patterns_cache[intent_name] = normalized_patterns

    def _filter_patterns_by_length(self, query: str, patterns: List[str],
                                   normalized_patterns: List[str]) -> List[Tuple[int, str, str]]:
        """Filter patterns by length difference"""
        query_len = len(query)
        max_diff = 40 if query_len < 20 else 30

        return [(i, p, norm) for i, (p, norm) in enumerate(zip(patterns, normalized_patterns))
                if abs(query_len - len(p)) <= max_diff]

    def _evaluate_single_intent_optimized(self, intent_name: str, intent_data: Dict,
                                         query: str, query_words: Set[str]) -> Optional[IntentEvaluation]:
        """Optimized intent evaluation with conservative filtering"""
        patterns = intent_data.get("patterns", [])
        normalized_patterns = self._normalized_patterns_cache.get(intent_name, [])

        if not patterns:
            return None

        if LENGTH_DIFF_FILTER_ENABLED:
            filtered = self._filter_patterns_by_length(query, patterns, normalized_patterns)
        else:
            filtered = [(i, p, normalized_patterns[i] if i < len(normalized_patterns)
                        else self.text_normalizer.normalize(p)) for i, p in enumerate(patterns)]

        if not filtered:
            return None

        # Filter patterns with word overlap
        patterns_to_check = []
        for idx, pattern, pattern_norm in filtered:
            pattern_words = set(self.text_normalizer.extract_filtered_words(pattern))
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

            if similarity > HIGH_SIMILARITY_EARLY_EXIT:
                break

        self.stats['patterns_evaluated'].append(len(patterns_to_check))

        return IntentEvaluation(max_similarity, best_pattern, best_breakdown)

    def _evaluate_all_intents_optimized(self, query: str, query_words: Set[str]) -> Dict:
        """Optimized intent evaluation using inverted index (if enabled)"""
        if INVERTED_INDEX_ENABLED and self.inverted_index:
            candidate_intents = self.inverted_index.get_candidate_intents(query_words)
        else:
            candidate_intents = [(intent, 0.0) for intent in self.patterns.keys() if intent != 'unknown']

        if not candidate_intents:
            candidate_intents = [(intent, 0.0) for intent in self.patterns.keys() if intent != 'unknown']

        self.stats['intents_evaluated'].append(len(candidate_intents))

        if self.enable_logging:
            self.logger.debug(f"Candidate intents: {[intent for intent, _ in candidate_intents]}")

        intent_scores = {}
        for intent_name, _ in candidate_intents:
            intent_data = self.patterns.get(intent_name)
            if not intent_data:
                continue

            evaluation = self._evaluate_single_intent_optimized(intent_name, intent_data, query, query_words)
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
        """Find the best matching intent for a given query"""
        if not query or not self.patterns:
            return "unknown", 0.0, "", {}

        query_words = set(self.text_normalizer.extract_filtered_words(query))
        if not query_words:
            return "unknown", 0.0, "", {}

        intent_scores = self._evaluate_all_intents_optimized(query, query_words)
        if not intent_scores:
            return "unknown", 0.0, "", {}

        if self.use_boost_engine:
            intent_scores = self.boost_engine.apply_all_boosts(query_words, intent_scores, query)

        return self._select_best_intent(intent_scores)

    def recognize(self, query: str) -> AlgorithmicResult:
        """Main recognition method"""
        self.stats['total_queries'] += 1

        intent_name, similarity, matched_pattern, breakdown = self.find_best_match(query)
        confidence_level = IntentRecognizerUtils.determine_confidence_level(similarity)

        method = 'none'
        if breakdown:
            if breakdown.get('keyword_similarity', 0) > 0.7:
                method = 'keyword'
            elif breakdown.get('levenshtein_similarity', 0) > 0.7:
                method = 'levenshtein'
            else:
                method = 'keyword + levenshtein'

        self.stats['intent_distribution'][intent_name] += 1
        self.stats['avg_confidence'].append(similarity)

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
        avg_conf = sum(self.stats['avg_confidence']) / len(self.stats['avg_confidence']) if self.stats['avg_confidence'] else 0.0
        avg_intents_checked = sum(self.stats['intents_evaluated']) / len(self.stats['intents_evaluated']) if self.stats['intents_evaluated'] else 0
        avg_patterns_checked = sum(self.stats['patterns_evaluated']) / len(self.stats['patterns_evaluated']) if self.stats['patterns_evaluated'] else 0

        return {
            'total_queries_processed': self.stats['total_queries'],
            'intent_distribution': dict(self.stats['intent_distribution']),
            'average_confidence': avg_conf,
            'avg_intents_evaluated_per_query': avg_intents_checked,
            'avg_patterns_evaluated_per_query': avg_patterns_checked,
        }