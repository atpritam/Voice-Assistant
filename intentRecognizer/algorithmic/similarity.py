"""
Similarity Calculation
Handles keyword-based and Levenshtein distance-based similarity metrics
"""

from typing import Dict, Set, Tuple, List, Optional
from dataclasses import dataclass
import Levenshtein

from utils.text_processor import TextProcessor

# SIMILARITY CALCULATION WEIGHTS
KEYWORD_WEIGHT = 0.50                       # Jaccard keyword similarity
LEVENSHTEIN_WEIGHT = 0.50                   # Levenshtein String edit distance

# JACCARD KEYWORD SIMILARITY COMPOSITION
EXACT_OVERLAP_WEIGHT = 0.7                  # Exact word overlap
SYNONYM_WEIGHT = 0.3                        # synonym-expanded overlap


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
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in self.__dict__.items()}


class SimilarityCalculator:
    """Handles all similarity metric calculations - OPTIMIZED"""

    def __init__(self, text_processor: TextProcessor, synonym_lookup: Dict):
        """Initialize similarity calculator

        Args:
            text_processor: TextProcessor instance for word extraction and normalization
            synonym_lookup: Dictionary mapping words to their synonym groups
        """
        self.text_processor = text_processor
        self.synonym_lookup = synonym_lookup

    def passes_length_prefilter(self, query_norm: str, pattern_norm: str) -> bool:
        """Quick length-based filter to skip expensive calculations"""
        len_query = len(query_norm)
        len_pattern = len(pattern_norm)

        if len_query <= 15 or len_pattern <= 15:
            return True

        if len_query >= 80 or len_pattern >= 80:
            return True

        len_diff = abs(len_query - len_pattern)
        len_ratio = len_query / len_pattern if len_pattern > 0 else 0

        return len_diff <= 30 and 0.4 <= len_ratio <= 2.5

    def calculate_keyword_similarity(self, query_set: Set[str], pattern_set: Set[str]) -> Tuple[float, float, float]:
        """Calculate keyword-based similarity metrics

        Args:
            query_set: Set of words from query
            pattern_set: Set of words from pattern

        Returns:
            Tuple of (keyword_similarity, exact_similarity, synonym_similarity)
        """
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
                        return 0.10 if n == 3 else 0.05
        return 0.0

    def calculate_keyword_bonus(self, query_set: Set[str], intent_name: Optional[str],
                               intent_critical_keywords: Dict) -> float:
        """Calculate bonus for matching critical intent keywords"""
        if intent_name and intent_name in intent_critical_keywords:
            critical_keywords = intent_critical_keywords[intent_name]
            num_matches = len(query_set & critical_keywords)
            if num_matches:
                return min(0.16, 0.08 + (num_matches - 1) * 0.04)

        max_bonus = 0.0
        for critical_keywords in intent_critical_keywords.values():
            num_matches = len(query_set & critical_keywords)
            if num_matches:
                bonus = min(0.16, 0.08 + (num_matches - 1) * 0.04)
                max_bonus = max(max_bonus, bonus)
        return max_bonus

    def calculate_similarity(self, query: str, pattern: str, intent_name: Optional[str],
                           pattern_norm: Optional[str], intent_critical_keywords: Dict) -> Tuple[float, SimilarityMetrics]:
        """Main similarity calculation coordinator

        Args:
            query: User query
            pattern: Pattern to match against
            intent_name: Name of the intent (for critical keyword bonus)
            pattern_norm: Pre-normalized pattern (optional, for performance)
            intent_critical_keywords: Dictionary of critical keywords per intent

        Returns:
            Tuple of (final_similarity_score, detailed_metrics)
        """
        query_norm = self.text_processor.normalize(query)
        pattern_norm = pattern_norm or self.text_processor.normalize(pattern)

        if not query_norm or not pattern_norm or not self.passes_length_prefilter(query_norm, pattern_norm):
            return 0.0, SimilarityMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        query_words = self.text_processor.extract_filtered_words(query)
        pattern_words = self.text_processor.extract_filtered_words(pattern)

        if not query_words or not pattern_words:
            return 0.0, SimilarityMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        query_set = set(query_words)
        pattern_set = set(pattern_words)

        keyword_sim, exact_sim, synonym_sim = self.calculate_keyword_similarity(query_set, pattern_set)

        # Early exit for very low keyword similarity
        if keyword_sim < 0.2:
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