"""
Unit tests for Jaccard similarity calculations

Tests the keyword-based similarity calculation functionality from
SimilarityCalculator, including exact matches, synonym expansion,
and real-world query matching.

Run with: python -m pytest test/unit/test_similarity_calculator.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.text_processor import TextProcessor
from intentRecognizer.algorithmic import SimilarityCalculator, LinguisticResourceLoader


class TestJaccardSimilarity:
    """Test Jaccard similarity (keyword-based) calculations"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.filler_words = {"um", "uh", "like", "please", "just"}
        self.text_processor = TextProcessor(self.filler_words)

        self.synonyms = {
            "order": {"order", "buy", "purchase", "get"},
            "delivery": {"delivery", "deliver", "shipped"},
            "complaint": {"complaint", "issue", "problem"}
        }
        self.synonym_lookup = LinguisticResourceLoader.build_synonym_lookup(self.synonyms)
        self.calculator = SimilarityCalculator(self.text_processor, self.synonym_lookup)

    @pytest.mark.parametrize("query_words,pattern_words,expected_exact_sim", [
        # Identical sets
        ({"order", "pizza", "large"}, {"order", "pizza", "large"}, 1.0),
        ({"menu"}, {"menu"}, 1.0),

        # No overlap
        ({"what", "is", "menu"}, {"where", "my", "pizza"}, 0.0),
        ({"order"}, {"complaint"}, 0.0),

        # Partial overlap (2 common, 4 total = 0.5)
        ({"order", "pizza", "delivery"}, {"pizza", "delivery", "fast"}, 0.5),

        # Subset (2 common, 4 total = 0.5)
        ({"order", "pizza"}, {"order", "pizza", "large", "pepperoni"}, 0.5),
    ])
    def test_exact_similarity_values(self, query_words, pattern_words, expected_exact_sim):
        """Test exact Jaccard similarity calculations"""
        keyword_sim, exact_sim, synonym_sim = self.calculator.calculate_keyword_similarity(
            query_words, pattern_words
        )
        assert abs(exact_sim - expected_exact_sim) < 0.01


    def test_synonym_expansion_increases_similarity(self):
        """Test that synonym expansion improves similarity"""
        # "buy" and "order" are synonyms
        query_set = {"buy", "pizza"}
        pattern_set = {"order", "pizza"}

        keyword_sim, exact_sim, synonym_sim = self.calculator.calculate_keyword_similarity(
            query_set, pattern_set
        )

        # Exact similarity should be ~0.33 (only "pizza" matches)
        assert abs(exact_sim - 0.333) < 0.01

        # Synonym similarity should be higher
        assert synonym_sim > exact_sim

        # Overall keyword similarity should be higher than exact
        assert keyword_sim > exact_sim

    @pytest.mark.parametrize("query_text,pattern_text,min_similarity", [
        # High similarity pairs
        ("I want to order a pizza", "I want to order a large pizza", 0.7),
        ("order large pepperoni", "order pepperoni large", 0.9),

        # Low similarity pairs
        ("What are your hours", "My pizza is cold", 0.0),
        ("Where is my order", "What is on the menu", 0.1),
    ])
    def test_real_world_query_similarity(self, query_text, pattern_text, min_similarity):
        """Test with realistic query/pattern pairs"""
        query_words = self.text_processor.normalize(query_text).split()
        pattern_words = self.text_processor.normalize(pattern_text).split()

        keyword_sim, _, _ = self.calculator.calculate_keyword_similarity(
            set(query_words), set(pattern_words)
        )
        assert keyword_sim >= min_similarity


if __name__ == '__main__':
    pytest.main([__file__, '-v'])