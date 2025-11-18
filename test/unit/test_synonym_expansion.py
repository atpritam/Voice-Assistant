"""
Unit tests for synonym expansion functionality

Tests the synonym expansion capabilities from SimilarityCalculator,
verifying that words are properly expanded to their synonym groups
and that synonym lookup works bidirectionally.

Run with: python -m pytest test/unit/test_synonym_expansion.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.text_processor import TextProcessor
from intentRecognizer.algorithmic import SimilarityCalculator, LinguisticResourceLoader


class TestExpandWithSynonyms:
    """Test synonym expansion functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.synonyms = {
            "order": {"order", "buy", "purchase", "get", "want"},
            "delivery": {"delivery", "deliver", "shipped", "transport"},
            "complaint": {"complaint", "issue", "problem", "wrong"}
        }
        self.synonym_lookup = LinguisticResourceLoader.build_synonym_lookup(self.synonyms)
        self.text_processor = TextProcessor()
        self.calculator = SimilarityCalculator(self.text_processor, self.synonym_lookup)

    @pytest.mark.parametrize("input_words,expected_expansions", [
        # Single word with synonyms
        ({"order"}, {"order", "buy", "purchase", "get", "want"}),
        ({"delivery"}, {"delivery", "deliver", "shipped", "transport"}),
        ({"complaint"}, {"complaint", "issue", "problem", "wrong"}),

        # Different words from same synonym group should expand to same set
        ({"buy"}, {"order", "buy", "purchase", "get", "want"}),
        ({"purchase"}, {"order", "buy", "purchase", "get", "want"}),
    ])
    def test_single_word_expansion(self, input_words, expected_expansions):
        """Test expanding single words with known synonyms"""
        expanded = self.calculator._expand_with_synonyms(input_words)
        assert expanded == expected_expansions

    @pytest.mark.parametrize("input_words,expected_in_expansion,expected_count_min", [
        # Multiple words from different groups
        ({"order", "delivery"}, {"order", "buy", "delivery", "deliver"}, 8),
        ({"order", "complaint"}, {"order", "buy", "complaint", "issue"}, 8),

        # Mixed: some with synonyms, some without
        ({"order", "pizza"}, {"order", "buy", "pizza", "purchase"}, 6),
        ({"delivery", "fast"}, {"delivery", "deliver", "fast"}, 5),
    ])
    def test_multiple_word_expansion(self, input_words, expected_in_expansion, expected_count_min):
        """Test expanding multiple words"""
        expanded = self.calculator._expand_with_synonyms(input_words)

        # Check that expected words are present
        for word in expected_in_expansion:
            assert word in expanded

        # Check minimum count
        assert len(expanded) >= expected_count_min

    def test_bidirectional_synonym_lookup(self):
        """Test that synonym lookup works bidirectionally"""
        # "buy" and "order" should expand to same set
        expanded_buy = self.calculator._expand_with_synonyms({"buy"})
        expanded_order = self.calculator._expand_with_synonyms({"order"})
        assert expanded_buy == expanded_order


if __name__ == '__main__':
    pytest.main([__file__, '-v'])