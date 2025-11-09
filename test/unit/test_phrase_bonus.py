"""
Unit tests for phrase bonus calculation

Tests the phrase bonus functionality from SimilarityCalculator,
which rewards consecutive word matches in queries and patterns.
Bonuses: 3-word phrases = 0.10, 2-word phrases = 0.05

Run with: python -m pytest test/unit/test_phrase_bonus.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.text_processor import TextProcessor
from intentRecognizer.algorithmic_recognizer import SimilarityCalculator


class TestPhraseBonus:
    """Test phrase bonus calculation functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.text_processor = TextProcessor()
        self.calculator = SimilarityCalculator(self.text_processor, {})

    @pytest.mark.parametrize("query_words,pattern_words,expected_bonus", [
        # 3-word phrase match (0.10 bonus)
        (["order", "large", "pepperoni", "pizza"], ["order", "large", "pepperoni"], 0.10),
        (["i", "want", "to", "order", "pizza"], ["want", "to", "order"], 0.10),
        (["check", "my", "order", "status"], ["my", "order", "status"], 0.10),

        # 2-word phrase match (0.05 bonus)
        (["order", "pizza", "now"], ["order", "pizza"], 0.05),
        (["my", "delivery", "is", "late"], ["my", "delivery"], 0.05),
        (["show", "the", "menu"], ["the", "menu"], 0.05),

        # No phrase match (0.0 bonus)
        (["i", "want", "pizza"], ["order", "food"], 0.0),
        (["where", "is", "order"], ["my", "delivery"], 0.0),
    ])
    def test_phrase_bonus_values(self, query_words, pattern_words, expected_bonus):
        """Test phrase bonus calculation with various inputs"""
        bonus = self.calculator.calculate_phrase_bonus(query_words, pattern_words)
        assert abs(bonus - expected_bonus) < 0.01, f"Expected {expected_bonus}, got {bonus}"

    def test_phrase_bonus_order_matters(self):
        """Test that word order is important for phrase matching"""
        query_words = ["order", "large", "pizza"]

        # Consecutive match
        pattern_consecutive = ["order", "large"]
        bonus_consecutive = self.calculator.calculate_phrase_bonus(query_words, pattern_consecutive)

        # Non-consecutive (should not match as a phrase)
        pattern_scrambled = ["large", "order"]
        bonus_scrambled = self.calculator.calculate_phrase_bonus(query_words, pattern_scrambled)

        assert bonus_consecutive == 0.05
        assert bonus_scrambled == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])