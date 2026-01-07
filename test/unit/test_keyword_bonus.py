"""
Unit tests for keyword bonus calculation

Tests the keyword bonus functionality from SimilarityCalculator,
which rewards matches with intent-specific critical keywords.
Bonuses: 1st match = 0.08, subsequent matches = 0.04 each, capped at 0.16

Run with: python -m pytest test/unit/test_keyword_bonus.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.text_processor import TextProcessor
from intentRecognizer.algorithmic import SimilarityCalculator


class TestKeywordBonus:
    """Test keyword bonus calculation functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.text_processor = TextProcessor()
        self.calculator = SimilarityCalculator(self.text_processor, {})

        self.intent_critical_keywords = {
            "order_pizza": {"order", "pizza", "delivery"},
            "complaint": {"complaint", "issue", "problem", "wrong"},
            "menu_inquiry": {"menu", "available", "options"},
            "track_order": {"track", "status", "where", "order"}
        }

    @pytest.mark.parametrize("query_set,intent_name,expected_bonus", [
        # Single keyword match (0.08 bonus)
        ({"order", "please"}, "order_pizza", 0.08),
        ({"show", "menu"}, "menu_inquiry", 0.08),
        ({"complaint", "service"}, "complaint", 0.08),

        # Two keyword matches (0.08 + 0.04 = 0.12 bonus)
        ({"order", "pizza"}, "order_pizza", 0.12),
        ({"complaint", "issue"}, "complaint", 0.12),
        ({"menu", "available"}, "menu_inquiry", 0.12),

        # Three keyword matches (0.08 + 0.04 + 0.04 = 0.16 bonus, capped at 0.16)
        ({"order", "pizza", "delivery"}, "order_pizza", 0.16),
        ({"complaint", "issue", "problem"}, "complaint", 0.16),

        # Four keyword matches (still capped at 0.16)
        ({"track", "status", "where", "order"}, "track_order", 0.16),

        # No keyword match (0.0 bonus)
        ({"hello", "world"}, "order_pizza", 0.0),
        ({"random", "words"}, "complaint", 0.0),
    ])
    def test_keyword_bonus_with_intent(self, query_set, intent_name, expected_bonus):
        """Test keyword bonus when intent is specified"""
        bonus = self.calculator.calculate_keyword_bonus(
            query_set, intent_name, self.intent_critical_keywords
        )
        assert abs(bonus - expected_bonus) < 0.01, f"Expected {expected_bonus}, got {bonus}"

    def test_keyword_bonus_scaling(self):
        """Test that keyword bonus scales correctly with number of matches"""
        # 1 match: 0.08
        bonus_1 = self.calculator.calculate_keyword_bonus(
            {"order"}, "order_pizza", self.intent_critical_keywords
        )

        # 2 matches: 0.08 + 0.04 = 0.12
        bonus_2 = self.calculator.calculate_keyword_bonus(
            {"order", "pizza"}, "order_pizza", self.intent_critical_keywords
        )

        # 3 matches: 0.08 + 0.04 + 0.04 = 0.16
        bonus_3 = self.calculator.calculate_keyword_bonus(
            {"order", "pizza", "delivery"}, "order_pizza", self.intent_critical_keywords
        )

        assert bonus_1 == 0.08
        assert bonus_2 == 0.12
        assert bonus_3 == 0.16
        assert bonus_2 > bonus_1
        assert bonus_3 > bonus_2

    def test_keyword_bonus_capped(self):
        """Test that keyword bonus is capped at 0.16"""
        # Query with many matching keywords
        large_query_set = {"order", "pizza", "delivery", "extra1", "extra2"}

        bonus = self.calculator.calculate_keyword_bonus(
            large_query_set, "order_pizza", self.intent_critical_keywords
        )

        # Should be capped at 0.16
        assert bonus <= 0.16
        assert abs(bonus - 0.16) < 0.01

    def test_keyword_bonus_intent_not_in_critical_keywords(self):
        """Test keyword bonus when intent has no critical keywords defined"""
        bonus = self.calculator.calculate_keyword_bonus(
            {"random", "words"}, "unknown_intent", self.intent_critical_keywords
        )
        assert bonus == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])