"""
Unit tests for text preprocessing and normalization

Tests the TextProcessor class functionality including
- Lowercase conversion
- Punctuation removal
- Contraction expansion
- Special character handling
- Filler word removal
- Real-world query normalization
- Number to Word conversion

Run with: python -m pytest test/unit/test_text_processor.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.text_processor import TextProcessor


class TestPreprocessText:
    """Test text preprocessing/normalization functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.filler_words = {"um", "uh", "like", "please", "so", "just", "really"}
        self.text_processor = TextProcessor(self.filler_words)

    @pytest.mark.parametrize("input_text,expected_output", [
        # Lowercase conversion
        ("HELLO WORLD", "hello world"),
        ("Order Pizza", "order pizza"),
        ("I WANT PIZZA", "i want pizza"),

        # Punctuation removal
        ("Hello, world!", "hello world"),
        ("What's the menu?", "what is the menu"),
        ("I'm hungry!", "i am hungry"),

        # Extra whitespace
        ("hello    world", "hello world"),
        ("order  pizza   delivery", "order pizza delivery"),

    ])
    def test_basic_normalization(self, input_text, expected_output):
        """Test basic normalization operations"""
        result = self.text_processor.normalize(input_text)
        assert result == expected_output

    @pytest.mark.parametrize("contraction,expected_words", [

        ("I'll", ["i", "will"]),
        ("don't", ["do", "not"]),
        ("can't", ["cannot"]),
        ("won't", ["will", "not"]),

        ("let's order", ["let", "us", "order"]),
        ("gimme this", ["give", "me", "this"]),
        ("gonna make it", ["going", "to", "make", "it"]),
        ("wanna go there", ["want", "to", "go", "there"]),
        ("lemme get that", ["let", "me", "get", "that"]),
        ("gotta go", ["got", "to", "go"]),

        ("I'll don't can't", ["i", "will", "do", "not", "cannot"]),
    ])
    def test_contraction_expansion(self, contraction, expected_words):
        """Test that contractions are properly expanded"""
        result = self.text_processor.normalize(contraction)
        for word in expected_words:
            assert word in result

    @pytest.mark.parametrize("input_text,expected_output", [
        # Em/en dashes
        ("order–delivery", "order delivery"),
        ("order—delivery", "order delivery"),

        # Numbers converted
        ("Order 5 pizzas", "order five pizzas"),
        ("Order 10 items", "order ten items"),
    ])
    def test_special_character_handling(self, input_text, expected_output):
        """Test handling of special characters"""
        result = self.text_processor.normalize(input_text)
        assert result == expected_output

    @pytest.mark.parametrize("text_with_fillers,words_to_keep,words_to_remove", [
        ("Um, I like really want to order please", ["i", "want", "order"], ["um", "like", "really", "please"]),
        ("Just give me pizza", ["give", "me", "pizza"], ["just"]),
        ("Uh, really?", [], ["uh", "really"]),
    ])
    def test_filler_word_removal(self, text_with_fillers, words_to_keep, words_to_remove):
        """Test that filler words are removed during extraction"""
        words = self.text_processor.normalize(text_with_fillers).split()

        # Content words should remain
        for word in words_to_keep:
            assert word in words

        # Filler words should be removed
        for word in words_to_remove:
            assert word not in words

    @pytest.mark.parametrize("query,expected_tokens", [
        ("So I'd like to order a large pepperoni pizza!",
         ["i", "would", "to", "order", "a", "large", "pepperoni", "pizza"]),
        ("Um what's on the menu?",
         ["what", "is", "on", "the", "menu"]),
        ("Where's like my delivery?",
         ["where", "is", "my", "delivery"]),
        ("I'm not happy with the uh service :(",
         ["i", "am", "not", "happy", "with", "the", "service"]),
        ("Can't wait for my pizza!",
         ["cannot", "wait", "for", "my", "pizza"]),
    ])
    def test_real_world_query_normalization(self, query, expected_tokens):
        """Test normalization of realistic queries with token output"""
        # Normalize and tokenize
        result = self.text_processor.normalize(query).split()
        assert result == expected_tokens

    def test_apostrophe_variant_handling(self):
        """Test that different apostrophe types are handled"""
        result1 = self.text_processor.normalize("I'm")
        result2 = self.text_processor.normalize("Iʼm")

        assert "am" in result1
        assert "am" in result2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])