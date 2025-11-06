"""
Text Processing Utilities for Voice Assistant
Text normalization, contraction expansion, and word extraction
"""

from typing import Set, List

class TextProcessor:

    CONTRACTION_SUFFIXES = [
        ("n't", " not"),
        ("'ve", " have"),
        ("'re", " are"),
        ("'ll", " will"),
        ("'d", " would"),
        ("'m", " am"),
        ("'s", " is"),
    ]

    CONTRACTION_SPECIAL_CASES = {
        "won't": "will not",
        "can't": "cannot",
        "ain't": "am not"
    }

    def __init__(self, filler_words: Set[str] = None):
        """
        Initialize TextProcessor

        Args:
            filler_words: Optional set of filler words to filter out during extraction
        """
        self.filler_words = filler_words or set()

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand English contractions using suffix-based matching"""
        words = text.split()
        expanded_words = []

        for word in words:
            word_lower = word.lower()

            if word_lower in TextProcessor.CONTRACTION_SPECIAL_CASES:
                expanded_words.append(TextProcessor.CONTRACTION_SPECIAL_CASES[word_lower])
                continue

            expanded = False
            for suffix, replacement in TextProcessor.CONTRACTION_SUFFIXES:
                if word_lower.endswith(suffix):
                    base = word_lower[:-len(suffix)]
                    expanded_words.append(base + replacement)
                    expanded = True
                    break

            if not expanded:
                expanded_words.append(word)

        return ' '.join(expanded_words)

    def normalize(self, text: str) -> str:
        """Lowercase, strip whitespace, Expand contractions, Remove punctuation"""
        if not text:
            return ""

        text = text.lower().strip()
        text = self.expand_contractions(text)
        text = text.translate(str.maketrans('', '', '!?.,;:\'"()[]{}/@'))
        return ' '.join(text.split())

    def extract_filtered_words(self, text: str) -> List[str]:
        words = self.normalize(text).split()
        return [w for w in words if w not in self.filler_words]

    def extract_words(self, text: str) -> List[str]:
        """Extract normalized words without filtering"""
        return self.normalize(text).split()