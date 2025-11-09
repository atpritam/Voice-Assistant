"""
Text Processing Utilities for Voice Assistant
Text normalization, contraction expansion, and word extraction
"""

import re
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
        "ain't": "am not",
        "let's": "let us",
        "y'all": "you all",
        "gonna": "going to",
        "wanna": "want to",
        "kinda": "kind of",
        "sorta": "sort of",
        "lemme": "let me",
        "gimme": "give me",
        "gotta": "got to",
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
        """Expand English contractions using both special cases and suffix-based rules."""
        text = text.replace("'", "'").replace("'", "'").replace("ʼ", "'")

        text = re.sub(r"([!?.,;:])", r" \1 ", text)
        words = text.split()
        expanded_words = []

        for word in words:
            word_lower = word.lower().strip("!?.,;:")
            punctuation = word[len(word_lower):] if len(word_lower) < len(word) else ""

            if word_lower in TextProcessor.CONTRACTION_SPECIAL_CASES:
                replacement = TextProcessor.CONTRACTION_SPECIAL_CASES[word_lower]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                expanded_words.append(replacement + punctuation)
                continue

            expanded = False
            for suffix, replacement in TextProcessor.CONTRACTION_SUFFIXES:
                if word_lower.endswith(suffix):
                    base = word[:-len(suffix)]

                    if suffix == "'s":
                        base_lower = word_lower[:-2]
                        is_contractable = base_lower in {
                            "it", "he", "she", "that", "there", "here", "what",
                            "where", "when", "who", "why", "how"
                        }
                        if not is_contractable:
                            break

                    expanded_words.append(base + replacement + punctuation)
                    expanded = True
                    break

            if not expanded:
                expanded_words.append(word)

        return ' '.join(expanded_words)

    def normalize(self, text: str) -> str:
        """Normalize text: expand contractions, lowercase, strip punctuation, and collapse spaces."""
        if not text:
            return ""

        text = text.strip()
        text = text.replace("–", " ").replace("—", " ").replace("\n", " ")
        text = self.expand_contractions(text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', '!?.,;:\'"()[]{}/@'))
        return ' '.join(text.split())

    def extract_filtered_words(self, text: str) -> List[str]:
        """Extract normalized words excluding filler words."""
        words = self.normalize(text).split()
        return [w for w in words if w not in self.filler_words]

    def extract_words(self, text: str) -> List[str]:
        """Extract normalized words without filtering."""
        return self.normalize(text).split()