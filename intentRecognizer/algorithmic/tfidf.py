"""
TF-IDF Inverted Index
Provides fast intent candidate selection using TF-IDF weighted inverted index
"""

import math
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class InvertedIndex:
    """Inverted index with TF-IDF weighting for fast intent candidate selection"""

    def __init__(self, patterns: Dict, text_processor):
        """Initialize inverted index with patterns

        Args:
            patterns: Intent patterns dictionary
            text_processor: TextProcessor instance for word extraction
        """
        self.text_processor = text_processor
        self.index = defaultdict(set)
        self.intent_keyword_scores = {}
        self.term_document_freq = defaultdict(int)
        self.idf_weights = {}
        self.total_intents = 0

        self._build_index(patterns)
        self._calculate_idf_weights()

    def _build_index(self, patterns: Dict):
        """Build inverted index with term frequency tracking

        Args:
            patterns: Intent patterns dictionary
        """
        intents_with_patterns = [name for name in patterns.keys()
                                if name != "unknown" and patterns[name].get("patterns")]
        self.total_intents = len(intents_with_patterns)

        for intent_name in intents_with_patterns:
            word_freq = defaultdict(int)
            total_words = 0
            unique_words = set()

            for pattern in patterns[intent_name].get("patterns", []):
                words = (pattern).split()
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
        """Get candidate intents ranked by TF-IDF weighted relevance

        Args:
            query_words: Set of words from the query

        Returns:
            List of (intent_name, score) tuples sorted by relevance (descending)
        """
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