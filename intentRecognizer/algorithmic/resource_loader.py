"""
Linguistic Resources Loader
Handles loading and processing of linguistic resources (synonyms, filler words, critical keywords)
"""

import os
import json
from typing import Dict, Set
from utils.logger import ConditionalLogger


class LinguisticResourceLoader:
    """Loads linguistic resources from external JSON file"""

    @staticmethod
    def load_resources(resource_file: str = None) -> Dict:
        """Load linguistic resources from JSON file

        Args:
            resource_file: Path to linguistic resources JSON file

        Returns:
            Dictionary containing synonyms, filler_words, and intent_critical_keywords
        """
        if resource_file is None:
            utils_dir = os.path.join(os.path.dirname(__file__), '../..', 'utils')
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
            logger = ConditionalLogger(__name__, True)
            logger.info(f"Linguistic resources file not found: {resource_file}. Expected at utils/linguistic_resources.json")
            logger.info("Using blank linguistic resources.")
            return {
                'synonyms': {},
                'filler_words': set(),
                'intent_critical_keywords': {}
            }
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