"""
Algorithmic Intent Recognition Module
Provides pattern-based intent recognition using keywords, TF-IDF, and string similarity
"""
from .boostEngine import BoostRuleEngine
from .recognizer import AlgorithmicRecognizer
from .similarity import SimilarityCalculator
from .resource_loader import LinguisticResourceLoader

__all__ = [
    'AlgorithmicRecognizer',
    'SimilarityCalculator',
    'BoostRuleEngine',
    'LinguisticResourceLoader',
]