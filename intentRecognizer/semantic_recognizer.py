"""
Semantic Intent Recognizer
Handles semantic similarity-based intent recognition using local Sentence Transformers
"""

import os
import sys
import hashlib
import pickle
from typing import Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import ConditionalLogger
from utils.statistics import StatisticsHelper
from utils.text_processor import TextProcessor

from .intent_recognizer import DEFAULT_MIN_CONFIDENCE, IntentRecognizerUtils

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    if TYPE_CHECKING:
        from sentence_transformers import SentenceTransformer
        import torch

DEFAULT_MODEL = "all-mpnet-base-v2"
CACHE_DIR = Path.home() / ".cache" / "voice-assistant" / "embeddings"


@dataclass
class SemanticResult:
    """Result from semantic recognition"""
    intent: str
    confidence: float
    confidence_level: str
    matched_pattern: str
    processing_method: str = "semantic"
    score_breakdown: Dict = None


class SemanticRecognizer:
    """
    Semantic Intent Recognizer using Sentence Transformers
    Uses pre-trained models to calculate semantic similarity between
    user queries and predefined intent patterns. Caches embeddings for fast initialization.
    """

    def __init__(
        self,
        patterns_file: str = None,
        model_name: str = DEFAULT_MODEL,
        enable_logging: bool = False,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        use_cache: bool = True,
        device: str = "auto"
    ):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required packages not installed. Run: pip install sentence-transformers scikit-learn")

        self.patterns_file = patterns_file or IntentRecognizerUtils.get_default_patterns_file()
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.use_cache = use_cache
        self.enable_logging = enable_logging
        self.device = device
        self.logger = ConditionalLogger(__name__, enable_logging)

        self.patterns = IntentRecognizerUtils.load_patterns_from_file(self.patterns_file, enable_logging)
        self.model = self._load_model()
        self.intent_embeddings = {}

        if use_cache:
            self._load_or_compute_embeddings()
        else:
            self._compute_embeddings()

        self.stats = StatisticsHelper.init_base_stats()

    def _load_model(self):
        """Load sentence transformer model"""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("SentenceTransformer not available")

        try:
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            self.logger.info(f"Loading Sentence Transformer model on device: {'GPU' if device == 'cuda' else 'CPU'}")

            return SentenceTransformer(self.model_name, device=device)
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _get_cache_path(self) -> Path:
        """Get path to cache file for current model and patterns"""
        try:
            with open(self.patterns_file, 'rb') as f:
                patterns_hash = hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error hashing patterns file: {e}")
            patterns_hash = ""
        return CACHE_DIR / f"{self.model_name.replace('/', '_')}_{patterns_hash}.pkl"

    def _load_embeddings_from_cache(self) -> bool:
        """Load precomputed embeddings from cache"""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            self.logger.info("No cache found, will compute embeddings")
            return False

        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            if not all(k in cached_data for k in ['embeddings', 'model_name']):
                self.logger.warning("Invalid cache format, will recompute")
                return False

            if cached_data['model_name'] != self.model_name:
                self.logger.warning("Cache model mismatch, will recompute")
                return False

            self.intent_embeddings = cached_data['embeddings']
            total_patterns = sum(len(data['patterns']) for data in self.intent_embeddings.values())
            self.logger.info(f"Loaded embeddings from cache for {len(self.intent_embeddings)} intents, total patterns: {total_patterns}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            return False

    def _save_embeddings_to_cache(self):
        """Save precomputed embeddings to cache"""
        cache_path = self._get_cache_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving embeddings to cache: {cache_path}")

            cache_data = {
                'embeddings': self.intent_embeddings,
                'model_name': self.model_name,
                'patterns_hash': cache_path.name.split('_')[-1].replace('.pkl', '')
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            self.logger.info("Embeddings cached successfully")
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")

    def _compute_embeddings(self):
        """Compute embeddings for all intent patterns"""
        self.logger.info("Precomputing pattern embeddings...")

        for intent_name, intent_data in self.patterns.items():
            if intent_name == "unknown" or not intent_data.get("patterns"):
                continue

            patterns = intent_data["patterns"]
            expanded_patterns = [TextProcessor.expand_contractions(p) for p in patterns]
            embeddings = self.model.encode(expanded_patterns, convert_to_numpy=True)
            self.intent_embeddings[intent_name] = {
                'patterns': patterns,
                'embeddings': embeddings,
                'threshold': intent_data.get('similarity_threshold', self.min_confidence)
            }

        total_patterns = sum(len(data['patterns']) for data in self.intent_embeddings.values())
        self.logger.info(f"Precomputed embeddings for {len(self.intent_embeddings)} intents, total patterns: {total_patterns}")

    def _load_or_compute_embeddings(self):
        """Load embeddings from cache or compute if not available"""
        if not self._load_embeddings_from_cache():
            self.logger.info("Computing embeddings (cache miss)")
            self._compute_embeddings()
            self._save_embeddings_to_cache()

    def clear_cache(self):
        """Clear all cached embeddings"""
        try:
            if CACHE_DIR.exists():
                for cache_file in CACHE_DIR.glob("*.pkl"):
                    cache_file.unlink()
                self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def _calculate_semantic_similarity(self, query_embedding: np.ndarray, intent_name: str, top_k: int = 3) -> Tuple[
        float, str, Dict]:
        """Calculate semantic similarity using top-K averaging"""
        intent_data = self.intent_embeddings[intent_name]
        pattern_embeddings = intent_data['embeddings']
        patterns = intent_data['patterns']

        similarities = cosine_similarity(query_embedding.reshape(1, -1), pattern_embeddings)[0]

        # Get top-K patterns
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        top_k_similarities = similarities[top_k_indices]

        # Weighted average: 50% max, 30% second, 20% third
        weights = [0.5, 0.3, 0.2][:len(top_k_similarities)]
        weights = np.array(weights) / sum(weights)  # Normalize

        max_similarity = float(np.sum(top_k_similarities * weights))
        best_pattern = patterns[top_k_indices[0]]

        breakdown = {
            'semantic_similarity': max_similarity,
            'matched_pattern': best_pattern,
            'top_k_patterns': [patterns[i] for i in top_k_indices],
            'top_k_scores': [float(similarities[i]) for i in top_k_indices],
            'all_similarities': {patterns[i]: float(similarities[i]) for i in range(len(patterns))}
        }
        return max_similarity, best_pattern, breakdown

    def recognize(self, query: str) -> SemanticResult:
        """Recognize intent using semantic similarity"""
        self.stats['total_queries'] += 1

        if not query or not self.intent_embeddings:
            return self._create_unknown_result("Empty query or no patterns")

        try:
            expanded_query = TextProcessor.expand_contractions(query)
            query_embedding = self.model.encode(expanded_query, convert_to_numpy=True)

            intent_scores = {}
            for intent_name in self.intent_embeddings.keys():
                similarity, pattern, breakdown = self._calculate_semantic_similarity(query_embedding, intent_name)
                intent_scores[intent_name] = {
                    'similarity': similarity,
                    'pattern': pattern,
                    'breakdown': breakdown
                }

            if not intent_scores:
                return self._create_unknown_result("No intent scores calculated")

            best_intent_name = max(intent_scores.items(), key=lambda x: x[1]['similarity'])[0]
            best_data = intent_scores[best_intent_name]
            best_similarity = best_data['similarity']
            best_pattern = best_data['pattern']
            breakdown = best_data['breakdown']

            threshold = self.intent_embeddings[best_intent_name]['threshold']

            if best_similarity < threshold:
                self.logger.info(
                f"unknown (best: {best_intent_name} {best_similarity:.3f}, "
                f"threshold: {threshold:.3f})"
                )
                return self._create_unknown_result(f"Best match {best_intent_name} below threshold", best_similarity)

            confidence_level = IntentRecognizerUtils.determine_confidence_level(best_similarity)

            self.stats['intent_distribution'][best_intent_name] = self.stats['intent_distribution'].get(best_intent_name, 0) + 1
            self.stats['avg_confidence'].append(best_similarity)

            self.logger.info(f"{best_intent_name} ({best_similarity:.3f}, {confidence_level})")

            return SemanticResult(
                intent=best_intent_name,
                confidence=best_similarity,
                confidence_level=confidence_level,
                matched_pattern=best_pattern,
                processing_method='semantic',
                score_breakdown=breakdown
            )

        except Exception as e:
            self.logger.error(f"Error during semantic recognition: {e}")
            return self._create_unknown_result(f"Error: {str(e)}")

    def _create_unknown_result(self, reason: str, confidence: float = 0.0) -> SemanticResult:
        """Create unknown result with given reason"""
        return SemanticResult(
            intent='unknown',
            confidence=confidence,
            confidence_level='low',
            matched_pattern='',
            processing_method='semantic',
            score_breakdown={'reason': reason}
        )

    def get_statistics(self) -> Dict:
        """Get semantic recognizer statistics"""
        avg_conf = StatisticsHelper.calculate_average(self.stats['avg_confidence'])

        return {
            'total_queries_processed': self.stats['total_queries'],
            'intent_distribution': self.stats['intent_distribution'],
            'average_confidence': avg_conf,
        }