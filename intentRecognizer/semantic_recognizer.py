"""
Semantic Intent Recognizer
Handles semantic similarity-based intent recognition using local Sentence Transformers
Includes embedding cache to avoid recomputing on every initialization
"""

import logging
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from intentRecognizer.intent_recognizer import DEFAULT_MIN_CONFIDENCE, IntentRecognizerUtils

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# Default model
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Cache directory
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
    user queries and predefined intent patterns.
    Caches embeddings for fast initialization.
    """

    def __init__(
        self,
        patterns_file: str = None,
        model_name: str = DEFAULT_MODEL,
        enable_logging: bool = False,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        use_cache: bool = True
    ):
        """
        Initialize Semantic Recognizer

        Args:
            patterns_file: Path to JSON file with intent patterns
            model_name: Name of sentence transformer model to use
            enable_logging: Enable detailed logging
            min_confidence: Minimum confidence threshold
            use_cache: Whether to use embedding cache
        """
        # Check dependencies
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required packages not installed. Run: "
                "pip install sentence-transformers scikit-learn"
            )

        # Configuration
        if patterns_file is None:
            patterns_file = IntentRecognizerUtils.get_default_patterns_file()

        self.patterns_file = patterns_file
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.use_cache = use_cache

        # Setup logging
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

        # Load patterns using shared utility
        self.patterns = IntentRecognizerUtils.load_patterns_from_file(
            patterns_file,
            enable_logging
        )

        # Initialize model
        self.model = self._load_model()

        # Load or compute embeddings
        self.intent_embeddings = {}
        if use_cache:
            self._load_or_compute_embeddings()
        else:
            self._precompute_embeddings()

        # Statistics
        self.stats = {
            'total_queries': 0,
            'intent_distribution': {},
            'avg_confidence': [],
            'model_name': model_name
        }

    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformer model"""
        try:
            import torch
            print("ML Model Device: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            model = SentenceTransformer(self.model_name, device=device)

            if self.enable_logging:
                self.logger.info(f"Model loaded on device: {device}")

            return model
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Error loading model: {e}")
            raise

    def _get_patterns_hash(self) -> str:
        """
        Generate hash of patterns file to detect changes

        Returns:
            MD5 hash of patterns file content
        """
        try:
            with open(self.patterns_file, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Error hashing patterns file: {e}")
            return ""

    def _get_cache_path(self) -> Path:
        """
        Get path to cache file for current model and patterns

        Returns:
            Path to cache file
        """
        patterns_hash = self._get_patterns_hash()
        cache_filename = f"{self.model_name.replace('/', '_')}_{patterns_hash}.pkl"
        return CACHE_DIR / cache_filename

    def _load_embeddings_from_cache(self) -> bool:
        """
        Load precomputed embeddings from cache

        Returns:
            True if loaded successfully, False otherwise
        """
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            if self.enable_logging:
                self.logger.info("No cache found, will compute embeddings")
            return False

        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            # Validate cache data
            if 'embeddings' not in cached_data or 'model_name' not in cached_data:
                if self.enable_logging:
                    self.logger.warning("Invalid cache format, will recompute")
                return False

            # Check if model matches
            if cached_data['model_name'] != self.model_name:
                if self.enable_logging:
                    self.logger.warning("Cache model mismatch, will recompute")
                return False

            self.intent_embeddings = cached_data['embeddings']

            if self.enable_logging:
                total_patterns = sum(len(data['patterns']) for data in self.intent_embeddings.values())
                self.logger.info(
                    f"Loaded embeddings from cache for {len(self.intent_embeddings)} intents, "
                    f"total patterns: {total_patterns}"
                )

            return True

        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Error loading cache: {e}")
            return False

    def _save_embeddings_to_cache(self):
        """Save precomputed embeddings to cache"""
        cache_path = self._get_cache_path()

        try:
            # Create cache directory if it doesn't exist
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            if self.enable_logging:
                self.logger.info(f"Saving embeddings to cache: {cache_path}")

            cache_data = {
                'embeddings': self.intent_embeddings,
                'model_name': self.model_name,
                'patterns_hash': self._get_patterns_hash()
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            if self.enable_logging:
                self.logger.info("Embeddings cached successfully")

        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Error saving cache: {e}")

    def _load_or_compute_embeddings(self):
        """Load embeddings from cache or compute if not available"""
        # Try to load from cache
        if self._load_embeddings_from_cache():
            return

        # Cache miss or invalid - compute embeddings
        if self.enable_logging:
            self.logger.info("Computing embeddings (cache miss)")

        self._precompute_embeddings()

        # Save to cache for next time
        self._save_embeddings_to_cache()

    def _precompute_embeddings(self):
        """Precompute embeddings for all intent patterns"""
        if self.enable_logging:
            self.logger.info("Precomputing pattern embeddings...")

        for intent_name, intent_data in self.patterns.items():
            if intent_name == "unknown":
                continue

            patterns = intent_data.get("patterns", [])
            if not patterns:
                continue

            # Encode all patterns for this intent
            embeddings = self.model.encode(patterns, convert_to_numpy=True)

            self.intent_embeddings[intent_name] = {
                'patterns': patterns,
                'embeddings': embeddings,
                'threshold': intent_data.get('similarity_threshold', self.min_confidence)
            }

        if self.enable_logging:
            total_patterns = sum(len(data['patterns']) for data in self.intent_embeddings.values())
            self.logger.info(
                f"Precomputed embeddings for {len(self.intent_embeddings)} intents, "
                f"total patterns: {total_patterns}"
            )

    def clear_cache(self):
        """Clear all cached embeddings"""
        try:
            if CACHE_DIR.exists():
                for cache_file in CACHE_DIR.glob("*.pkl"):
                    cache_file.unlink()
                if self.enable_logging:
                    self.logger.info("Cache cleared successfully")
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Error clearing cache: {e}")

    def _calculate_semantic_similarity(
        self,
        query_embedding: np.ndarray,
        intent_name: str
    ) -> Tuple[float, str, Dict]:
        """
        Calculate semantic similarity between query and intent patterns

        Args:
            query_embedding: Embedding vector for the query
            intent_name: Name of intent to compare against

        Returns:
            Tuple of (max_similarity, best_pattern, breakdown)
        """
        intent_data = self.intent_embeddings[intent_name]
        pattern_embeddings = intent_data['embeddings']
        patterns = intent_data['patterns']

        # Calculate cosine similarity with all patterns
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            pattern_embeddings
        )[0]

        # Find best match
        max_idx = np.argmax(similarities)
        max_similarity = float(similarities[max_idx])
        best_pattern = patterns[max_idx]

        # Create breakdown
        breakdown = {
            'semantic_similarity': max_similarity,
            'matched_pattern': best_pattern,
            'all_similarities': {
                patterns[i]: float(similarities[i])
                for i in range(len(patterns))
            }
        }

        return max_similarity, best_pattern, breakdown

    def recognize(self, query: str) -> SemanticResult:
        """
        Recognize intent using semantic similarity

        Args:
            query: User input string

        Returns:
            SemanticResult object with intent information
        """
        self.stats['total_queries'] += 1

        if not query or not self.intent_embeddings:
            return self._create_unknown_result("Empty query or no patterns")

        try:
            # Encode query
            query_embedding = self.model.encode(query, convert_to_numpy=True)

            # Calculate similarity with all intents
            intent_scores = {}
            for intent_name in self.intent_embeddings.keys():
                similarity, pattern, breakdown = self._calculate_semantic_similarity(
                    query_embedding, intent_name
                )
                intent_scores[intent_name] = {
                    'similarity': similarity,
                    'pattern': pattern,
                    'breakdown': breakdown
                }

            # Find best match
            if not intent_scores:
                return self._create_unknown_result("No intent scores calculated")

            best_intent_name = max(intent_scores.items(), key=lambda x: x[1]['similarity'])[0]
            best_data = intent_scores[best_intent_name]
            best_similarity = best_data['similarity']
            best_pattern = best_data['pattern']
            breakdown = best_data['breakdown']

            # Check against threshold
            threshold = self.intent_embeddings[best_intent_name]['threshold']

            if best_similarity < threshold:
                if self.enable_logging:
                    self.logger.info(
                        f"[SEMANTIC] Intent: unknown "
                        f"(best: {best_intent_name} with {best_similarity:.3f}, "
                        f"below threshold: {threshold:.3f})"
                    )
                return self._create_unknown_result(
                    f"Best match {best_intent_name} below threshold",
                    best_similarity
                )

            confidence_level = IntentRecognizerUtils.determine_confidence_level(best_similarity)

            # Update statistics
            self.stats['intent_distribution'][best_intent_name] = \
                self.stats['intent_distribution'].get(best_intent_name, 0) + 1
            self.stats['avg_confidence'].append(best_similarity)

            # Logging
            if self.enable_logging:
                self.logger.info(
                    f"[SEMANTIC] Intent: {best_intent_name} "
                    f"(confidence: {best_similarity:.3f}, level: {confidence_level})"
                )

            return SemanticResult(
                intent=best_intent_name,
                confidence=best_similarity,
                confidence_level=confidence_level,
                matched_pattern=best_pattern,
                processing_method='semantic',
                score_breakdown=breakdown
            )

        except Exception as e:
            if self.enable_logging:
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
        avg_conf = (
            sum(self.stats['avg_confidence']) / len(self.stats['avg_confidence'])
            if self.stats['avg_confidence'] else 0.0
        )

        return {
            'total_queries_processed': self.stats['total_queries'],
            'intent_distribution': self.stats['intent_distribution'],
            'average_confidence': avg_conf,
            'model_name': self.stats['model_name'],
            'total_patterns_encoded': sum(
                len(data['patterns']) for data in self.intent_embeddings.values()
            ),
            'cache_enabled': self.use_cache,
            'cache_location': str(CACHE_DIR) if self.use_cache else None
        }