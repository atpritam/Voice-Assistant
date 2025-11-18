"""
Embedding Cache Manager
Handles caching and loading of precomputed embeddings
"""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional
from utils.logger import ConditionalLogger


CACHE_DIR = Path.home() / ".cache" / "voice-assistant" / "embeddings"


class EmbeddingCache:
    """
    Manages caching of precomputed embeddings for semantic intent recognition.
    Caches embeddings based on model name and patterns file hash.
    """

    def __init__(
        self,
        model_name: str,
        patterns_file: str,
        enable_logging: bool = False
    ):
        """
        Initialize the embedding cache manager.

        Args:
            model_name: Name of the sentence transformer model
            patterns_file: Path to the intent patterns file
            enable_logging: Whether to enable detailed logging
        """
        self.model_name = model_name
        self.patterns_file = patterns_file
        self.logger = ConditionalLogger(__name__, enable_logging)

    def _get_patterns_hash(self) -> str:
        """Calculate MD5 hash of patterns file for cache invalidation."""
        try:
            with open(self.patterns_file, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error hashing patterns file: {e}")
            return ""

    def _get_cache_path(self) -> Path:
        """Get path to cache file for current model and patterns."""
        patterns_hash = self._get_patterns_hash()
        cache_filename = f"{self.model_name.replace('/', '_')}_{patterns_hash}.pkl"
        return CACHE_DIR / cache_filename

    def load(self) -> Optional[Dict]:
        """Load precomputed embeddings from cache."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            # Validate cache structure
            if not all(k in cached_data for k in ['embeddings', 'model_name']):
                self.logger.warning("Invalid cache format, ignoring cache")
                return None

            # Validate model name matches
            if cached_data['model_name'] != self.model_name:
                self.logger.warning("Cache model mismatch, ignoring cache")
                return None

            intent_embeddings = cached_data['embeddings']
            total_patterns = sum(len(data['patterns']) for data in intent_embeddings.values())
            self.logger.info(
                f"Loaded embeddings from cache for {len(intent_embeddings)} intents, "
                f"total patterns: {total_patterns}"
            )
            return intent_embeddings

        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            return None

    def save(self, intent_embeddings: Dict):
        """Save precomputed embeddings to cache."""
        cache_path = self._get_cache_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving embeddings to cache: {cache_path}")

            cache_data = {
                'embeddings': intent_embeddings,
                'model_name': self.model_name,
                'patterns_hash': self._get_patterns_hash()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            self.logger.info("Embeddings cached successfully")
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")

    def clear_all(self):
        """Clear all cached embeddings in the cache directory."""
        try:
            if CACHE_DIR.exists():
                for cache_file in CACHE_DIR.glob("*.pkl"):
                    cache_file.unlink()
                self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")