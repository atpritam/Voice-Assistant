"""
Shared utilities for the test suite
Provides configuration, recognizer factory, and result analysis
"""

import os
import sys
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from intentRecognizer.intent_recognizer import IntentRecognizer


# === CONFIGURATION ===
@dataclass
class Config:
    """Test configuration"""
    utils_dir: str = os.path.join(os.path.dirname(__file__), '..', 'utils')
    pattern_file: str = os.path.join(utils_dir, 'intent_patterns.json')
    min_confidence: float = 0.5
    semantic_model: str = "all-mpnet-base-v2"  # all-MiniLM-L6-v2

    # LLM Configuration
    use_local_llm: bool = True
    llm_model: str = "llama3.2:3b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"

    # Pipeline defaults
    enable_algo: bool = True
    enable_semantic: bool = True
    enable_llm: bool = True
    use_boost_engine: bool = True
    thresh_algo: float = 0.65
    thresh_semantic: float = 0.5

    # Test settings
    test_mode: bool = True
    include_edge_cases: bool = True

    @property
    def llm_model_name(self) -> str:
        """Return LLM model name based on backend"""
        return self.llm_model if self.use_local_llm else "gpt-5-nano"


# Global config instance
CONFIG = Config()


# === UTILITIES ===
def format_time(seconds: float) -> str:
    """Format time in ms or seconds"""
    return f"{seconds*1000:.1f}ms" if seconds < 1 else f"{seconds:.2f}s"


def describe_pipeline(algo: bool, semantic: bool, llm: bool) -> str:
    """Generate pipeline description"""
    llm_label = "LLM (Ollama)" if CONFIG.use_local_llm else "LLM (OpenAI)"
    layers = [name for enabled, name in [
        (algo, "Algorithmic"),
        (semantic, "Semantic"),
        (llm, llm_label)
    ] if enabled]
    return " -> ".join(layers) or "NO LAYERS"


def print_section(title: str, char: str = "=", width: int = 80) -> None:
    """Print formatted section header"""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


# === INTENT RECOGNIZER ===
class RecognizerFactory:
    """Factory for creating and managing IntentRecognizer instances"""

    @staticmethod
    def create(algo: bool = True, semantic: bool = True, llm: bool = True,
               log: bool = True, boost: Optional[bool] = None,
               thresh_algo: Optional[float] = None,
               thresh_semantic: Optional[float] = None) -> IntentRecognizer:
        """Create IntentRecognizer with given configuration"""
        try:
            return IntentRecognizer(
                enable_logging=False,
                enable_algorithmic=algo,
                enable_semantic=semantic,
                enable_llm=llm,
                use_boost_engine=boost if boost is not None else CONFIG.use_boost_engine,
                algorithmic_threshold=thresh_algo or CONFIG.thresh_algo,
                semantic_threshold=thresh_semantic or CONFIG.thresh_semantic,
                semantic_model=CONFIG.semantic_model,
                llm_model=CONFIG.llm_model_name,
                min_confidence=CONFIG.min_confidence,
                patterns_file=CONFIG.pattern_file,
                test_mode=CONFIG.test_mode,
                use_local_llm=CONFIG.use_local_llm,
                ollama_base_url=CONFIG.ollama_base_url
            )
        except ValueError as e:
            print(f"\nIntent Recognizer Configuration Error: {e}\n")
            sys.exit(1)

    @staticmethod
    def warmup(recognizer: IntentRecognizer, semantic: bool, llm: bool, local_llm: bool) -> None:
        """
        Warmup pipeline by running a dummy query and reset statistics
        """
        if semantic or (llm and local_llm):
            logger = logging.getLogger()
            original_level = logger.level
            logger.setLevel(logging.CRITICAL)

            try:
                recognizer.recognize_intent("sample text", [])
            finally:
                logger.setLevel(original_level)

        # Reset statistics after warmup to exclude warmup call from metrics
        recognizer.reset_statistics()


# === RESULT ANALYSIS ===
class ResultAnalyzer:
    """Analyze and display test results"""

    @staticmethod
    def print_config_info(algo: bool, semantic: bool, llm: bool) -> None:
        """Print configuration information"""
        print(f"\nPipeline: {describe_pipeline(algo, semantic, llm)}")
        if semantic:
            print(f"Semantic Model: {CONFIG.semantic_model}")
        if llm:
            print(f"LLM Model: {CONFIG.llm_model_name}")
        mode = "TEST MODE (intent recognition only)" if CONFIG.test_mode else "PRODUCTION MODE (with responses)"
        print(f"Mode: {mode}\n")
        print(f"Boost Engine: {CONFIG.use_boost_engine}")
        print(f"Edge Cases Included: {CONFIG.include_edge_cases}\n")

    @staticmethod
    def print_overall_results(ev: Dict, duration: float, total: int) -> None:
        """Print overall test results"""
        print("\nOVERALL RESULTS\n" + "-" * 80)
        print(f"Accuracy: {ev['accuracy']:.2%}")
        print(f"Correct: {ev['correct']} / {ev['total_queries']}")
        print(f"Avg Query Time: {format_time(duration / total)}")
        print(f"Queries/s: {total / duration:.1f}\n")

    @staticmethod
    def print_layer_usage(ev: Dict, total: int) -> None:
        """Print layer usage statistics"""
        print("LAYER USAGE\n" + "-" * 80)
        for layer, key in [("Algorithmic", "algo"), ("Semantic", "semantic"), ("LLM", "llm")]:
            count = ev.get(f"{key}_used_count")
            if count:
                acc = ev[f'{key}_accuracy']
                pct = count / total * 100
                print(f"{layer:<12}: {count:3d} ({pct:5.1f}%)  Acc: {acc:.2%}")

    @staticmethod
    def print_token_usage(stats: Dict, total: int) -> None:
        """Print LLM token usage statistics"""
        llm_stats = stats.get('llm_layer', {})
        tokens = llm_stats.get('total_tokens_used', 0)
        llm_calls = llm_stats.get('total_api_calls', 0)

        if tokens > 0 or llm_calls > 0:
            print("\nLLM TOKEN USAGE\n" + "-" * 80)
            print(f"Total Tokens: {tokens:,}")
            print(f"Avg Tokens/Query: {tokens/total:.1f}")

    @staticmethod
    def print_confidence_levels(ev: Dict, total: int) -> None:
        """Print confidence level distribution"""
        print("\nCONFIDENCE LEVELS\n" + "-" * 80)
        for level, key in [("High (≥0.8)", "high"), ("Medium (0.6-0.8)", "medium"), ("Low (<0.6)", "low")]:
            count = ev[f"{key}_confidence_count"]
            pct = count / total * 100
            print(f"{level:<22}: {count:3d} ({pct:5.1f}%)")

    @staticmethod
    def print_incorrect_predictions(ev: Dict) -> None:
        """Print incorrect predictions"""
        wrong = [r for r in ev["detailed_results"] if not r["correct"]]
        if not wrong:
            print("\n✓ ALL CORRECT!\n")
            return

        print("\nINCORRECT PREDICTIONS\n" + "-" * 80)
        for i, r in enumerate(wrong, 1):
            print(f"\n{i}. '{r['query']}' -> {r['predicted']} "
                  f"(exp: {r['expected']}, conf: {r['confidence']:.2f}, layer: {r['layer_used']})")


def get_available_intents() -> List[str]:
    """Load available intents from intent_patterns.json"""
    try:
        with open(CONFIG.pattern_file, 'r') as f:
            patterns = json.load(f)
        return sorted(patterns.keys())
    except Exception as e:
        print(f"Error loading intents: {e}")
        return []

def create_single_query_dataset(query: str, expected_intent: str) -> List[Tuple[str, str]]:
    """Create a minimal test dataset with a single query"""
    return [(query, expected_intent)]