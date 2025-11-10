"""
Shared utilities for the test suite
Provide configuration, recognizer factory, and result analysis
"""

import os
import sys
import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from intentRecognizer import IntentRecognizer


# === CONFIGURATION ===
@dataclass
class Config:
    """Test configuration"""
    utils_dir: str = os.path.join(os.path.dirname(__file__), '../..', 'utils')
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
        """Return LLM model name"""
        return self.llm_model


# Global config instance
CONFIG = Config()


# === UTILITY FUNCTIONS ===
def format_time(seconds: float) -> str:
    """Return formatted string representation of a time duration."""
    return f"{seconds * 1000:.1f}ms" if seconds < 1 else f"{seconds:.2f}s"


def describe_pipeline(algo: bool, semantic: bool, llm: bool) -> str:
    """Generate pipeline description"""
    llm_label = "LLM (Ollama)"
    layers = [
        name for enabled, name in [
            (algo, "Algorithmic"),
            (semantic, "Semantic"),
            (llm, llm_label)
        ] if enabled
    ]
    return " -> ".join(layers) or "NO LAYERS"


def print_section(title: str, char: str = "=", width: int = 80) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")

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


# === INTENT RECOGNIZER FACTORY ===
class RecognizerFactory:
    """Factory for creating and managing IntentRecognizer instances"""

    @staticmethod
    def create(
        algo: bool = True,
        semantic: bool = True,
        llm: bool = True,
        log: bool = True,
        boost: Optional[bool] = None,
        thresh_algo: Optional[float] = None,
        thresh_semantic: Optional[float] = None
    ) -> IntentRecognizer:
        """Create and configure an IntentRecognizer instance."""
        try:
            return IntentRecognizer(
                enable_logging=log,
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
        """Warm up the recognizer pipeline by running a dummy query."""
        if semantic or (llm and local_llm):
            logger = logging.getLogger()
            original_level = logger.level
            logger.setLevel(logging.CRITICAL)
            try:
                recognizer.recognize_intent("sample text", [])
            finally:
                logger.setLevel(original_level)

        recognizer.reset_statistics()


# === RESULT PRINTER ===
class ResultPrinter:
    """Utility class for standardized result printing."""

    @staticmethod
    def print_config_info(
        algo: Optional[bool] = None,
        semantic: Optional[bool] = None,
        llm: Optional[bool] = None,
        additional_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print test configuration details."""
        print("\nConfiguration:")
        if algo is not None:
            print(f"  Algorithmic: {algo}")
        if semantic is not None:
            print(f"  Semantic: {semantic}")
        if llm is not None:
            print(f"  LLM: {llm}")

        if ((semantic or semantic is None) or (llm or llm is None)) and (algo or algo is None):
            print(f"  Algorithmic Threshold: {CONFIG.thresh_algo}")
        if (llm or llm is None) and (semantic or semantic is None):
            print(f"  Semantic Threshold: {CONFIG.thresh_semantic}")
        if semantic or semantic is None:
            print(f"  Semantic Model: {CONFIG.semantic_model}")
        if llm or llm is None:
            print(f"  LLM Backend: Ollama")
            print(f"  LLM Model: {CONFIG.llm_model_name}")

        print(f"  Boost Engine: {CONFIG.use_boost_engine}")
        print(f"  Edge Cases: {CONFIG.include_edge_cases}")
        print()

        if additional_config:
            for key, value in additional_config.items():
                print(f"  {key}: {value}")

    @staticmethod
    def print_overall_results(ev: Dict[str, Any], duration: float, total: int) -> None:
        """Print overall evaluation results summary."""
        print("\nOVERALL RESULTS\n" + "-" * 80)
        print(f"Accuracy: {ev['accuracy']:.2%}")
        print(f"Correct: {ev['correct']} / {ev['total_queries']}")
        print(f"Avg Query Time: {format_time(duration / total)}")
        print(f"Queries/s: {total / duration:.1f}\n")

    @staticmethod
    def print_layer_usage(ev: Dict[str, Any], total: int) -> None:
        """Print usage statistics for each recognition layer."""
        print("LAYER USAGE\n" + "-" * 80)
        for layer, key in [("Algorithmic", "algo"), ("Semantic", "semantic"), ("LLM", "llm")]:
            count = ev.get(f"{key}_used_count", 0)
            if count:
                acc = ev.get(f'{key}_accuracy', 0)
                pct = count / total * 100
                print(f"{layer:<12}: {count:3d} ({pct:5.1f}%)  Acc: {acc:.2%}")

    @staticmethod
    def print_token_usage(stats: Dict[str, Any], total: int) -> None:
        """Print LLM token usage statistics."""
        llm_stats = stats.get('llm_layer', {})
        tokens = llm_stats.get('total_tokens_used', 0)
        llm_calls = llm_stats.get('total_api_calls', 0)

        if tokens > 0 or llm_calls > 0:
            print("\nLLM TOKEN USAGE\n" + "-" * 80)
            print(f"Total Tokens: {tokens:,}")
            print(f"Avg Tokens/Query: {tokens / total:.1f}")

    @staticmethod
    def print_confidence_levels(ev: Dict[str, Any], total: int) -> None:
        """Print distribution of confidence levels across predictions."""
        print("\nCONFIDENCE LEVELS\n" + "-" * 80)
        for level, key in [
            ("High (≥0.8)", "high"),
            ("Medium (0.6-0.8)", "medium"),
            ("Low (<0.6)", "low")
        ]:
            count = ev.get(f"{key}_confidence_count", 0)
            pct = count / total * 100
            print(f"{level:<22}: {count:3d} ({pct:5.1f}%)")

    @staticmethod
    def print_incorrect_predictions(ev: Dict[str, Any]) -> None:
        """Print incorrectly predicted test queries."""
        wrong = [r for r in ev.get("detailed_results", []) if not r.get("correct", False)]
        if not wrong:
            print("\n✓ ALL CORRECT!\n")
            return

        print("\nINCORRECT PREDICTIONS\n" + "-" * 80)
        for i, r in enumerate(wrong, 1):
            print(f"\n{i}. '{r['query']}' -> {r['predicted']} "
                  f"(exp: {r['expected']}, conf: {r['confidence']:.2f}, layer: {r['layer_used']})")

    @staticmethod
    def print_quick_summary(ev: Dict[str, Any], duration: float, total: int, tokens: int = 0) -> None:
        """Print quick result summary for small runs."""
        print(f"✓ Accuracy: {ev['accuracy']:.2%} ({ev['correct']}/{ev['total_queries']} correct)")
        print(f"  Time: {format_time(duration)} | Avg: {format_time(duration / total)} | {total / duration:.1f} q/s")
        print(f"  Layers Used - Algo: {ev.get('algo_used_count', 0)}, "
              f"Semantic: {ev.get('semantic_used_count', 0)}, LLM: {ev.get('llm_used_count', 0)}")
        if tokens > 0:
            print(f"  Tokens: {tokens:,} ({tokens / total:.1f} avg/query)")

    @staticmethod
    def print_comparison_table(results: List[Dict[str, Any]], total: int) -> None:
        """Print comparison table summarizing multiple configurations."""
        print("\nPipeline Comparison\n" + "-" * 80)
        print(f"{'Configuration':<25} {'Accuracy':<10} {'Total Time':<12} {'Avg Time':<10} {'Q/s':<10}")
        print("-" * 80)
        for r in results:
            avg_time = r['time'] / total if total else 0
            print(f"{r['name']:<25} {r['acc']:>8.2%}  {format_time(r['time']):>10}  "
                  f"{format_time(avg_time):>8}  {r['qps']:>8.1f}")

    @staticmethod
    def print_layer_usage_table(results: List[Dict[str, Any]]) -> None:
        """Print table comparing layer usage across configurations."""
        print("\nLAYER USAGE COMPARISON\n" + "-" * 80)
        print(f"{'Configuration':<25} {'Algo':<8} {'Semantic':<10} {'LLM':<8} {'Unrecognized Intent':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['name']:<25} {r['algo']:>6}  {r['sem']:>8}  {r['llm']:>6} {r['unknown']:>8}")

    @staticmethod
    def print_token_usage_table(results: List[Dict[str, Any]], total: int) -> None:
        """Print token usage comparison table."""
        if not any(r.get('tokens', 0) > 0 for r in results):
            return

        print("\nTOKEN USAGE COMPARISON\n" + "-" * 80)
        print(f"{'Configuration':<25} {'LLM Calls':<12} {'Total Tokens':<15} {'Avg/Query':<12} {'vs Full Pipeline':<15}")
        print("-" * 80)

        full_pipeline_tokens = next((r['tokens'] for r in results if r['name'] == 'Full Pipeline'), 0)

        for r in results:
            tokens = r.get('tokens', 0)
            llm_calls = r.get('llm_calls', 0)
            if tokens > 0 or llm_calls > 0:
                avg_tokens = tokens / total if total > 0 else 0
                if full_pipeline_tokens > 0 and r['name'] != 'Full Pipeline':
                    comparison = f"+{((tokens / full_pipeline_tokens - 1) * 100):>6.1f}%"
                elif r['name'] == 'Full Pipeline':
                    comparison = "baseline"
                else:
                    comparison = "N/A"
                print(f"{r['name']:<25} {llm_calls:>10}  {tokens:>13,}  {avg_tokens:>10.1f}  {comparison:>13}")


# === BASE TEST RUNNER ===
class BaseTestRunner:
    """Base class for test runners with shared functionality"""

    def __init__(self, custom_data=None, log_level=logging.INFO):
        """
        Initialize base test runner

        Args:
            custom_data: Optional custom test dataset
            log_level: Logging level (default: INFO)
        """
        from .data import get_test_dataset
        from utils.logger import setup_logging

        if not logging.getLogger().handlers:
            setup_logging(level=log_level)

        self.factory = RecognizerFactory()
        self.test_data = custom_data if custom_data is not None else get_test_dataset(
            include_edge_cases=CONFIG.include_edge_cases
        )

    def _run_evaluation(self, recognizer: IntentRecognizer) -> Tuple[Dict, float]:
        """Run evaluation and return results with duration"""
        start = time.time()
        ev = recognizer.evaluate(self.test_data)
        duration = time.time() - start
        return ev, duration

    def _get_result_dict(self, ev: Dict, name: str, duration: float, stats: Dict, boost: Optional[bool] = None) -> Dict:
        """
        Create standardized result dictionary

        Args:
            ev: Evaluation results
            name: Configuration name
            duration: Duration in seconds
            stats: Statistics dictionary
            boost: Optional boost engine flag

        Returns:
            Standardized result dictionary
        """
        llm_stats = stats.get('llm_layer', {})
        result = {
            "name": name,
            "acc": ev["accuracy"],
            "correct": ev["correct"],
            "unknown": sum(1 for r in ev.get("detailed_results", []) if r["predicted"] == "unknown"),
            "total": ev["total_queries"],
            "time": duration,
            "qps": len(self.test_data) / duration if duration > 0 else 0,
            "algo": ev.get("algo_used_count", 0),
            "sem": ev.get("semantic_used_count", 0),
            "llm": ev.get("llm_used_count", 0),
            "high_conf": ev.get("high_confidence_count", 0),
            "medium_conf": ev.get("medium_confidence_count", 0),
            "low_conf": ev.get("low_confidence_count", 0),
            "algo_acc": ev.get("algo_accuracy", 0),
            "sem_acc": ev.get("semantic_accuracy", 0),
            "llm_acc": ev.get("llm_accuracy", 0),
            "tokens": llm_stats.get('total_tokens_used', 0),
            "llm_calls": llm_stats.get('total_api_calls', 0)
        }
        if boost is not None:
            result["boost"] = boost
        return result