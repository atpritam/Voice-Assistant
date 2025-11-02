"""
Comparative test runner
Runs multiple pipeline configurations and compares results
"""

import sys
import os
import time
import logging
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from intentRecognizer.intent_recognizer import IntentRecognizer
from test.data import get_test_dataset
from test.common import (
    CONFIG,
    RecognizerFactory,
    format_time,
    print_section
)
from utils.logger_config import setup_logging


class ComparativeTestRunner:
    """Run comparative analysis across multiple configurations"""

    CONFIGS = [
        ("Full Pipeline", True, True, True),
        ("Algorithmic -> Semantic", True, True, False),
        ("Algorithmic -> LLM", True, False, True),
        ("Semantic -> LLM", False, True, True),
        ("Algorithmic Only", True, False, False),
        ("Semantic Only", False, True, False),
    ]

    CONFIGS += [("LLM Only", False, False, True)] if CONFIG.use_local_llm else []

    def __init__(self, custom_data=None):
        """Initialize test runner"""
        setup_logging(level=logging.INFO)
        self.factory = RecognizerFactory()
        self.test_data = custom_data if custom_data is not None else get_test_dataset(include_edge_cases=CONFIG.include_edge_cases)

    def run(self) -> None:
        """Run comparative analysis"""
        self._print_header()
        results = self._run_all_configs()
        self._print_comparison(results)

    def _print_header(self) -> None:
        """Print test header with configuration info"""
        print_section("COMPARATIVE TEST")
        print(f"\nTesting multiple pipeline configurations for comparative results")
        print(f"Semantic Model: {CONFIG.semantic_model}")
        print(f"LLM Model: {CONFIG.llm_model_name}")
        mode = "TEST MODE (intent recognition only)" if CONFIG.test_mode else "Test Mode Disabled"
        print(f"Mode: {mode}")
        print(f"Boost Engine: {CONFIG.use_boost_engine}")
        print(f"Edge Cases Included: {CONFIG.include_edge_cases}\n")
        print(f"Test Dataset Size: {len(self.test_data)} queries\n")

    def _run_all_configs(self) -> List[Dict]:
        """
        Run all configurations

        Returns:
            List of result dictionaries
        """
        results = []
        for name, algo, semantic, llm in self.CONFIGS:
            print(f"\n{'─' * 80}\n{name}\n{'─' * 80}")
            try:
                rec = self.factory.create(algo, semantic, llm, log=False)
                self.factory.warmup(rec, semantic, llm, CONFIG.use_local_llm)
                ev, duration = self._run_evaluation(rec)

                stats = rec.get_statistics()
                llm_stats = stats.get('llm_layer', {})
                tokens = llm_stats.get('total_tokens_used', 0)
                llm_calls = llm_stats.get('total_api_calls', 0)

                results.append({
                    "name": name,
                    "acc": ev["accuracy"],
                    "correct": ev["correct"],
                    "unknown": sum(1 for r in ev["detailed_results"] if r["predicted"] == "unknown"),
                    "total": ev["total_queries"],
                    "time": duration,
                    "qps": len(self.test_data) / duration,
                    "algo": ev.get("algo_used_count", 0),
                    "sem": ev.get("semantic_used_count", 0),
                    "llm": ev.get("llm_used_count", 0),
                    "tokens": tokens,
                    "llm_calls": llm_calls
                })
                print(f"✓ Accuracy: {ev['accuracy']:.2%} ({ev['correct']}/{ev['total_queries']} correct)\n"
                      f"Time: {format_time(duration)}")
                if tokens > 0:
                    print(f"Tokens: {tokens:,} ({tokens/len(self.test_data):.1f} avg/query)")
            except Exception as e:
                print(f"✗ {name} failed: {e}")
                results.append({
                    "name": name, "acc": 0, "correct": 0, "total": len(self.test_data),
                    "time": 0, "qps": 0, "algo": 0, "sem": 0, "llm": 0, "unknown": len(self.test_data),
                    "tokens": 0, "llm_calls": 0
                })
        return results

    def _run_evaluation(self, recognizer: IntentRecognizer) -> Tuple[Dict, float]:
        """
        Run evaluation and return results with duration

        Args:
            recognizer: IntentRecognizer instance

        Returns:
            Tuple of (evaluation_dict, duration_seconds)
        """
        start = time.time()
        ev = recognizer.evaluate(self.test_data)
        duration = time.time() - start
        return ev, duration

    def _print_comparison(self, results: List[Dict]) -> None:
        """
        Print comparative analysis

        Args:
            results: List of result dictionaries
        """
        print_section("COMPARATIVE ANALYSIS")

        print("\nPipeline Comparison\n" + "-" * 80)
        print(f"{'Configuration':<25} {'Accuracy':<10} {'Total Time':<12} {'Avg Time':<10} {'Q/s':<10}")
        print("-" * 80)
        for r in results:
            avg_time = r['time'] / len(self.test_data) if len(self.test_data) else 0
            print(f"{r['name']:<25} {r['acc']:>8.2%}  {format_time(r['time']):>10}  "
                  f"{format_time(avg_time):>8}  {r['qps']:>8.1f}")

        print("\nLAYER USAGE COMPARISON\n" + "-"*80)
        print(f"{'Configuration':<25} {'Algo':<8} {'Semantic':<10} {'LLM':<8} {'Unrecognized Intent':<12}")
        print("-"*80)
        for r in results:
            print(f"{r['name']:<25} {r['algo']:>6}  {r['sem']:>8}  {r['llm']:>6} {r['unknown']:>8}")

        # Token usage comparison
        if any(r.get('tokens', 0) > 0 for r in results):
            print("\nTOKEN USAGE COMPARISON\n" + "-"*80)
            print(f"{'Configuration':<25} {'LLM Calls':<12} {'Total Tokens':<15} {'Avg/Query':<12} {'vs Full Pipeline':<15}")
            print("-"*80)
            full_pipeline_tokens = next((r['tokens'] for r in results if r['name'] == 'Full Pipeline'), 0)
            for r in results:
                tokens = r.get('tokens', 0)
                llm_calls = r.get('llm_calls', 0)
                if tokens > 0 or llm_calls > 0:
                    avg_tokens = tokens / len(self.test_data) if len(self.test_data) > 0 else 0
                    if full_pipeline_tokens > 0 and r['name'] != 'Full Pipeline':
                        comparison = f"+{((tokens/full_pipeline_tokens - 1) * 100):>6.1f}%"
                    elif r['name'] == 'Full Pipeline':
                        comparison = "baseline"
                    else:
                        comparison = "N/A"
                    print(f"{r['name']:<25} {llm_calls:>10}  {tokens:>13,}  {avg_tokens:>10.1f}  {comparison:>13}")
        print()