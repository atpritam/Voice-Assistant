"""
Comparative test runner
Runs multiple pipeline configurations and compares results
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Dict, List
from .common import (
    CONFIG,
    BaseTestRunner,
    ResultPrinter,
    print_section
)


class ComparativeTestRunner(BaseTestRunner):
    """Run comparative analysis across multiple configurations"""

    CONFIGS = [
        ("Full Pipeline", True, True, True),
        ("Algorithmic -> Semantic", True, True, False),
        ("Algorithmic -> LLM", True, False, True),
        ("Semantic -> LLM", False, True, True),
        ("Algorithmic Only", True, False, False),
        ("Semantic Only", False, True, False),
        ("LLM Only", False, False, True)
    ]

    def __init__(self, custom_data=None):
        """Initialize test runner"""
        super().__init__(custom_data)
        self.printer = ResultPrinter()

    def run(self) -> None:
        """Run comparative analysis"""
        self._print_header()
        results = self._run_all_configs()
        self._print_comparison(results)

    def _print_header(self) -> None:
        """Print test header with configuration info"""
        print_section("COMPARATIVE TEST")
        print(f"\nTesting multiple pipeline configurations for comparative results")
        self.printer.print_config_info()  # Don't show specific layer flags for comparative test
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

                result = self._get_result_dict(ev, name, duration, stats)
                results.append(result)

                self.printer.print_quick_summary(
                    ev, duration, len(self.test_data), result['tokens']
                )
            except Exception as e:
                print(f"✗ {name} failed: {e}")
                results.append({
                    "name": name, "acc": 0, "correct": 0, "total": len(self.test_data),
                    "time": 0, "qps": 0, "algo": 0, "sem": 0, "llm": 0, "unknown": len(self.test_data),
                    "tokens": 0, "llm_calls": 0
                })
        return results

    def _print_comparison(self, results: List[Dict]) -> None:
        """
        Print comparative analysis

        Args:
            results: List of result dictionaries
        """
        print_section("COMPARATIVE ANALYSIS")
        self.printer.print_comparison_table(results, len(self.test_data))
        self.printer.print_layer_usage_table(results)
        self.printer.print_token_usage_table(results, len(self.test_data))
        print()