"""
Boost engine comparative analysis runner
Compares performance and accuracy with and without boost engine
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Dict, List
from .common import (
    CONFIG,
    BaseTestRunner,
    ResultPrinter,
    format_time,
    print_section
)


class BoostEngineTestRunner(BaseTestRunner):
    """Run boost engine comparative analysis"""

    CONFIGS = [
        ("Algorithmic Only (WITH Boost)", True, False, False, True),
        ("Algorithmic Only (NO Boost)", True, False, False, False),
        ("Full Pipeline (WITH Boost)", True, True, True, True),
        ("Full Pipeline (NO Boost)", True, True, True, False),
    ]

    def __init__(self, custom_data=None):
        """Initialize test runner"""
        super().__init__(custom_data)
        self.printer = ResultPrinter()

    def run(self) -> None:
        """Run boost engine comparative analysis"""
        self._print_header()
        results = self._run_all_configs()
        self._print_impact_analysis(results)

    def _print_header(self) -> None:
        """Print test header with configuration info"""
        print_section("BOOST ENGINE COMPARATIVE ANALYSIS")
        print(f"Comparing Algorithmic Only and Full Pipeline with/without Boost Engine")
        self.printer.print_config_info()  # Don't show specific layer flags for boost test
        print(f"Test Dataset Size: {len(self.test_data)} queries\n")

    def _run_all_configs(self) -> List[Dict]:
        """
        Run all configurations

        Returns:
            List of result dictionaries
        """
        results = []
        for name, algo, semantic, llm, boost in self.CONFIGS:
            print(f"\n{'─' * 80}\n{name}\n{'─' * 80}")

            try:
                rec = self.factory.create(algo, semantic, llm, log=False, boost=boost)
                self.factory.warmup(rec, semantic, llm, CONFIG.llm_model)
                ev, duration = self._run_evaluation(rec)
                stats = rec.get_statistics()

                result = self._get_result_dict(ev, name, duration, stats, boost)
                results.append(result)

                self.printer.print_quick_summary(
                    ev, duration, len(self.test_data), result['tokens']
                )

            except Exception as e:
                print(f"✗ {name} failed: {e}")
                traceback.print_exc()
                results.append({
                    "name": name, "boost": boost, "acc": 0, "correct": 0, "unknown": len(self.test_data),
                    "total": len(self.test_data), "time": 0, "qps": 0, "algo": 0, "sem": 0, "llm": 0,
                    "high_conf": 0, "medium_conf": 0, "low_conf": 0, "algo_acc": 0, "sem_acc": 0, "llm_acc": 0
                })

        return results

    def _print_impact_analysis(self, results: List[Dict]) -> None:
        """
        Print boost engine impact analysis

        Args:
            results: List of result dictionaries
        """
        print_section("BOOST ENGINE IMPACT ANALYSIS")

        self._print_pipeline_comparison(
            results, "Algorithmic Only",
            "Algorithmic Only (WITH Boost)", "Algorithmic Only (NO Boost)"
        )

        self._print_pipeline_comparison(
            results, "Full Pipeline",
            "Full Pipeline (WITH Boost)", "Full Pipeline (NO Boost)",
            show_layer_usage=True
        )

    def _print_pipeline_comparison(self, results: List[Dict], title: str,
                                   boost_name: str, no_boost_name: str,
                                   show_layer_usage: bool = False) -> None:
        """
        Print comparison for a specific pipeline configuration

        Args:
            results: List of result dictionaries
            title: Pipeline title
            boost_name: Name of configuration with boost
            no_boost_name: Name of configuration without boost
            show_layer_usage: Whether to show layer usage metrics
        """
        boost = next((r for r in results if boost_name in r['name']), None)
        no_boost = next((r for r in results if no_boost_name in r['name']), None)

        if not (boost and no_boost):
            return

        print(f"{title} Pipeline:")
        print("-" * 80)

        # Calculate all metrics
        acc_diff = (boost['acc'] - no_boost['acc']) * 100
        correct_diff = boost['correct'] - no_boost['correct']
        time_diff = boost['time'] - no_boost['time']
        time_percent = (abs(time_diff) / no_boost['time']) * 100 if no_boost['time'] else 0
        faster = time_diff < 0
        sign = '-' if faster else '+'

        rows = [
            ("Accuracy", f"{acc_diff:+.2f}%", f"{no_boost['acc']:.2%}", f"{boost['acc']:.2%}"),
            ("Correct Predictions", f"{correct_diff:+d}", f"{no_boost['correct']}", f"{boost['correct']}"),
            ("Time", f"{sign}{time_percent:.0f}%",
             f"{format_time(no_boost['time'])}", f"{format_time(boost['time'])}")
        ]

        if not show_layer_usage:
            high_diff = boost['high_conf'] - no_boost['high_conf']
            rows.append(("High Confidence", f"{high_diff:+d}", f"{no_boost['high_conf']}", f"{boost['high_conf']}"))
        else:
            algo_diff = boost['algo'] - no_boost['algo']
            sem_diff = boost['sem'] - no_boost['sem']
            llm_diff = boost['llm'] - no_boost['llm']
            rows.extend([
                ("Algo Layer Usage", f"{algo_diff:+d}", f"{no_boost['algo']}", f"{boost['algo']}"),
                ("Semantic Usage", f"{sem_diff:+d}", f"{no_boost['sem']}", f"{boost['sem']}"),
                ("LLM Fallback", f"{llm_diff:+d}", f"{no_boost['llm']}", f"{boost['llm']}")
            ])

        print(f"{'Metric':<22} {'Without Boost':<15} {'With Boost':<15} {'Impact':<35}")
        print("-" * 80)
        for metric, impact, without, with_boost in rows:
            print(f"{metric:<22} {without:<15} {with_boost:<15} {impact:<35}")

        print()