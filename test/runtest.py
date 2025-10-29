"""
Test script for Intent Recognizer
Run:
  python -m test.runtest
  python -m test.runtest --c
  python -m test.runtest --b
"""

import sys
import os
import time
import argparse
import traceback
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from intentRecognizer.intent_recognizer import IntentRecognizer
from test.data import get_test_dataset
from utils.logger_config import setup_logging

# === CONFIGURATION ===
@dataclass
class Config:
    utils_dir: str = os.path.join(os.path.dirname(__file__), '..', 'utils')
    pattern_file: str = os.path.join(utils_dir, 'intent_patterns.json')
    min_confidence: float = 0.5
    semantic_model: str = "all-mpnet-base-v2" # all-MiniLM-L6-v2

    # LLM Configuration
    use_local_llm: bool = True
    llm_model: str = "llama3.2:3b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"

    # Pipeline defaults
    enable_algo: bool = True
    enable_semantic: bool = True
    enable_llm: bool = True
    use_boost_engine: bool = True
    thresh_algo: float = 0.6
    thresh_semantic: float = 0.5

    # Test settings
    test_mode: bool = True
    include_edge_cases: bool = True

    @property
    def llm_model_name(self) -> str:
        return self.llm_model if self.use_local_llm else "gpt-5-nano"

CONFIG = Config()
setup_logging(level=logging.INFO)

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
        """Warmup pipeline"""
        if semantic or (llm and local_llm):
            recognizer.recognize_intent("sample text", [])

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

# === TEST RUNNERS ===
class TestRunner:
    """Base class for test execution"""

    def __init__(self):
        self.factory = RecognizerFactory()
        self.analyzer = ResultAnalyzer()
        self.test_data = get_test_dataset(include_edge_cases=CONFIG.include_edge_cases)

    def run_evaluation(self, recognizer: IntentRecognizer) -> Tuple[Dict, float]:
        """Run evaluation and return results with duration"""
        start = time.time()
        ev = recognizer.evaluate(self.test_data)
        duration = time.time() - start
        return ev, duration

class ComprehensiveTestRunner(TestRunner):
    """Run comprehensive single-pipeline test"""

    def run(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        print_section("COMPREHENSIVE TEST")
        self.analyzer.print_config_info(CONFIG.enable_algo, CONFIG.enable_semantic, CONFIG.enable_llm)
        print(f"Test Dataset Size: {len(self.test_data)} queries\n")

        try:
            recognizer = self.factory.create(CONFIG.enable_algo, CONFIG.enable_semantic, CONFIG.enable_llm)
            self.factory.warmup(recognizer, CONFIG.enable_semantic, CONFIG.enable_llm, CONFIG.use_local_llm)
        except Exception as e:
            print(f"INIT ERROR: {e}")
            traceback.print_exc()
            return None, None

        ev, duration = self.run_evaluation(recognizer)

        self.analyzer.print_overall_results(ev, duration, len(self.test_data))
        self.analyzer.print_layer_usage(ev, len(self.test_data))
        self.analyzer.print_confidence_levels(ev, len(self.test_data))
        self.analyzer.print_incorrect_predictions(ev)

        return ev, recognizer.get_statistics()

class ComparativeTestRunner(TestRunner):
    """Run comparative analysis across multiple configurations"""

    CONFIGS = [
        ("Full Pipeline", True, True, True),
        ("Algorithmic -> Semantic", True, True, False),
        ("Algorithmic -> LLM", True, False, True),
        ("Semantic -> LLM", False, True, True),
        ("Algorithmic Only", True, False, False),
        ("Semantic Only", False, True, False),
    ]

    def run(self) -> None:
        self._print_header()
        results = self._run_all_configs()
        self._print_comparison(results)

    def _print_header(self) -> None:
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
        results = []
        for name, algo, semantic, llm in self.CONFIGS:
            print(f"\n{'─' * 80}\n{name}\n{'─' * 80}")
            try:
                rec = self.factory.create(algo, semantic, llm, log=False)
                self.factory.warmup(rec, semantic, llm, CONFIG.use_local_llm)
                ev, duration = self.run_evaluation(rec)

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
                    "llm": ev.get("llm_used_count", 0)
                })
                print(f"✓ Accuracy: {ev['accuracy']:.2%} ({ev['correct']}/{ev['total_queries']} correct)\n"
                      f"Time: {format_time(duration)}")
            except Exception as e:
                print(f"✗ {name} failed: {e}")
                results.append({
                    "name": name, "acc": 0, "correct": 0, "total": len(self.test_data),
                    "time": 0, "qps": 0, "algo": 0, "sem": 0, "llm": 0, "unknown": len(self.test_data)
                })
        return results

    def _print_comparison(self, results: List[Dict]) -> None:
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
        print()

class BoostEngineTestRunner(TestRunner):
    """Run boost engine comparative analysis"""

    CONFIGS = [
        ("Algorithmic Only (WITH Boost)", True, False, False, True),
        ("Algorithmic Only (NO Boost)", True, False, False, False),
        ("Full Pipeline (WITH Boost)", True, True, True, True),
        ("Full Pipeline (NO Boost)", True, True, True, False),
    ]

    def run(self) -> None:
        self._print_header()
        results = self._run_all_configs()
        self._print_impact_analysis(results)

    def _print_header(self) -> None:
        print_section("BOOST ENGINE COMPARATIVE ANALYSIS")
        print(f"Comparing Algorithmic Only and Full Pipeline with/without Boost Engine")
        print(f"Semantic Model: {CONFIG.semantic_model}")
        print(f"LLM Model: {CONFIG.llm_model_name}")
        mode = "TEST MODE (intent recognition only)" if CONFIG.test_mode else "Test Mode Disabled"
        print(f"Mode: {mode}")
        print(f"Edge Cases Included: {CONFIG.include_edge_cases}")
        print(f"Test Dataset Size: {len(self.test_data)} queries\n")

    def _run_all_configs(self) -> List[Dict]:
        results = []
        for name, algo, semantic, llm, boost in self.CONFIGS:
            print(f"\n{'─' * 80}\n{name}\n{'─' * 80}")

            try:
                rec = self.factory.create(algo, semantic, llm, log=False, boost=boost)
                self.factory.warmup(rec, semantic, llm, CONFIG.use_local_llm)
                ev, duration = self.run_evaluation(rec)

                results.append({
                    "name": name,
                    "boost": boost,
                    "acc": ev["accuracy"],
                    "correct": ev["correct"],
                    "unknown": sum(1 for r in ev.get("detailed_results", []) if r["predicted"] == "unknown"),
                    "total": ev["total_queries"],
                    "time": duration,
                    "qps": len(self.test_data) / duration,
                    "algo": ev.get("algo_used_count", 0),
                    "sem": ev.get("semantic_used_count", 0),
                    "llm": ev.get("llm_used_count", 0),
                    "high_conf": ev.get("high_confidence_count", 0),
                    "medium_conf": ev.get("medium_confidence_count", 0),
                    "low_conf": ev.get("low_confidence_count", 0),
                    "algo_acc": ev.get("algo_accuracy", 0),
                    "sem_acc": ev.get("semantic_accuracy", 0),
                    "llm_acc": ev.get("llm_accuracy", 0)
                })

                print(f"✓ Accuracy: {ev['accuracy']:.2%} ({ev['correct']}/{ev['total_queries']} correct)")
                print(f"  Time: {format_time(duration)} | Avg: {format_time(duration/len(self.test_data))} | "
                      f"{len(self.test_data)/duration:.1f} q/s")
                print(f"  Layers Used - Algo: {ev.get('algo_used_count', 0)}, "
                      f"Semantic: {ev.get('semantic_used_count', 0)}, LLM: {ev.get('llm_used_count', 0)}")

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


def main():
    parser = argparse.ArgumentParser(description="Test Intent Recognizer")
    parser.add_argument("--c", action="store_true", help="Run comparative analysis")
    parser.add_argument("--b", action="store_true", help="Run boost engine comparative analysis")
    args = parser.parse_args()

    try:
        if args.b:
            BoostEngineTestRunner().run()
        elif args.c:
            ComparativeTestRunner().run()
        else:
            ComprehensiveTestRunner().run()
    except FileNotFoundError:
        print("\nERROR: intent_patterns.json not found in utils/\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user\n")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}\n")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()