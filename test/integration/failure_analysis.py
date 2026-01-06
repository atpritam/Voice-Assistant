"""
Failure Analysis Test Runner
Provides detailed logging and debugging information for incorrect predictions
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Dict, List, Tuple, Any
from .common import (
    CONFIG,
    BaseTestRunner,
    ResultPrinter,
    print_section
)
from utils.logger import CleanFormatter


class LogCapture(logging.Handler):
    """Logging handler to capture logs"""
    def __init__(self):
        super().__init__()
        self.logs = []
        
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
    
    def get_logs(self):
        return self.logs
    
    def clear(self):
        self.logs = []


class FailureAnalysisRunner(BaseTestRunner):
    """Run comprehensive test with detailed failure analysis output"""

    def __init__(self, custom_data=None):
        super().__init__(custom_data, log_level=logging.DEBUG)
        self.failed_queries: List[Dict[str, Any]] = []
        self.log_capture = None
        self.total_queries = 0

    def run(self) -> None:
        """
        Run failure analysis test
        
        Executes the pipeline and collects detailed information
        about incorrect predictions with complete score breakdowns
        """
        print_section("INCORRECT PREDICTIONS WITH DETAILED LOGGING")
        
        # Initialize recognizer
        recognizer = self.factory.create(
            CONFIG.enable_algo,
            CONFIG.enable_semantic,
            CONFIG.enable_llm
        )
        
        printer = ResultPrinter()
        printer.print_config_info(
            CONFIG.enable_algo,
            CONFIG.enable_semantic,
            CONFIG.enable_llm
        )
        
        print(f"Test Dataset Size: {len(self.test_data)} queries\n")
        
        self.factory.warmup(
            recognizer,
            CONFIG.enable_semantic,
            CONFIG.enable_llm,
            CONFIG.llm_model
        )
        
        self.log_capture = LogCapture()
        self.log_capture.setFormatter(CleanFormatter())
        
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        
        for handler in original_handlers:
            root_logger.removeHandler(handler)
        root_logger.addHandler(self.log_capture)
        
        self.total_queries = len(self.test_data)
        
        for query, expected_intent in self.test_data:
            self.log_capture.clear()
            result = recognizer.recognize_intent(query)
            is_correct = result.intent == expected_intent
            
            if not is_correct:
                print_section(
                    f"Query: '{query}'\nExpected Intent: {expected_intent}",
                    char="-"
                )

                for log in self.log_capture.get_logs():
                    print(log)
                
                print("\nFINAL RESULT:")
                print(f"  Predicted: {result.intent}")
                print(f"  Confidence: {result.confidence:.3f}")
                print(f"  Layer: {result.layer_used}")
                print(f"  Match: ✗ INCORRECT")
                
                if result.score_breakdown:
                    print("\n  Score Breakdown:")
                    self._print_score_breakdown(result.score_breakdown, indent=4)
                
                print("=" * 80)
                print()
                
                self.failed_queries.append({
                    'query': query,
                    'expected': expected_intent,
                    'predicted': result.intent,
                    'confidence': result.confidence,
                    'layer_used': result.layer_used
                })

        root_logger.removeHandler(self.log_capture)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        
        self._print_summary()

    def _print_score_breakdown(self, breakdown: Dict[str, Any], indent: int = 4) -> None:
        """Recursively print score breakdown with proper formatting"""
        spaces = " " * indent
        
        for key, value in breakdown.items():
            if isinstance(value, dict):
                if key == 'all_similarities':
                    formatted_dict = ', '.join([f"'{k}': {v:.3f}" if isinstance(v, float) else f"'{k}': {v}" for k, v in value.items()])
                    print(f"{spaces}{key}: {{{formatted_dict}}}")
                else:
                    print(f"{spaces}{key}:")
                    self._print_score_breakdown(value, indent + 2)
            elif isinstance(value, list):
                # top_k_patterns, top_k_scores
                if len(value) > 0 and isinstance(value[0], (int, float)):
                    formatted = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in value]
                    print(f"{spaces}{key}: [{', '.join(formatted)}]")
                else:
                    print(f"{spaces}{key}: {value}")
            elif isinstance(value, float):
                print(f"{spaces}{key}: {value:.3f}")
            elif isinstance(value, bool):
                print(f"{spaces}{key}: {value}")
            else:
                print(f"{spaces}{key}: {value}")

    def _print_summary(self) -> None:
        """Print summary of incorrect predictions"""
        print_section("SUMMARY")
        
        correct_queries = self.total_queries - len(self.failed_queries)
        accuracy = (correct_queries / self.total_queries * 100) if self.total_queries > 0 else 0
        
        print(f"Test Dataset Size: {self.total_queries} queries")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Correct: {correct_queries} / {self.total_queries}\n")
        
        if not self.failed_queries:
            print("✓ ALL PREDICTIONS CORRECT!\n")
            return
        
        print(f"Incorrect Predictions ({len(self.failed_queries)}):")
        for result in self.failed_queries:
            print(f"  '{result['query']}'")
            print(f"    Expected: {result['expected']}, Got: {result['predicted']} "
                  f"({result['layer_used']}, conf={result['confidence']:.2f})")
        print()