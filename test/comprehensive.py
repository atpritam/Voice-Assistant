"""
Comprehensive test runner
Runs a single configuration evaluation with detailed results
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, Optional, Tuple
from test.common import (
    CONFIG,
    BaseTestRunner,
    ResultPrinter,
    print_section
)


class ComprehensiveTestRunner(BaseTestRunner):
    """Run comprehensive single-pipeline test"""

    def __init__(self, custom_data=None):
        """Initialize test runner"""
        super().__init__(custom_data)
        self.printer = ResultPrinter()

    def run(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Run comprehensive test

        Returns:
            Tuple of (evaluation_results, statistics) or (None, None) on error
        """
        print_section("COMPREHENSIVE TEST")
        self.printer.print_config_info(
            CONFIG.enable_algo,
            CONFIG.enable_semantic,
            CONFIG.enable_llm
        )
        print(f"Test Dataset Size: {len(self.test_data)} queries\n")

        # Initialize recognizer
        try:
            recognizer = self.factory.create(
                CONFIG.enable_algo,
                CONFIG.enable_semantic,
                CONFIG.enable_llm
            )
            self.factory.warmup(
                recognizer,
                CONFIG.enable_semantic,
                CONFIG.enable_llm,
                CONFIG.use_local_llm
            )
        except Exception as e:
            print(f"INIT ERROR: {e}")
            traceback.print_exc()
            return None, None

        # Run evaluation
        ev, duration = self._run_evaluation(recognizer)
        stats = recognizer.get_statistics()

        # Print results
        self.printer.print_overall_results(ev, duration, len(self.test_data))
        self.printer.print_layer_usage(ev, len(self.test_data))
        self.printer.print_token_usage(stats, len(self.test_data))
        self.printer.print_confidence_levels(ev, len(self.test_data))
        self.printer.print_incorrect_predictions(ev)

        return ev, stats