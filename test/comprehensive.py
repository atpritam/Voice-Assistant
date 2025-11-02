"""
Comprehensive test runner
Runs a single configuration evaluation with detailed results
"""

import sys
import os
import time
import traceback
import logging
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from intentRecognizer.intent_recognizer import IntentRecognizer
from test.data import get_test_dataset
from test.common import (
    CONFIG,
    RecognizerFactory,
    ResultAnalyzer,
    print_section
)
from utils.logger_config import setup_logging


class ComprehensiveTestRunner:
    """Run comprehensive single-pipeline test"""

    def __init__(self, custom_data=None):
        """Initialize test runner"""
        if not logging.getLogger().handlers:
            setup_logging(level=logging.INFO)
        self.factory = RecognizerFactory()
        self.analyzer = ResultAnalyzer()
        self.test_data = custom_data if custom_data is not None else get_test_dataset(include_edge_cases=CONFIG.include_edge_cases)

    def run(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Run comprehensive test

        Returns:
            Tuple of (evaluation_results, statistics) or (None, None) on error
        """
        print_section("COMPREHENSIVE TEST")
        self.analyzer.print_config_info(
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
        self.analyzer.print_overall_results(ev, duration, len(self.test_data))
        self.analyzer.print_layer_usage(ev, len(self.test_data))
        self.analyzer.print_token_usage(stats, len(self.test_data))
        self.analyzer.print_confidence_levels(ev, len(self.test_data))
        self.analyzer.print_incorrect_predictions(ev)

        return ev, stats

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