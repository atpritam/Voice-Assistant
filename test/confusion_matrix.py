"""
Confusion matrix test runner
Generates confusion matrix and error analysis for intent recognition
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test.common import (
    CONFIG,
    RecognizerFactory,
    ResultPrinter,
    print_section
)


class ConfusionMatrixAnalyzer:
    """Generate and analyze confusion matrix for intent recognition"""

    def __init__(self, intent_names):
        """Initialize with list of intent names"""
        self.intent_names = sorted(intent_names)
        self.intent_to_idx = {intent: idx for idx, intent in enumerate(self.intent_names)}
        self.matrix = np.zeros((len(self.intent_names), len(self.intent_names)), dtype=int)

    def add_prediction(self, actual: str, predicted: str):
        """Add a prediction to the confusion matrix"""
        if actual in self.intent_to_idx and predicted in self.intent_to_idx:
            actual_idx = self.intent_to_idx[actual]
            predicted_idx = self.intent_to_idx[predicted]
            self.matrix[actual_idx][predicted_idx] += 1

    def get_matrix(self):
        """Get the confusion matrix as numpy array"""
        return self.matrix

    def print_matrix(self):
        """Print confusion matrix"""
        print("\n" + "="*80)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*80)
        print("\nRows = Actual Intent, Columns = Predicted Intent\n")

        max_intent_len = max(len(intent) for intent in self.intent_names)
        col_width = max(max_intent_len, 8)

        print(" " * (col_width + 2), end="")
        for intent in self.intent_names:
            print(f"{intent[:col_width]:>{col_width}}", end="  ")
        print()
        print("-" * (col_width + 2 + (col_width + 2) * len(self.intent_names)))

        # Print rows
        for i, actual_intent in enumerate(self.intent_names):
            print(f"{actual_intent:<{col_width}}  ", end="")
            for j in range(len(self.intent_names)):
                count = self.matrix[i][j]
                if i == j:
                    # correct predictions
                    print(f"\033[92m{count:>{col_width}}\033[0m", end="  ")
                elif count > 0:
                    # misclassifications
                    print(f"\033[91m{count:>{col_width}}\033[0m", end="  ")
                else:
                    print(f"{count:>{col_width}}", end="  ")
            print()

        print("\n" + "="*80)

    def print_statistics(self):
        """Print detailed statistics from confusion matrix"""

        total_predictions = np.sum(self.matrix)
        correct_predictions = np.trace(self.matrix)
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        print(f"\nOverall Accuracy: {overall_accuracy:.2%} ({correct_predictions}/{total_predictions})")
        print("\nPer-Intent Statistics:")
        print("-" * 80)
        print(f"{'Intent':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)

        for i, intent in enumerate(self.intent_names):
            # True Positives: diagonal element
            tp = self.matrix[i][i]

            # False Positives: sum of column excluding diagonal
            fp = np.sum(self.matrix[:, i]) - tp

            # False Negatives: sum of row excluding diagonal
            fn = np.sum(self.matrix[i, :]) - tp

            # Support: total actual instances
            support = np.sum(self.matrix[i, :])

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{intent:<20} {precision:<12.2%} {recall:<12.2%} {f1:<12.2%} {support:<10}")

        print("-" * 80)

    def print_misclassification_analysis(self):
        """Analyze and print common misclassifications"""
        print("\n" + "="*80)

        misclassifications = []

        for i, actual in enumerate(self.intent_names):
            for j, predicted in enumerate(self.intent_names):
                if i != j and self.matrix[i][j] > 0:
                    misclassifications.append((
                        actual, predicted, self.matrix[i][j]
                    ))

        if not misclassifications:
            print("\n✓ No misclassifications detected!\n")
            return

        misclassifications.sort(key=lambda x: x[2], reverse=True)

        print(f"\nTop Misclassifications (Total: {len(misclassifications)} types):\n")
        print(f"{'Actual Intent':<20} {'Predicted As':<20} {'Count':<10} {'% of Actual':<15}")
        print("-" * 70)

        for actual, predicted, count in misclassifications[:15]:  # Top 15
            actual_total = np.sum(self.matrix[self.intent_to_idx[actual], :])
            percentage = (count / actual_total * 100) if actual_total > 0 else 0
            print(f"{actual:<20} {predicted:<20} {count:<10} {percentage:<15.1f}")


class ConfusionMatrixTestRunner:
    """Run confusion matrix analysis"""

    def __init__(self, test_data=None):
        """Initialize test runner"""
        import logging
        from test.data import get_test_dataset
        from utils.logger import setup_logging

        setup_logging(level=logging.WARNING)
        self.factory = RecognizerFactory()
        self.printer = ResultPrinter()
        self.test_data = test_data if test_data is not None else get_test_dataset(include_edge_cases=CONFIG.include_edge_cases)

    def run(self) -> ConfusionMatrixAnalyzer:
        """
        Run confusion matrix analysis

        Returns:
            ConfusionMatrixAnalyzer instance with results
        """
        self._print_header()

        # Initialize recognizer
        try:
            recognizer = self.factory.create(
                CONFIG.enable_algo,
                CONFIG.enable_semantic,
                CONFIG.enable_llm,
                log=False
            )
            self.factory.warmup(
                recognizer,
                CONFIG.enable_semantic,
                CONFIG.enable_llm,
                CONFIG.use_local_llm
            )
        except Exception as e:
            print(f"\n✗ Failed to initialize recognizer: {e}")
            return None

        all_intents = set()
        for _, intent in self.test_data:
            all_intents.add(intent)

        # Initialize confusion matrix
        cm = ConfusionMatrixAnalyzer(all_intents)

        # Run predictions
        print(f"Test Dataset Size: {len(self.test_data)} queries\n")
        for query, expected_intent in self.test_data:
            result = recognizer.recognize_intent(query)
            cm.add_prediction(expected_intent, result.intent)

        # Display results
        cm.print_matrix()
        cm.print_statistics()
        cm.print_misclassification_analysis()

        return cm

    def _print_header(self) -> None:
        """Print test header with configuration info"""
        print_section("CONFUSION MATRIX TEST")
        self.printer.print_config_info(
            CONFIG.enable_algo,
            CONFIG.enable_semantic,
            CONFIG.enable_llm
        )


def run_confusion_matrix_test(
    enable_algorithmic: bool = True,
    enable_semantic: bool = True,
    enable_llm: bool = True,
    use_boost_engine: bool = True,
    include_edge_cases: bool = True,
    use_local_llm: bool = True,
    llm_model: str = "llama3.2:3b-instruct-q4_K_M",
    test_data=None
):
    """
    Run test and generate confusion matrix (backward compatibility wrapper)

    Args:
        enable_algorithmic: Enable algorithmic layer
        enable_semantic: Enable semantic layer
        enable_llm: Enable LLM layer
        use_boost_engine: Enable boost engine
        include_edge_cases: Include edge cases in test
        use_local_llm: Use local Ollama LLM instead of OpenAI API
        llm_model: LLM model to use
        test_data: Optional custom test dataset (list of tuples)
    """
    CONFIG.enable_algo = enable_algorithmic
    CONFIG.enable_semantic = enable_semantic
    CONFIG.enable_llm = enable_llm
    CONFIG.use_boost_engine = use_boost_engine
    CONFIG.include_edge_cases = include_edge_cases
    CONFIG.use_local_llm = use_local_llm
    if llm_model:
        CONFIG.llm_model = llm_model

    runner = ConfusionMatrixTestRunner(test_data=test_data)
    return runner.run()