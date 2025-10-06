"""
Test script for Intent Recognizer
Runs test dataset and displays statistics
Activate VENV first: source .venv/bin/activate
Run: python -m testData.test_intent_recognizer
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from intentRecognizer.intent_recognizer import IntentRecognizer
from testData.test_data import get_test_dataset


def run_comprehensive_test():
    """Run comprehensive tests on the intent recognizer"""

    # Recognizer with logging enabled
    recognizer = IntentRecognizer(enable_logging=True, min_confidence=0.5)
    print()

    # Comprehensive test dataset
    test_data = get_test_dataset()

    print(f"Running tests on {len(test_data)} queries...")
    print()

    evaluation = recognizer.evaluate(test_data)

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Accuracy: {evaluation['accuracy']:.2%}")
    print(f"Total Queries: {evaluation['total_queries']}")
    print(f"Correct: {evaluation['correct']}")
    print(f"Incorrect: {evaluation['incorrect']}")
    print()

    print("CONFIDENCE DISTRIBUTION")
    print("-" * 80)
    print(f"High Confidence (≥0.8): {evaluation['high_confidence_count']} queries")
    print(f"  Accuracy: {evaluation['high_confidence_accuracy']:.2%}")
    print(f"Medium Confidence (0.6-0.8): {evaluation['medium_confidence_count']} queries")
    print(f"Low Confidence (<0.6): {evaluation['low_confidence_count']} queries")
    print()

    # Statistics
    stats = recognizer.get_statistics()

    print("PROCESSING STATISTICS")
    print("-" * 80)
    print(f"Total Queries Processed: {stats['total_queries_processed']}")
    print(f"Average Confidence: {stats['average_confidence']:.3f}")
    print()

    print("INTENT DISTRIBUTION")
    print("-" * 80)
    for intent, count in sorted(stats['intent_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_queries_processed']) * 100
        print(f"  {intent:25s}: {count:3d} ({percentage:5.1f}%)")
    print()

    incorrect_results = [r for r in evaluation['detailed_results'] if not r['correct']]

    if incorrect_results:
        print("\n" + "=" * 80)
        print("Predictions (Incorrect)")
        print("=" * 80)
        for i, result in enumerate(incorrect_results, 1):
            print(f"\n{i}. Query: '{result['query']}'")
            print(f"   Expected: {result['expected']}")
            print(f"   Predicted: {result['predicted']}")
            print(f"   Confidence: {result['confidence']:.3f}")
    else:
        print("\n✓ All predictions were correct!")

    print()

    return evaluation, stats


if __name__ == "__main__":
    try:
        evaluation, stats = run_comprehensive_test()
    except FileNotFoundError as e:
        print(f"Error: Could not find intent_patterns.json file")
        print(f"Please ensure the file exists in the utils directory")
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback

        traceback.print_exc()