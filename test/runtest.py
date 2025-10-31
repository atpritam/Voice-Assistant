"""
Test script for Intent Recognizer
Default dataset in data.py

Run examples:
  python -m test.runtest                                    # Comprehensive test
  python -m test.runtest "where is my pizza?"               # Single query test
  python -m test.runtest -c                                 # Comparative analysis
  python -m test.runtest -b                                 # Boost engine analysis
  python -m test.runtest -mx                                # Confusion matrix
  python -m test.runtest -c --no-boost                      # Comparative without boost
  python -m test.runtest -b --no-edge                       # Boost analysis without edge cases
  python -m test.runtest --no-semantic --no-llm             # Comprehensive with only algorithmic layer
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test.comprehensive import ComprehensiveTestRunner
from test.comparative import ComparativeTestRunner
from test.boost_analysis import BoostEngineTestRunner
from test.confusion_matrix import run_confusion_matrix_test
from test.common import CONFIG, prompt_for_intent, create_single_query_dataset
from utils.logger_config import setup_logging


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Intent Recognizer Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m test.runtest                       # Comprehensive test with all layers
  python -m test.runtest -c                    # Compare all pipeline configurations
  python -m test.runtest -b                    # Analyze boost engine impact
  python -m test.runtest -mx                   # Generate confusion matrix
  python -m test.runtest --no-semantic         # Test without semantic layer
  python -m test.runtest -c --no-boost         # Comparative test without boost engine
        """
    )

    # Test mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-c", "--comparative", action="store_true",
                      help="Run comparative analysis across multiple configurations")
    mode.add_argument("-b", "--boost", action="store_true",
                      help="Run boost engine comparative analysis")
    mode.add_argument("-mx", "--matrix", action="store_true",
                      help="Generate confusion matrix and error analysis")

    # Configuration arguments
    parser.add_argument("--no-algo", action="store_true",
                        help="Disable algorithmic layer (comprehensive test only)")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Disable semantic layer (comprehensive test only)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM layer (comprehensive test only)")
    parser.add_argument("--no-boost", action="store_true",
                        help="Disable boost engine")
    parser.add_argument("--no-edge", action="store_true",
                        help="Exclude edge cases from test dataset")

    # LLM backend selection
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--ollama", action="store_true",
                           help="Use local Ollama LLM (default)")
    llm_group.add_argument("--openai", action="store_true",
                           help="Use OpenAI API")

    # Single query test
    parser.add_argument("query", nargs='?', default=None,
                        help="Single query to test")

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate argument combinations and show warnings

    Args:
        args: Parsed arguments

    Raises:
        SystemExit: If invalid argument combination detected
    """
    if args.boost and args.no_boost:
        print("\nERROR: Cannot use -b (boost engine test) with --no-boost")
        print("The boost engine test compares performance WITH and WITHOUT boost engine.\n")
        sys.exit(1)

    if args.boost and (args.no_algo or args.no_semantic or args.no_llm):
        print("\nWARNING: Pipeline configuration flags (--no-algo, --no-semantic, --no-llm) "
              "are ignored in boost engine test mode.")
        print("Boost engine test runs predefined configurations.\n")

    if args.comparative and (args.no_algo or args.no_semantic or args.no_llm):
        print("\nWARNING: Pipeline configuration flags (--no-algo, --no-semantic, --no-llm) "
              "are ignored in comparative test mode.")
        print("Comparative test runs all pipeline configurations.\n")


def configure_from_args(args: argparse.Namespace) -> None:
    """
    Configure global CONFIG from parsed arguments

    Args:
        args: Parsed arguments
    """
    # LLM backend
    CONFIG.use_local_llm = False if args.openai else True

    # edge cases
    CONFIG.include_edge_cases = not args.no_edge

    # Set configuration based on test mode
    if args.boost:
        pass
    elif args.comparative:
        CONFIG.use_boost_engine = not args.no_boost
    elif args.matrix:
        CONFIG.enable_algo = not args.no_algo
        CONFIG.enable_semantic = not args.no_semantic
        CONFIG.enable_llm = not args.no_llm
        CONFIG.use_boost_engine = not args.no_boost
    else:
        CONFIG.enable_algo = not args.no_algo
        CONFIG.enable_semantic = not args.no_semantic
        CONFIG.enable_llm = not args.no_llm
        CONFIG.use_boost_engine = not args.no_boost


def main():
    """Main entry point"""
    args = parse_arguments()
    validate_arguments(args)
    configure_from_args(args)

    try:
        if args.matrix and args.query:
            print("\nERROR: Confusion matrix mode does not support single query testing")
            sys.exit(1)

        # single query mode
        test_data = None
        if args.query:
            setup_logging(level=logging.DEBUG)
            expected_intent = prompt_for_intent(args.query)
            test_data = create_single_query_dataset(args.query, expected_intent)

        # Route to appropriate test runner
        if args.matrix:
            run_confusion_matrix_test(
                enable_algorithmic=CONFIG.enable_algo,
                enable_semantic=CONFIG.enable_semantic,
                enable_llm=CONFIG.enable_llm,
                use_boost_engine=CONFIG.use_boost_engine,
                include_edge_cases=CONFIG.include_edge_cases,
                use_local_llm=CONFIG.use_local_llm,
                llm_model=CONFIG.llm_model_name
            )
        elif args.boost:
            BoostEngineTestRunner(custom_data=test_data).run()
        elif args.comparative:
            ComparativeTestRunner(custom_data=test_data).run()
        else:
            ComprehensiveTestRunner(custom_data=test_data).run()

    except FileNotFoundError:
        print("\nERROR: intent_patterns.json not found in utils/\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user\n")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()