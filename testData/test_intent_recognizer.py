"""
Test script for Intent Recognizer
Run:
  python -m testData.test_intent_recognizer
  python -m testData.test_intent_recognizer --comparative
"""

import sys, os, time, argparse, traceback, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from intentRecognizer.intent_recognizer import IntentRecognizer
from testData.test_data import get_test_dataset

# === CONFIG ===
utils_dir = os.path.join(os.path.dirname(__file__), '..', 'utils')
PATTERN_FILE = os.path.join(utils_dir, 'intent_patterns.json')
MIN_CONFIDENCE = 0.5
SEMANTIC_MODEL = "all-MiniLM-L6-v2"

# LLM Configuration
USE_LOCAL_LLM = True  # Set to True for Ollama, False for OpenAI
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M" if USE_LOCAL_LLM else "gpt-5-nano"
OLLAMA_BASE_URL = "http://localhost:11434"

INCLUDE_EDGE_CASES = True

# Default pipeline
ENABLE_ALGO, ENABLE_SEMANTIC, ENABLE_LLM = True, True, True
USE_BOOST_ENGINE = True # Contextual boosting for Algorithmic layer
THRESH_ALGO, THRESH_SEMANTIC = 0.6, 0.5

# CRITICAL: Enable TEST_MODE for pure intent recognition testing
TEST_MODE = True

logging.basicConfig(
    level=logging.INFO,
    format='- %(name)s - %(levelname)s - %(message)s',
    force=True
)

def format_time(s):
    return f"{s*1000:.1f}ms" if s < 1 else f"{s:.2f}s"


def describe_pipeline(a, s, l):
    llm_label = "LLM (Ollama)" if USE_LOCAL_LLM else "LLM (OpenAI)"
    layers = []
    if a:
        layers.append("Algorithmic")
    if s:
        layers.append("Semantic")
    if l:
        layers.append(llm_label)
    return " -> ".join(layers) or "NO LAYERS"


def init_recognizer(a=True, s=True, l=True, log=True, ta=THRESH_ALGO, ts=THRESH_SEMANTIC):
    return IntentRecognizer(
        enable_logging=log,
        enable_algorithmic=a, enable_semantic=s, enable_llm=l,
        use_boost_engine=USE_BOOST_ENGINE,
        algorithmic_threshold=ta or 0.6, semantic_threshold=ts or 0.5,
        semantic_model=SEMANTIC_MODEL, llm_model=LLM_MODEL,
        min_confidence=MIN_CONFIDENCE, patterns_file=PATTERN_FILE,
        test_mode=TEST_MODE,
        use_local_llm=USE_LOCAL_LLM,
        ollama_base_url=OLLAMA_BASE_URL
    )

def warmup_pipeline(recognizer):
    if ENABLE_SEMANTIC or (ENABLE_LLM and USE_LOCAL_LLM) :
        recognizer.recognize_intent("sample text", [])

def run_comprehensive_test():
    print(f"\nPipeline: {describe_pipeline(ENABLE_ALGO, ENABLE_SEMANTIC, ENABLE_LLM)}")
    if ENABLE_SEMANTIC:
        print(f"Semantic Model: {SEMANTIC_MODEL}")
    if ENABLE_LLM:
        print(f"LLM Model: {LLM_MODEL}")
    print(f"Mode: {'TEST MODE (intent recognition only)' if TEST_MODE else 'PRODUCTION MODE (with responses)'}\n")
    print(f"Boost Engine: {USE_BOOST_ENGINE}")
    print(f"Edge Cases Included: {INCLUDE_EDGE_CASES}\n")

    test_data = get_test_dataset(include_edge_cases=INCLUDE_EDGE_CASES)
    print(f"Running tests on {len(test_data)} queries...\n")

    try:
        recognizer = init_recognizer(ENABLE_ALGO, ENABLE_SEMANTIC, ENABLE_LLM, log=True)
        warmup_pipeline(recognizer)
    except Exception as e:
        print(f"INIT ERROR: {e}")
        traceback.print_exc()
        return None, None

    start = time.time()
    ev = recognizer.evaluate(test_data)
    dur = time.time() - start

    print("\nOVERALL RESULTS\n" + "-" * 80)
    print(f"Accuracy: {ev['accuracy']:.2%}")
    print(f"Correct: {ev['correct']} / {ev['total_queries']}")
    print(f"Avg Query Time: {format_time(dur / len(test_data))}")
    print(f"Queries/s: {len(test_data) / dur:.1f}\n")

    print("LAYER USAGE\n" + "-" * 80)
    for layer, key in [("Algorithmic", "algo"), ("Semantic", "semantic"), ("LLM", "llm")]:
        if ev.get(f"{key}_used_count"):
            c = ev[f"{key}_used_count"]
            print(f"{layer:<12}: {c:3d} ({c/len(test_data)*100:5.1f}%)  Acc: {ev[f'{key}_accuracy']:.2%}")

    print("\nCONFIDENCE LEVELS\n" + "-" * 80)
    for level, rng in [("High (≥0.8)", "high"), ("Medium (0.6-0.8)", "medium"), ("Low (<0.6)", "low")]:
        c = ev[f"{rng}_confidence_count"]
        print(f"{level:<22}: {c:3d} ({c/len(test_data)*100:5.1f}%)")

    wrong = [r for r in ev["detailed_results"] if not r["correct"]]
    if not wrong:
        print("\n✓ ALL CORRECT!\n")
    else:
        print("\nINCORRECT PREDICTIONS\n" + "-" * 80)
        for i, r in enumerate(wrong, 1):
            print(f"\n{i}. '{r['query']}' -> {r['predicted']} (exp: {r['expected']}, conf: {r['confidence']:.2f}, layer: {r['layer_used']})")

    return ev, recognizer.get_statistics()


def run_comparative_analysis():
    test_data = get_test_dataset(include_edge_cases=INCLUDE_EDGE_CASES)
    print(f"\nTesting multiple pipeline configurations for comparative results")
    print(f"Semantic Model: {SEMANTIC_MODEL}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Mode: {'TEST MODE (intent recognition only)' if TEST_MODE else 'Test Mode Disabled'}")
    print(f"Boost Engine: {USE_BOOST_ENGINE}")
    print(f"Edge Cases Included: {INCLUDE_EDGE_CASES}\n")
    print(f"\nRunning tests on {len(test_data)} queries...\n")

    configs = [
        ("Full Pipeline", True, True, True),
        ("Algorithmic -> Semantic", True, True, False),
        ("Algorithmic -> LLM", True, False, True),
        ("Semantic -> LLM", False, True, True),
        ("Algorithmic Only", True, False, False),
        ("Semantic Only", False, True, False),
        # ("LLM Only", False, False, True)
    ]

    results = []
    for name, a, s, l in configs:
        print(f"\n{'─' * 80}\n{name}\n{'─' * 80}")
        try:
            rec = init_recognizer(a, s, l, log=False)
            warmup_pipeline(rec)
            t0 = time.time()
            ev = rec.evaluate(test_data)
            t = time.time() - t0
            results.append({
                "name": name,
                "acc": ev["accuracy"],
                "correct": ev["correct"],
                "unknown": sum(1 for r in ev["detailed_results"] if r["predicted"] == "unknown"),
                "total": ev["total_queries"],
                "time": t,
                "qps": len(test_data) / t,
                "algo": ev.get("algo_used_count", 0),
                "sem": ev.get("semantic_used_count", 0),
                "llm": ev.get("llm_used_count", 0)
            })
            print(
                f"✓ Accuracy: {ev['accuracy']:.2%} ({ev['correct']}/{ev['total_queries']} correct)\nTime: {format_time(t)}")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results.append(
                {"name": name, "acc": 0, "correct": 0, "total": len(test_data), "time": 0, "qps": 0, "algo": 0,
                 "sem": 0, "llm": 0})

    print("\n" + "=" * 80 + "\n")
    print("\nPipeline Comparison\n" + "-"*80)
    print(f"{'Configuration':<25} {'Accuracy':<10} {'Total Time':<12} {'Avg Time':<10} {'Q/s':<10}")
    print("-" * 80)
    for r in results:
        avg_time = r['time'] / len(test_data) if len(test_data) else 0
        print(
            f"{r['name']:<25} {r['acc']:>8.2%}  {format_time(r['time']):>10}  {format_time(avg_time):>8}  {r['qps']:>8.1f}")

    print("\nLAYER USAGE COMPARISON\n" + "-"*80)
    print(f"{'Configuration':<25} {'Algo':<8} {'Semantic':<10} {'LLM':<8} {'Unrecognized Intent':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<25} {r['algo']:>6}  {r['sem']:>8}  {r['llm']:>6} {r['unknown']:>8}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Intent Recognizer")
    parser.add_argument("--comparative", action="store_true", help="Run comparative analysis")
    args = parser.parse_args()

    try:
        if args.comparative:
            run_comparative_analysis()
        else:
            run_comprehensive_test()
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