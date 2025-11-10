# Voice Response System with Multi-Layer Intent Recognition Pipeline

A **hybrid intent recognition system** combining classical NLP pattern matching, neural semantic embeddings, and large language models for robust conversational AI. The system has a novel three-layer architecture that balances speed, accuracy, and interpretability.

**Use Case**: Domain-agnostic customer service

**Current Domain**: Pizza Restaurant

**Research Focus**: Hybrid NLP approach combining rule-based, embedding-based, and generative techniques

### Key Features

- **Multi-Layer Intent Recognition Pipeline**
  - Layer 1 - Pattern Matching: Algorithmic recognition using Levenshtein edit distance, Jaccard similarity, TF-IDF weighted indexing, and synonym expansion
  - Layer 2 - Neural Embeddings: Semantic similarity using Sentence Transformer embeddings (SBERT) for context-aware matching
  - Layer 3 - Generative AI: LLM-based classification with conversation history awareness (Ollama local/cloud models)
  - Boost Engine: Domain-specific contextual rules applying negative sentiment detection, entity keyword matching, and co-occurrence patterns

- **End-to-End Voice Interaction**
  - **ASR (Automatic Speech Recognition)**: OpenAI Whisper with audio preprocessing for robust transcription
  - **TTS (Text-to-Speech)**: Coqui TTS neural vocoder (VITS architecture)
  - **Full conversational loop**: Speech → Text → Intent → Response → Speech

## Quick Start

Get the system running in 3 steps:

### 1. Install Dependencies

```bash
# Install system dependencies
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv ffmpeg espeak-ng

# Clone and setup
git clone https://github.com/atpritam/Voice-Assistant.git
cd Voice-Assistant
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. LLM Backend

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

**Option A: Local Ollama LLM (Recommended - Free & Private)**

```bash
# Pull model and start service
ollama pull llama3.2:3b-instruct-q4_K_M
ollama serve &
```

**Option B: Cloud Ollama Models (Faster setup, requires Ollama Signin)**

```bash
ollama pull gpt-oss:120b-cloud
ollama ollama signin
```
Sign in to Ollama with the prompted link.

Edit `app.py` and set `USE_LOCAL_LLM = False`

### 3. Run the Application

```bash
# Create .env file with generated SECRET_KEY
echo "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > .env

# Start the application
python app.py
```

Access the web interface at `http://localhost:5000`

### Web Interface

<img src="static/assets/screenshot.png" alt="Voice System Interface" width="800">

The web interface provides a simple and intuitive way to interact with the voice assistant, showing real-time transcription and system responses.

## System Architecture

### Intent Recognition Pipeline

The system implements a cascading three-layer architecture where each layer applies progressively sophisticated techniques until confident intent recognition is achieved:

1. **Algorithmic Layer**: Fast pattern matching using NLP preprocessing, TF-IDF candidate ranking, synonym expansion, and fuzzy string similarity
2. **Semantic Layer**: Neural semantic matching using transformer-based sentence embeddings (SBERT) for distributional semantic similarity
3. **LLM Layer**: Fallback to large language models for ambiguous queries and natural response generation

Each layer can be independently enabled or disabled with configurable confidence thresholds.

```
User Voice Input
     ↓
[ASR - Whisper]
     ↓
Text Query
     ↓
┌─────────────────────────┐
│ Intent Recognition      │
│                         │
│ 1. Algorithmic (fast)   │ ← 80% of queries
│    ├─ Pattern matching  │
│    ├─ Levenshtein      │
│    └─ Boost Engine      │
│         ↓               │
│ 2. Semantic (accurate)  │ ← 14% of queries
│    └─ Neural embeddings │
│         ↓               │
│ 3. LLM (fallback)       │ ← 6% of queries
│    └─ Ollama/Cloud     │
└─────────────────────────┘
     ↓
Response Generation
     ↓
[TTS - Coqui VITS]
     ↓
Voice Output
```

### Full Pipeline vs. LLM-Only

**Intent Recognition Performance Comparison (400 queries, test mode):**

| Configuration                    | Accuracy | Latency | Q/s | Token Usage Δ |
|----------------------------------|----------|---------|-----|---------------|
| **Full Pipeline (Llama3.2 3B)**  | **98.00%** | **24.9ms** | **40.2** | baseline      |
| **Full Pipeline (GPT-OSS 120B)** | **98.00%** | **91.2ms** | **11.0** | +44%          |
| LLM-Only (Llama3.2 3B, local)    | 86.00% | 262.2ms | 3.8 | +1,450%       |
| LLM-Only (GPT-OSS 120B, cloud)   | 92.75% | **1.42s** | 0.7 | +2,078%       |

See `testResults/comparativeTest/` for detailed comparative analysis.

## System Requirements

**Minimum:**
- CPU: Modern multi-core processor
- RAM: 8GB
- Storage: 6GB free space
- Python: **3.11 (Required)** - Coqui TTS limitation

**Recommended:**
- CPU: 4+ cores
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM (CUDA-compatible)
- Storage: 10GB free space

**Note**: All components work on CPU-only systems. GPU acceleration is optional but significantly improves performance.

## Configuration

### Environment Configuration

Create a `.env` file in the project root:

```env
# Required
SECRET_KEY=your_generated_secret_key_here
```

Generate a secure `SECRET_KEY`:
```bash
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

### Pipeline Configuration

Edit `app.py` to configure the intent recognition pipeline:

```python
# Layer enabling
ENABLE_ALGORITHMIC = True
ENABLE_SEMANTIC = True
ENABLE_LLM = True

# Confidence thresholds
ALGORITHMIC_THRESHOLD = 0.65
SEMANTIC_THRESHOLD = 0.5

# Model selection
SEMANTIC_MODEL = "all-MiniLM-L6-v2" # Options: all-mpnet-base-v2
USE_LOCAL_LLM = True  # True: use Ollama local LLM, False: use Ollama Cloud API
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M" # Options: "gpt-oss:120b-cloud", "gemma3:4b-it-qat"
```

### TTS Configuration

```python
ENABLE_TTS = True
TTS_MODEL = "tts_models/en/ljspeech/vits"
```

### ASR Configuration

```python
ENABLE_ASR = True
ASR_MODEL = "tiny.en"  # Options: tiny.en, base.en, small.en, medium.en, large
ENABLE_AUDIO_PREPROCESSING = True
```

## Usage

### Running the Application inside vENV

```bash
python app.py
```

Access the web interface at `http://localhost:5000`

### Using Ollama (Local LLM)

1. Install Ollama from https://ollama.ai
    ```
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Pull the desired model:
   ```bash
   ollama pull llama3.2:3b-instruct-q4_K_M
   ```
3. Ensure Ollama is running on `http://localhost:11434`
    ```bash
   ollama serve &
    ```
   ```bash
   # test local llm 
   ollama run llama3.2:3b-instruct-q4_K_M "Say hello in 5 words"
    ```
4. Set `USE_LOCAL_LLM = True` in `app.py`

## Running Tests

The project includes comprehensive test suites for the intent recognition pipeline. Tests are located in the `test/` directory.

### Basic Comprehensive Test

Run a full evaluation of the current pipeline configuration:

```bash
python -m test.runtest
```

### Comparative Analysis

Compare multiple pipeline configurations side-by-side:

```bash
python -m test.runtest -c
```

### Boost Engine Analysis

Evaluate the impact of the contextual boost engine:

```bash
python -m test.runtest -b
```

### Confusion Matrix and Error Analysis

Generate confusion matrix with per-intent metrics:

```bash
python -m test.runtest -mx
```

### Custom Test Configurations

```bash
# Test without semantic layer
python -m test.runtest --no-semantic

# Test algorithmic layer only
python -m test.runtest --no-semantic --no-llm

# Comparative test without boost engine
python -m test.runtest -c --no-boost

# Test without edge cases (standard dataset only)
python -m test.runtest --no-edge

# Single query test
python -m test.runtest "where is my pizza?" --exp delivery
```

### Unit Tests

Runs pytest-based checks for the algorithmic recognizer Similarity scoring and text processing. (77 tests)

```bash
python -m test.runtest -unit 
```

## Performance Benchmarks

All tests use semantic model `all-mpnet-base-v2` and LLM model `llama3.2:3b-instruct-q4_K_M`.

### Extended Test Dataset (400 queries - with edge cases)

Comprehensive testing with 400 queries including 105 edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 98.00% | 24.9ms | 40.2 |
| Algorithmic + Semantic | 94.75% | 5.9ms | 170.6 |
| Algorithmic + LLM | 96.50% | 55.4ms | 18.0 |
| Semantic + LLM | 93.50% | 44.4ms | 22.5 |
| Algorithmic Only | 90.25% | 3.4ms | 292.6 |
| Semantic Only | 89.25% | 18.3ms | 54.8 |
| LLM Only | 86.00% | 262.2ms | 3.8 |

### Layer Distribution (Full Pipeline)

- Algorithmic layer: 80.0% of queries (320/400)
- Semantic layer: 13.5% of queries (54/400)
- LLM layer: 6.5% of queries (26/400)

### Boost Engine Impact
Comparison with and without contextual boost rules on full pipeline (400 queries with edge cases):

| Metric              | Without Boost | With Boost | Improvement |
|---------------------|---------------|------------|-------------|
| Accuracy            | 94.75% | 98.00% | +3.25% |
| Correct Predictions | 379 | 392 | +13 |
| Query Time          | 33.4ms | 23.3ms | 30% faster |
| Algorithmic Usage   | 268 | 320 | +52 |
| Semantic Usage      | 95 | 54 | -41 |
| LLM Fallback        | 37 | 26 | -11 |

### Confusion Matrix Results (Full Pipeline - 400 queries)

#### Per-Intent Performance

| Intent | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| complaint | 100.00% | 96.43% | 98.18% | 84 |
| delivery | 94.74% | 96.43% | 95.58% | 56 |
| general | 100.00% | 100.00% | 100.00% | 22 |
| hours_location | 96.83% | 98.39% | 97.60% | 62 |
| menu_inquiry | 98.85% | 97.73% | 98.29% | 88 |
| order | 96.67% | 98.86% | 97.75% | 88 |


See `testResults/` directory for detailed analyses.

### Test Dataset Quality

The benchmark results above are validated against dataset with:

- **400 total queries** (295 normal + 105 edge cases)
- **6 intent categories**: order (88), complaint (84), menu_inquiry (88), hours_location (62), delivery (56), general (22)
- **Diversity score: 0.97/1.0** - High lexical variety, not repetitive memorization
- **Edge cases include**: Multi-intent queries, sarcasm, typos, slang, very short queries, ambiguous phrasing
- **Zero duplicates** - Unbiased evaluation

## Key Technical Features

### Algorithmic Layer

**Core Pattern Matching Algorithms:**

1. **Levenshtein Distance** (50% weight)
   - **Purpose**: Character-level string similarity
   - **Function**: Measures minimum edit operations (insertions, deletions, substitutions)

2. **Jaccard Similarity** (50% weight)
   - **Purpose**: Word-level set similarity
   - **Function**: Compares word overlap between query and pattern using set intersection/union
   - **Sub-components**:
     - Exact word overlap: 70% weight
     - Synonym-expanded overlap: 30% weight

**Additional Features:**

- **Faster Processing**: Inverted index with TF-IDF weighting for efficient candidate selection and ranking
- **Synonym Expansion**: Domain-specific synonym dictionaries for lexical variation handling
- **NLP Preprocessing**: Contraction expansion, filler word removal, and tokenization
- **N-gram Matching**: Multi-word phrase detection with contextual bonuses
- **Contextual Boost Engine**:
  - Negative sentiment keyword detection for complaint classification
  - Keyword co-occurrence patterns and contextual heuristics

### Semantic Layer

- **Transformer Embeddings**: Sentence-BERT models (MPNet, MiniLM) for semantic similarity
- **Embedding Caching**: Persistent storage for instant initialization and reduced cold-start latency
- **Batch Processing**: Optimized pattern encoding

### LLM Layer

- **Conversation Context**: Conversation history tracking for context-aware classification and response generation
- Support for both cloud and local Ollama models

## Development

### Extending to Other Domains

1. Update `utils/res_info.json` with your domain information
2. Update intent patterns in `utils/intent_patterns.json`
3. Modify linguistics in `utils/linguistic_resources.json`
4. Adjust boost rules in `intentRecognizer/boostEngine.py` if using algorithmic layer
5. Update llm prompt templates in `intentRecognizer/llm_recognizer.py`
6. Update test dataset in `test/data.py`

## License

This project is part of a Bachelor's Thesis. All rights reserved.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Coqui TTS for speech synthesis
- Sentence Transformers for semantic embeddings
- Ollama for LLM inference
