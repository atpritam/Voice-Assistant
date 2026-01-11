# Voice Response System with Multi-Layer Intent Recognition Pipeline

A **hybrid intent recognition system** combining classical NLP pattern matching, neural semantic embeddings, and large language models for robust conversational AI. The system has a novel three-layer architecture that balances speed, accuracy, and interpretability.

**Use Case**: Domain-agnostic customer service

**Current Domain**: Pizza Restaurant

**Research Focus**: Hybrid NLP approach combining rule-based, embedding-based, and generative techniques

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [System Requirements](#system-requirements)
- [Configuration](#configuration)
- [Pattern File](#intent-patterns)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Performance Benchmarks](#performance-benchmarks)
- [Key Technical Features](#key-technical-features)
- [Development](#development)

## Key Features

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
ollama signin
```
Sign in to Ollama with the link provided in the terminal.

Edit `app.py` and set `LLM_MODEL = "gpt-oss:120b-cloud"` (cloud models are auto-detected by the `-cloud` suffix)

### 3. Run the Application

```bash
# Create .env file with generated SECRET_KEY
echo "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > .env

# Start the application
python app.py
```

Access the web interface at `http://localhost:5000`

### Windows Installation

The project runs on Windows, but some dependencies (e.g., audio processing) may require additional setup. For best performance, use WSL 2 (Windows Subsystem for Linux) to follow the Ubuntu instructions above. Alternatively, install natively:

1. **Install Prerequisites**:
   - Install Python 3.11 from the official site (include PATH option).
   - Install Git from git-scm.com.
   - Install Chocolatey (package manager): Run PowerShell as Admin and execute `Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))`.
   - Install FFmpeg: `choco install ffmpeg`.
   - Install Espeak-ng (for Coqui TTS): `choco install espeak-ng`.

2. **Clone Repository**:
     ```bash
     git clone https://github.com/your-repo/Voice-Assistant.git
     cd Voice-Assistant
     ```

3. **Virtual Environment**:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```

4. **Install Dependencies**:
     ```bash
     pip install -r requirements.txt
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # For CUDA
     ip install git+https://github.com/openai/whisper.git  # Whisper
     pip install coqui-tts  # TTS (may need manual Espeak path if issues)
     ```

5. **Ollama Setup**:
     - Download and run OllamaSetup.exe from ollama.com/download/windows.
     - Pull model: `ollama pull llama3.2:3b-instruct-q4_K_M`.

6. **Run the System**:
     Access at http://127.0.0.1:5000.

**Notes**: For full Linux compatibility, enable WSL 2 via `wsl --install` in PowerShell, install Ubuntu, and follow Linux steps. 

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
│ 1. Algorithmic (fast)   │ ← 72% of queries
│    ├─ Pattern matching  │
│    ├─ Levenshtein/Jaccard│
│    └─ Boost Engine      │
│         ↓               │
│ 2. Semantic (accurate)  │ ← 18% of queries
│    └─ Neural embeddings │
│         ↓               │
│ 3. LLM (fallback)       │ ← 10% of queries
│    └─ Ollama/Cloud       │
└─────────────────────────┘
     ↓
Response Generation
     ↓
[TTS - Coqui VITS]
     ↓
Voice Output
```

### Full Pipeline vs. LLM-Only

**Intent Recognition Performance Comparison (600 queries, test mode):**

| Configuration                    | Accuracy | Latency | Q/s | Token Usage Δ |
|----------------------------------|----------|---------|-----|---------------|
| **Full Pipeline (Llama3.2 3B)**  | **97.50%** | **31.4ms** | **31.8** | baseline      |
| **Full Pipeline (GPT-OSS 120B)** | **97.50%** | **106.8ms** | **9.4** | +53%          |
| LLM-Only (Llama3.2 3B, local)    | 87.17% | 278.9ms | 3.6 | +981%       |
| LLM-Only (GPT-OSS 120B, cloud)   | 94.83% | **1.03s** | 1.0 | +1,524%       |

\*GPT-OSS 120B is the latest Open Source GPT model released by OpenAI on August 5, 2025.

See `testResults/comparativeTest/` for detailed comparative analysis.

## System Requirements

**Minimum:**
- CPU: Modern multi-core processor
- RAM: 8GB
- Storage: 8GB free space
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

### Intent Patterns

The system's intent classification is driven by the `utils/intent_patterns.json` file, which contains **290 training patterns** across 6 intent categories:

```json
{
  "intent_name": {
    "patterns": ["example phrase 1", "example phrase 2", ...],
    "similarity_threshold": 0.40,
    "default_response": "Response text"
  }
}
```

**Pattern Distribution:**
- `order`: 44 patterns - Purchase and ordering intents
- `complaint`: 50 patterns - Issues, refunds, and service problems
- `menu_inquiry`: 55 patterns - Menu questions and product information
- `delivery`: 54 patterns - Delivery status, tracking, and options
- `hours_location`: 46 patterns - Business hours and location queries
- `general`: 41 patterns - Greetings, thanks, and general conversation
- `unknown`: 0 patterns - Fallback for unrecognized intents

Diversity score: 0.970/1.000

Each intent has:
- **patterns**: Training examples for pattern matching and embedding generation
- **similarity_threshold**: Minimum confidence score required to accept the intent for algorithmic/semantic layer classification
- **default_response**: Templated response when no LLM is available

Check Pattern File and Test Dataset distribution and diversity score with: `python test/data.py`

## Usage

### Running the Application inside .venv

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

### Failure Analysis

Generate detailed logs and score breakdown of misclassifications:

```bash
python -m test.runtest -f
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

Runs pytest-based checks for the algorithmic recognizer Similarity scoring and text processing. (76 tests)

```bash
python -m test.runtest -unit 
```

## Performance Benchmarks

All tests use semantic model `all-mpnet-base-v2` and LLM model `llama3.2:3b-instruct-q4_K_M`.

### Extended Test Dataset (600 queries - with edge cases)

Comprehensive testing with 600 queries including 125 edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 97.33% | 31.4ms | 31.8 |
| Algorithmic + Semantic | 91.67% | 4.1ms | 244.8 |
| Algorithmic + LLM | 96.17% | 75.3ms | 13.3 |
| Semantic + LLM | 93.00% | 56.1ms | 17.8 |
| Algorithmic Only | 85.00% | 0.5ms | 1824.2 |
| Semantic Only | 86.17% | 13.7ms | 72.8 |
| LLM Only | 87.17% | 278.9ms | 3.6 |

**Note**: Due to the inherent non-deterministic nature of large language models, full pipeline accuracy scores exhibit variance across repeated evaluations, ranging from 97.17% to 97.50% (±0.33 percentage points). This variability is attributable to the stochastic sampling mechanisms employed in the LLM inference layer and represents expected behavior in hybrid architectures incorporating generative components.

### Layer Distribution (Full Pipeline)

- Algorithmic layer: 72.7% of queries (436/600)
- Semantic layer: 18.0% of queries (108/600)
- LLM layer: 9.3% of queries (56/600)

### Boost Engine Impact
Comparison with and without contextual boost rules on full pipeline (600 queries with edge cases):

| Metric              | Without Boost | With Boost | Improvement |
|---------------------|---------------|------------|-------------|
| Accuracy            | 94.00% | 97.50% | +3.50% |
| Correct Predictions | 564 | 585 | +21 |
| Query Time          | 41.0ms | 30.8ms | 25% faster |
| Algorithmic Usage   | 366 | 436 | +70 |
| Semantic Usage      | 164 | 108 | -56 |
| LLM Fallback        | 70 | 56 | -14 |

### Confusion Matrix Results (Full Pipeline - 600 queries)

#### Per-Intent Performance

| Intent | Precision | Recall  | F1-Score | Support |
|--------|-----------|---------|----------|---------|
| complaint | 99.10%   | 95.65%  | 97.35%   | 115      |
| delivery | 95.83%    | 97.87%  | 96.84%   | 94      |
| general | 96.67%   | 98.31% | 97.48%  | 59      |
| hours_location | 97.87%    | 97.87%  | 97.87%   | 94      |
| menu_inquiry | 98.40%    | 98.40%  | 98.40%   | 125      |
| order | 96.49%    | 97.35%  | 96.92%   | 113      |


See `testResults/` directory for detailed analyses.

### Test Dataset Quality

The benchmark results above are validated against dataset with:

- **600 total queries** (475 normal + 125 edge cases)
- **6 intent categories**: order (113), complaint (115), menu_inquiry (125), hours_location (94), delivery (94), general (59)
- **Diversity score: 0.968/1.000** - High lexical variety, not repetitive memorization
- **Edge cases**: Multi-intent queries, sarcasm, typos, slang, very short queries, ambiguous phrasing, Noise/Formatting
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
4. Adjust boost rules in `intentRecognizer/algorithmic/boostEngine.py` if using algorithmic layer
5. Update llm prompt templates in `intentRecognizer/llm/templates.py`
6. Update test dataset in `test/integration/data.py`

## License

This project is part of a Bachelor's Thesis. All rights reserved.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Coqui TTS for speech synthesis
- Sentence Transformers for semantic embeddings
- Ollama for LLM inference
