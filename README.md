# Voice Response System with Multi-Layer Intent Recognition Pipeline

[![NLP](https://img.shields.io/badge/NLP-Intent%20Recognition-orange.svg)]()


A **hybrid intent recognition system** combining classical NLP pattern matching, neural semantic embeddings, and large language models for robust conversational AI. The system achieves **98% intent recognition accuracy** through a novel three-layer architecture that balances speed, accuracy, and interpretability.

**Use Case**: Domain-agnostic customer service

**Current Domain**: Pizza Restaurant

**Research Focus**: Hybrid NLP approach combining rule-based, embedding-based, and generative techniques

### Key Features

- **Multi-Layer Intent Recognition Pipeline**
  - Layer 1 - Pattern Matching: Algorithmic recognition using Levenshtein edit distance, TF-IDF weighted indexing, and synonym expansion
  - Layer 2 - Neural Embeddings: Semantic similarity using Sentence Transformer embeddings (SBERT) for context-aware matching
  - Layer 3 - Generative AI: LLM-based classification with conversation history awareness (OpenAI API or local Ollama)
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

### 2. Choose Your LLM Backend

**Option A: Local LLM with Ollama (Recommended - Free & Private)**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model and start service
ollama pull llama3.2:3b-instruct-q4_K_M
ollama serve &
```

**Option B: OpenAI API (Faster setup, requires API key)**

Edit `app.py` and set `USE_LOCAL_LLM = False`

```bash
# Get API key from https://platform.openai.com/api-keys
```

### 3. Run the Application

```bash
# Create .env file with generated SECRET_KEY
echo "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > .env

# If using OpenAI (Option B above), also add:
echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env

# Start the application
python app.py
```

Access the web interface at `http://localhost:5000` 

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
│ 1. Algorithmic (fast)   │ ← 84% of queries
│    ├─ Pattern matching  │
│    ├─ Levenshtein      │
│    └─ Boost Engine      │
│         ↓               │
│ 2. Semantic (accurate)  │ ← 11% of queries
│    └─ Neural embeddings │
│         ↓               │
│ 3. LLM (fallback)       │ ← 5% of queries
│    └─ Ollama/OpenAI    │
└─────────────────────────┘
     ↓
Response Generation
     ↓
[TTS - Coqui VITS]
     ↓
Voice Output
```


## Installation

### System Requirements

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

### Detailed Setup Instructions

#### Installing Python 3.11

If you don't have Python 3.11 or need it alongside your existing Python installation:

```bash
# Ubuntu/Debian
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify installation
python3.11 --version 
```

```bash
# Ubuntu/Debian
sudo apt install ffmpeg espeak-ng
```

#### GPU Setup (Optional)

If you have an NVIDIA GPU and want acceleration:

```bash
# Check CUDA availability
nvidia-smi
```

The system will automatically detect and use GPU if available.

### Alternative Installation Methods

#### Using Conda/Mamba

```bash
conda create -n voice-assistant python=3.11
conda activate voice-assistant
conda install -c conda-forge ffmpeg espeak-ng
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# Required
SECRET_KEY=your_generated_secret_key_here

# Optional - only if using OpenAI instead of local Ollama
OPENAI_API_KEY=your_openai_api_key_here
```

Generate a secure `SECRET_KEY`:
```bash
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

## Configuration

### Pipeline Configuration

Edit `app.py` to configure the intent recognition pipeline:

```python
# Layer enabling
ENABLE_ALGORITHMIC = True
ENABLE_SEMANTIC = True
ENABLE_LLM = True

# Confidence thresholds
ALGORITHMIC_THRESHOLD = 0.6
SEMANTIC_THRESHOLD = 0.5

# Model selection
SEMANTIC_MODEL = "all-MiniLM-L6-v2" # Options: all-mpnet-base-v2
USE_LOCAL_LLM = True  # True for Ollama, False for OpenAI
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"  # or "gpt-5-nano" for OpenAI
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

# Use OpenAI instead of Ollama
python -m test.runtest --openai

# Single query test
python -m test.runtest "where is my pizza?"
```

## Performance Benchmarks

All tests use semantic model `all-mpnet-base-v2` and LLM model `llama3.2:3b-instruct-q4_K_M`.

### Standard Test Dataset (301 queries - no edge cases)

Pipeline configurations tested on 301 queries without edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 99.00% | 11.8ms | 85.0 |
| Algorithmic + Semantic | 97.67% | 4.9ms | 204.3 |
| Algorithmic + LLM | 99.34% | 22.5ms | 44.5 |
| Semantic + LLM | 95.02% | 39.4ms | 25.4 |
| Algorithmic Only | 95.35% | 3.0ms | 338.3 |
| Semantic Only | 92.36% | 14.5ms | 68.8 |

### Extended Test Dataset (400 queries - with edge cases)

Comprehensive testing with 400 queries including 99 edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 98.00% | 19.1ms | 52.5 |
| Algorithmic + Semantic | 95.50% | 4.0ms | 247.5 |
| Algorithmic + LLM | 96.25% | 46.4ms | 21.5 |
| Semantic + LLM | 93.25% | 53.0ms | 18.9 |
| Algorithmic Only | 89.25% | 2.9ms | 350.2 |
| Semantic Only | 88.00% | 13.4ms | 74.5 |

### Layer Distribution (Full Pipeline)

#### Standard Dataset (301 queries):
- Algorithmic layer: 92.7% of queries (279/301)
- Semantic layer: 4.7% of queries (14/301)
- LLM layer: 2.7% of queries (8/301)

#### Extended Dataset (400 queries with edge cases):
- Algorithmic layer: 84.0% of queries (336/400)
- Semantic layer: 10.8% of queries (43/400)
- LLM layer: 5.2% of queries (21/400)

### Boost Engine Impact

Comparison with and without contextual boost rules on full pipeline (400 queries with edge cases):

| Metric | Without Boost | With Boost | Improvement |
|--------|---------------|------------|-------------|
| Accuracy | 93.75% | 98.00% | +4.25% |
| Correct Predictions | 375 | 392 | +17 |
| Query Time | 32.2ms | 20.1ms | 37% faster |
| Algorithmic Usage | 304 | 336 | +32 |
| Semantic Fallback | 60 | 43 | -17 |
| LLM Fallback | 36 | 21 | -15 |

### Confusion Matrix Results (Full Pipeline - 400 queries)

#### Per-Intent Performance

| Intent | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| complaint | 100.00% | 98.81% | 99.40% | 84 |
| delivery | 91.80% | 98.25% | 94.92% | 57 |
| general | 95.65% | 100.00% | 97.78% | 22 |
| hours_location | 100.00% | 95.16% | 97.52% | 62 |
| menu_inquiry | 100.00% | 100.00% | 100.00% | 87 |
| order | 97.70% | 96.59% | 97.14% | 88 |


See `testResults/` directory for detailed  analyses.

## Key Technical Features

### Algorithmic Layer

- **Information Retrieval**: Inverted index with TF-IDF weighting for efficient candidate selection
- **String Similarity Metrics**: Levenshtein edit distance with length-based prefiltering
- **Synonym Expansion**: Domain-specific synonym dictionaries for lexical variation handling
- **NLP Preprocessing**: Contraction expansion, filler word removal, and tokenization
- **N-gram Matching**: Multi-word phrase detection with contextual bonuses
- **Contextual Boost Engine**:
  - Negative sentiment keyword detection for complaint classification
  - Dictionary-based entity keyword matching for domain entities
  - Keyword co-occurrence patterns and contextual heuristics
  - Domain-specific rule-based disambiguation

### Semantic Layer

- **Transformer Embeddings**: Sentence-BERT models (MPNet, MiniLM) for semantic similarity
- **Embedding Caching**: Persistent storage for instant initialization and reduced cold-start latency
- **Batch Processing**: Optimized pattern encoding

### LLM Layer

- **Conversation Context**: Conversation history tracking for context-aware classification and response generation
- Support for both cloud (OpenAI) and local (Ollama) models

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
- Ollama for local LLM inference