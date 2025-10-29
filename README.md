# Voice Response System with Query Recognition

A voice-enabled customer service system that recognizes user intent through a multi-layer pipeline combining algorithmic pattern matching, semantic analysis, and LLM support. The system processes voice input, determines intent, generates appropriate responses, and converts them back to speech.

**Use Case**: Customer service (The System is Domain Agnostic.)
**Current Domain**: Pizza Restaurant

### Key Features

- **Multi-layer Intent Recognition Pipeline**
  - Layer 1: Algorithmic pattern matching with Levenshtein distance and keyword analysis
  - Layer 2: Semantic similarity using Sentence Transformers
  - Layer 3: LLM-based classification and response generation (OpenAI API or local Ollama)

- **Voice Interaction**
  - Automatic Speech Recognition (ASR) using OpenAI Whisper
  - Text-to-Speech (TTS) using Coqui TTS VITS model
  - Audio preprocessing for improved transcription accuracy

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

The system uses a cascading pipeline where each layer is tried sequentially until confident recognition is achieved:

1. **Algorithmic Layer**: Fast pattern matching using keyword extraction, synonym expansion, and string similarity metrics
2. **Semantic Layer**: Neural embedding-based similarity using pre-trained transformer models
3. **LLM Layer**: Fallback to language models for complex queries and response generation

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
python -m test.runtest --c
```

### Boost Engine Analysis

Evaluate the impact of the contextual boost engine:

```bash
python -m test.runtest --b
```

## Performance Benchmarks

All tests use semantic model `all-mpnet-base-v2` and LLM model `llama3.2:3b-instruct-q4_K_M`.

### Standard Test Dataset (304 queries)

Pipeline configuration tested on 304 queries without edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 99.01% | 11.2ms | 88.9 |
| Algorithmic + Semantic | 97.70% | 3.5ms | 288.9 |
| Algorithmic + LLM | 99.34% | 22.6ms | 44.3 |
| Semantic + LLM | 95.07% | 37.6ms | 26.6 |
| Algorithmic Only | 95.39% | 3.0ms | 330.6 |
| Semantic Only | 92.43% | 6.5ms | 153.6 |

### Extended Test Dataset (400 queries with edge cases)

Comprehensive testing with 400 queries including edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 98.75% | 18.9ms | 53.0 |
| Algorithmic + Semantic | 96.00% | 6.1ms | 164.8 |
| Algorithmic + LLM | 97.00% | 45.3ms | 22.1 |
| Semantic + LLM | 93.25% | 48.6ms | 20.6 |
| Algorithmic Only | 89.75% | 2.8ms | 361.2 |
| Semantic Only | 88.00% | 6.8ms | 147.6 |

### Layer Distribution (Full Pipeline)

#### Standard Dataset (304 queries):
- Algorithmic layer: 92.8% of queries (282/304)
- Semantic layer: 4.6% of queries (14/304)
- LLM layer: 2.6% of queries (8/304)

#### Extended Dataset (400 queries with edge cases):
- Algorithmic layer: 84.3% of queries (337/400)
- Semantic layer: 10.8% of queries (43/400)
- LLM layer: 5.0% of queries (20/400)

### Boost Engine Impact

Comparison with and without contextual boost rules (400 queries with edge cases):

#### Algorithmic Only Pipeline

| Metric | Without Boost | With Boost | Improvement |
|--------|---------------|------------|-------------|
| Accuracy | 83.75% | 89.75% | +6.00% |
| Correct Predictions | 335 | 359 | +24 |
| High Confidence | 203 | 264 | +61 |

#### Full Pipeline

| Metric | Without Boost | With Boost | Improvement |
|--------|---------------|------------|-------------|
| Accuracy | 94.50% | 98.75% | +4.25% |
| Correct Predictions | 378 | 395 | +17 |
| Query Time | 31.9ms | 19.4ms | 39% faster |
| Algorithmic Usage | 305 | 337 | +32 |
| Semantic Fallback | 60 | 43 | -17 |
| LLM Fallback | 35 | 20 | -15 |

See `testResults/` directory for detailed comparative analyses.

## Key Technical Features

### Algorithmic Layer Optimizations

- Inverted index with TF-IDF weighting for fast candidate selection
- Length-based prefiltering to skip expensive calculations
- Synonym expansion and filler word removal
- Multi-word phrase matching with bonuses
- Domain-specific contextual boost rules

### Semantic Layer Features

- Embedding caching for instant initialization
- Batch processing for pattern encoding

### LLM Layer Capabilities

- Conversation history awareness
- Context-based response generation
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