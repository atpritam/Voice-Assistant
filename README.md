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
│ 1. Algorithmic (fast)   │ ← 77% of queries
│    ├─ Pattern matching  │
│    ├─ Levenshtein      │
│    └─ Boost Engine      │
│         ↓               │
│ 2. Semantic (accurate)  │ ← 16% of queries
│    └─ Neural embeddings │
│         ↓               │
│ 3. LLM (fallback)       │ ← 7% of queries
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

**Recommended:**
- CPU: 4+ cores
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM (CUDA-compatible)
- Storage: 10GB free space

**Note**: All components work on CPU, but GPU provides 5-10x speedup.

### Prerequisites

- Python >3.8 ; <=3.11
- Virtual environment (recommended)
- CUDA-compatible GPU (minimum 4GB of VRAM)
- FFmpeg (required for audio processing)
- espeak-ng (required for TTS)

### System Dependencies

```bash
# Install Python 3.11 besides your existing System Python version
# Coqui TTS currently only supports Python <=3.11
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# check python 3.11 version
python3.11 --version
```

```bash
sudo apt install ffmpeg espeak-ng
```

### Python Dependencies

```bash
# Clone the repository
git clone https://github.com/atpritam/Voice-Assistant.git
cd Voice-Assistant

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Note: OpenAI API key is only required if using OpenAI models. For local LLM inference with Ollama, no API key is needed.

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
SEMANTIC_MODEL = "all-MiniLM-L6-v2"
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

### Standard Test Dataset (213 queries)

Pipeline configuration tested on 213 queries without edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 98.59% | 10.8ms | 92.5 |
| Algorithmic + Semantic | 96.24% | 1.9ms | 519.4 |
| Algorithmic + LLM | 98.12% | 44.7ms | 22.4 |
| Semantic + LLM | 94.37% | 25.3ms | 39.6 |
| Algorithmic Only | 92.96% | 1.3ms | 775.5 |
| Semantic Only | 92.02% | 9.1ms | 109.6 |

### Extended Test Dataset (300 queries with edge cases)

Comprehensive testing with 300 queries including edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 95.67% | 21.3ms | 46.9 |
| Algorithmic + Semantic | 93.00% | 4.0ms | 248.8 |
| Algorithmic + LLM | 94.33% | 58.6ms | 17.1 |
| Semantic + LLM | 91.33% | 38.7ms | 25.8 |
| Algorithmic Only | 85.00% | 1.4ms | 737.4 |
| Semantic Only | 88.33% | 9.4ms | 106.7 |

### Layer Distribution (Full Pipeline)

#### Standard Dataset (213 queries):
- Algorithmic layer: 87.8% of queries (187/213)
- Semantic layer: 9.4% of queries (20/213)
- LLM layer (local ollama): 2.8% of queries (6/213)

#### Extended Dataset (300 queries with edge cases):
- Algorithmic layer: 78.7% of queries (97.46% accuracy)
- Semantic layer: 15.0% of queries (88.89% accuracy)
- LLM layer (local ollama): 6.3% of queries (94.74% accuracy)

### Boost Engine Impact

Comparison with and without contextual boost rules (300 queries):

| Metric | With Boost | Without Boost | Improvement  |
|--------|------------|---------------|--------------|
| Accuracy | 95.67%     | 92.00%        | ~ +4.00%     |
| Algorithmic Usage | 78.7%      | 69.3%         | ~ +10.0%     |
| Avg Query Time | 21.3ms     | 27.2ms        | ~ 22% faster |

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
6. Update test dataset in `testData/test_data.py`

## License

This project is part of a Bachelor's Thesis. All rights reserved.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Coqui TTS for speech synthesis
- Sentence Transformers for semantic embeddings
- Ollama for local LLM inference