# Voice Response System with Query Recognition

A voice-enabled customer service system that recognizes user intent through a multi-layer pipeline combining algorithmic pattern matching, semantic analysis, and LLM support. The system processes voice input, determines intent, generates appropriate responses, and converts them back to speech.

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

## Installation

### Prerequisites

- Python >3.8 ; <=3.11
- Virtual environment (recommended)
- CUDA-compatible GPU (minimum 4GB of VRAM)
- FFmpeg (required for audio processing)
- espeak-ng (required for TTS)

GPU Support:
   - Optional but recommended
   - All components fall back to CPU automatically
   - Expected performance: GPU provides 5-10x speedup

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

### Testing the Intent Recognizer

Run comprehensive tests:

```bash
python -m testData.test_intent_recognizer
```

Run comparative analysis across different pipeline configurations:

```bash
python -m testData.test_intent_recognizer --comparative
```

## Performance Benchmarks

### Standard Test Dataset (213 queries)

Pipeline configuration tested on 213 queries without edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 98.59% | 22.4ms | 44.6 |
| Algorithmic + Semantic | 96.24% | 2.2ms | 449.8 |
| Algorithmic + LLM | 98.12% | 91.6ms | 10.9 |
| Semantic + LLM | 94.37% | 57.1ms | 17.5 |
| Algorithmic Only | 92.96% | 1.5ms | 666.0 |
| Semantic Only | 92.02% | 10.0ms | 100.4 |

### Extended Test Dataset (300 queries with edge cases)

Comprehensive testing with 300 queries including edge cases:

| Configuration | Accuracy | Avg Time | Queries/s |
|--------------|----------|----------|-----------|
| Full Pipeline | 95.67% | 51.4ms | 19.4 |
| Algorithmic + Semantic | 93.00% | 2.6ms | 391.2 |
| Algorithmic + LLM | 93.33% | 169.4ms | 5.9 |
| Semantic + LLM | 90.67% | 95.6ms | 10.5 |
| Algorithmic Only | 85.00% | 1.5ms | 663.0 |
| Semantic Only | 88.33% | 4.8ms | 209.7 |

### Layer Distribution (Full Pipeline)

#### Standard Dataset (213 queries):
- Algorithmic layer: 87.3% of queries (99.46% accuracy)
- Semantic layer: 9.9% of queries (90.48% accuracy)
- LLM layer (local ollama): 2.8% of queries (100.00% accuracy)

#### Extended Dataset (300 queries with edge cases):
- Algorithmic layer: 78.7% of queries (97.46% accuracy)
- Semantic layer: 15.0% of queries (88.89% accuracy)
- LLM layer (local ollama): 6.3% of queries (94.74% accuracy)

### Boost Engine Impact

Comparison with and without contextual boost rules (300 queries):

| Metric | With Boost | Without Boost | Improvement |
|--------|-----------|---------------|-------------|
| Accuracy | 95.67% | 91.67% | +4.00% |
| Algorithmic Usage | 78.7% | 68.7% | +10.0% |
| Avg Query Time | 51.4ms | 68.0ms | 24.4% faster |

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

1. Replace `utils/res_info.json` with your domain information
2. Update intent patterns in `utils/intent_patterns.json`
3. Modify critical keywords in `utils/linguistic_resources.json`
4. Adjust boost rules in `boostEngine.py` if using algorithmic layer
5. Update test dataset in `testData/test_data.py`

## License

This project is part of a Bachelor's Thesis. All rights reserved.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Coqui TTS for speech synthesis
- Sentence Transformers for semantic embeddings
- Ollama for local LLM inference