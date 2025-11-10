from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sys
import os
import time
import base64
import logging
from dotenv import load_dotenv
from collections import defaultdict
from utils.logger import setup_logging
setup_logging(level=logging.INFO)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
warnings.filterwarnings('ignore', message='.*is deprecated.*')

from intentRecognizer import IntentRecognizer
from ttsModule import TTSService
from asrModule import ASRService

load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='/static')
SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set in .env")
app.config['SECRET_KEY'] = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10 ** 7, manage_session=False)
CORS(app)

logger = logging.getLogger(__name__)

session_conversations = defaultdict(list)
MAX_CONVERSATION_HISTORY = 50                       # Max conversation turns to keep per session

# PIPELINE CONFIGURATION
ENABLE_ALGORITHMIC = True
ENABLE_SEMANTIC = True
ENABLE_LLM = True

MIN_CONFIDENCE = 0.5                                # Minimum confidence to accept intent
ALGORITHMIC_THRESHOLD = 0.65                        # Min confidence for algorithmic layer to skip next layers
SEMANTIC_THRESHOLD = 0.5                            # Min confidence for semantic layer to skip LLM layer

SEMANTIC_MODEL = "all-mpnet-base-v2"                # Options: "all-MiniLM-L6-v2"

USE_LOCAL_LLM = True                                # True: use Ollama local LLM, False: use Ollama Cloud API
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"           # Options: "gpt-oss:120b-cloud", "gemma3:4b-it-qat"

ENABLE_LOGGING = True
TEST_MODE = False

# TTS Configuration
TTS_MODEL = "tts_models/en/ljspeech/vits"
TTS_OUTPUT_DIR = "./static/audio"

# ASR Configuration
ASR_MODEL = "tiny.en"
ENABLE_AUDIO_PREPROCESSING = True                   # Noise reduction, Normalization, Silence trimming

def initialize_intent_recognizer():
    """Initialize intent recognizer with error handling"""
    try:
        recognizer = IntentRecognizer(
            enable_logging=ENABLE_LOGGING,
            enable_algorithmic=ENABLE_ALGORITHMIC,
            enable_semantic=ENABLE_SEMANTIC,
            enable_llm=ENABLE_LLM,
            algorithmic_threshold=ALGORITHMIC_THRESHOLD,
            semantic_threshold=SEMANTIC_THRESHOLD,
            semantic_model=SEMANTIC_MODEL,
            llm_model=LLM_MODEL,
            device="auto", # "cuda" , "cpu" , "auto" (for semantic model)
            min_confidence=MIN_CONFIDENCE,
            test_mode=TEST_MODE,
            use_local_llm=USE_LOCAL_LLM
        )

        return recognizer

    except ValueError as e:
        logger.error(f"Intent Recognizer Configuration Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Intent Recognizer Initialization Error: {e}\n")
        sys.exit(1)

def initialize_tts_service():
    """Initialize TTS service with error handling"""
    try:
        tts = TTSService(
            model_name=TTS_MODEL,
            device="auto", # "cuda" , "cpu" , "auto"
            enable_logging=ENABLE_LOGGING,
            output_dir=TTS_OUTPUT_DIR,
        )

        return tts

    except ImportError as e:
        print(f"\nTTS Import Error: {e}")
        print("Install with: pip install -r requirements.txt\n")
    except Exception as e:
        logger.error(f"\nTTS Initialization Error: {e}\n")


def initialize_asr_service():
    """Initialize ASR service with error handling"""
    try:
        asr = ASRService(
            model_size=ASR_MODEL,
            device="auto", # "cuda" , "cpu" , "auto"
            enable_logging=ENABLE_LOGGING,
            enable_preprocessing=ENABLE_AUDIO_PREPROCESSING
        )

        return asr

    except ImportError as e:
        print("\nASR Error: Dependencies not installed")
        print("Install with: pip install openai-whisper")
        if ENABLE_AUDIO_PREPROCESSING:
            print("For preprocessing: pip install librosa soundfile noisereduce\n")
    except Exception as e:
        logger.error(f"\nASR Initialization Error: {e}\n")


def perform_system_warmup():
    """Warm up Local LLM and TTS components to eliminate cold start latency"""

    logger.info("Performing system warmup")
    warmup_start = time.time()

    if USE_LOCAL_LLM and intent_recognizer:
        try:
            intent_recognizer.recognize_intent("sample text", [])
        except Exception as e:
            logger.warning(f"Warning: Intent recognizer warmup query failed: {e}")

    if tts_service:
        try:
            warmup_audio = tts_service.generate_speech("System warmup", output_filename="warmup.wav")
            if warmup_audio and os.path.exists(warmup_audio):
                os.unlink(warmup_audio)
        except Exception as e:
            logger.warning(f"\nWarning: TTS warmup failed: {e}")

    total_warmup_time = time.time() - warmup_start
    intent_recognizer.reset_statistics()
    tts_service.reset_statistics()
    logger.info(f"Total System Warm-up Time: {total_warmup_time:.2f}s")

def generate_tts_audio(tts, response):
    """Generate TTS audio and return URL"""
    if not (tts and response):
        return None

    try:
        timestamp = int(time.time() * 1000)
        audio_filename = f"response_{timestamp}.wav"
        audio_path = tts.generate_speech(text=response, output_filename=audio_filename)

        if audio_path:
            return f"/static/audio/{audio_filename}"
    except Exception as e:
        logger.error(f"TTS generation error: {e}")

    return None

def get_session_id() -> str:
    """Get or create a unique session ID for client"""
    return request.sid  # type: ignore

def add_to_conversation_history(session_id, entry_type, message, intent_info=None, audio_url=None, is_audio=False):
    """Add entry to conversation history for a specific session"""
    entry = {
        'type': entry_type,
        'message': message,
        'timestamp': int(time.time() * 1000),
        'audio': is_audio
    }

    if intent_info:
        entry.update({
            'intent': intent_info.intent,
            'confidence': intent_info.confidence_level,
            'similarity': intent_info.confidence,
            'layer_used': intent_info.layer_used,
            'processing_method': intent_info.processing_method,
            'audio_url': audio_url
        })

    session_conversations[session_id].append(entry)

    # Enforce maximum conversation history limit
    if len(session_conversations[session_id]) > MAX_CONVERSATION_HISTORY:
        session_conversations[session_id] = session_conversations[session_id][-MAX_CONVERSATION_HISTORY:]

def process_user_input(session_id, user_text, is_audio=False):
    """Process user input and generate response for a specific session"""
    add_to_conversation_history(session_id, 'user', user_text, is_audio=is_audio)

    conversation_history = session_conversations[session_id]
    intent_info = intent_recognizer.recognize_intent(user_text, conversation_history)
    response = intent_info.response

    audio_url = generate_tts_audio(tts_service, response)

    add_to_conversation_history(session_id, 'assistant', response, intent_info, audio_url, is_audio)

    return {
        'user_text': user_text,
        'assistant_response': response,
        'conversation_history': conversation_history,
        'intent_info': {
            'intent': intent_info.intent,
            'confidence': intent_info.confidence_level,
            'similarity': intent_info.confidence,
            'layer_used': intent_info.layer_used,
            'processing_method': intent_info.processing_method,
        },
        'audio_url': audio_url
    }

@socketio.on('connect')
def handle_connect():
    session_id = get_session_id()
    logger.info(f'Client connected : [{session_id}]')

@socketio.on('disconnect')
def handle_disconnect():
    session_id = get_session_id()
    logger.info(f'Client disconnected : [{session_id}]')
    # Clean up session data
    if session_id in session_conversations:
        del session_conversations[session_id]

@socketio.on('voice_input')
def handle_voice_input(data):
    """Handle incoming voice audio for transcription and processing"""
    if not asr_service:
        emit('error', {'message': 'ASR service not available'})
        return

    try:
        session_id = get_session_id()
        audio_data = data.get('audio')
        if not audio_data:
            emit('error', {'message': 'No audio data received'})
            return

        audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)

        emit('transcription_status', {'status': 'processing'})

        transcribed_text, confidence = asr_service.transcribe_audio_data(audio_bytes)

        if not transcribed_text:
            emit('transcription_result', {
                'text': '',
                'confidence': 0.0,
                'error': 'No speech detected'
            })
            return

        emit('transcription_result', {
            'text': transcribed_text,
            'confidence': confidence
        })

        result = process_user_input(session_id, transcribed_text, is_audio=True)
        result['transcribed_text'] = transcribed_text

        emit('voice_response', result)

    except Exception as e:
        logger.error(f" Voice input error: {e}", exc_info=True)
        emit('error', {'message': f'Processing error: {str(e)}'})

@socketio.on('text_input')
def handle_text_input(data):
    """Handle incoming text messages directly without ASR"""
    try:
        session_id = get_session_id()
        text = data.get('text')
        if not text:
            emit('error', {'message': 'No text received'})
            return

        result = process_user_input(session_id, text, is_audio=False)

        emit('text_response', result)

    except Exception as e:
        logger.error(f"Text input error: {e}", exc_info=True)
        emit('error', {'message': f'Processing error: {str(e)}'})

@socketio.on('clear_history')
def handle_clear_history():
    session_id = get_session_id()
    session_conversations[session_id].clear()
    logger.info(f'Cleared History : [{session_id}]')
    emit('history_cleared')

@app.route('/')
def index():
    """Static HTML page"""
    return app.send_static_file('index.html')

@app.route('/statistics')
def get_statistics():
    """Get system statistics including layer usage, TTS, and ASR"""
    from flask import jsonify
    stats = intent_recognizer.get_statistics()

    stats.pop('intent_distribution', None)
    stats.pop('average_confidence', None)

    if tts_service:
        stats['tts'] = tts_service.get_statistics()

    if asr_service:
        stats['asr'] = asr_service.get_statistics()

    organized = {
        "overview": {
            "total_queries": stats.pop('total_queries_processed', 0),
            "pipeline": stats.pop('pipeline_configuration', {}),
            "layer_usage": stats.pop('layer_usage', {})
        },
        "layers": {k: v for k, v in stats.items() if k.endswith('_layer')},
        "services": {k: v for k, v in stats.items() if k in ['tts', 'asr']}
    }

    organized['layers'].get('algorithmic_layer', {}).pop('patterns_evaluated', None)

    return jsonify(organized)

# System Start
intent_recognizer = initialize_intent_recognizer()
tts_service = initialize_tts_service()
asr_service = initialize_asr_service()

layers = []
if ENABLE_ALGORITHMIC:
    layers.append("Algorithmic")
if ENABLE_SEMANTIC:
    layers.append("Semantic")
if ENABLE_LLM:
    layers.append("LLM (Ollama)" if USE_LOCAL_LLM else "LLM (OpenAI)")
pipeline = "Pipeline: " + " -> ".join(layers)
logger.info(pipeline)

# System Warmup
perform_system_warmup()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, log_output=True)