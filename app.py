"""
Run with: uvicorn app:socket_app --host 0.0.0.0 --port 5000 --reload
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import socketio
import sys
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

# Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*', max_http_buffer_size=10 ** 7)

app = FastAPI()
socket_app = socketio.ASGIApp(sio, app)
app.mount("/static", StaticFiles(directory="static"), name="static")

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
            test_mode=TEST_MODE
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
        logger.error(f"\nTTS Import Error: {e}")
        logger.error("Install with: pip install -r requirements.txt\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTTS Initialization Error: {e}\n")
        sys.exit(1)


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
        logger.error("\nASR Error: Dependencies not installed")
        logger.error("Install with: pip install openai-whisper")
        if ENABLE_AUDIO_PREPROCESSING:
            logger.info("For preprocessing: pip install librosa soundfile noisereduce\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nASR Initialization Error: {e}\n")
        sys.exit(1)


def perform_system_warmup():
    """Warm up Local LLM and TTS components to eliminate cold start latency"""

    logger.info("Performing system warmup")
    warmup_start = time.time()

    if intent_recognizer:
        try:
            if ENABLE_LLM and not LLM_MODEL.endswith('-cloud'):
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

@sio.on('connect')
async def handle_connect(sid, environ):
    logger.info(f'Client connected : [{sid}]')

@sio.on('disconnect')
async def handle_disconnect(sid):
    logger.info(f'Client disconnected : [{sid}]')
    # Clean up session data
    if sid in session_conversations:
        del session_conversations[sid]

@sio.on('voice_input')
async def handle_voice_input(sid, data):
    """Handle incoming voice audio for transcription and processing"""
    if not asr_service:
        await sio.emit('error', {'message': 'ASR service not available'}, room=sid)
        return

    try:
        audio_data = data.get('audio')
        if not audio_data:
            await sio.emit('error', {'message': 'No audio data received'}, room=sid)
            return

        audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)

        await sio.emit('transcription_status', {'status': 'processing'}, room=sid)

        transcribed_text, confidence = asr_service.transcribe_audio_data(audio_bytes)

        if not transcribed_text:
            await sio.emit('transcription_result', {
                'text': '',
                'confidence': 0.0,
                'error': 'No speech detected'
            }, room=sid)
            return

        await sio.emit('transcription_result', {
            'text': transcribed_text,
            'confidence': confidence
        }, room=sid)

        result = process_user_input(sid, transcribed_text, is_audio=True)
        result['transcribed_text'] = transcribed_text

        await sio.emit('voice_response', result, room=sid)

    except Exception as e:
        logger.error(f" Voice input error: {e}", exc_info=True)
        await sio.emit('error', {'message': f'Processing error: {str(e)}'}, room=sid)

@sio.on('text_input')
async def handle_text_input(sid, data):
    """Handle incoming text messages directly without ASR"""
    try:
        text = data.get('text')
        if not text:
            await sio.emit('error', {'message': 'No text received'}, room=sid)
            return

        result = process_user_input(sid, text, is_audio=False)

        await sio.emit('text_response', result, room=sid)

    except Exception as e:
        logger.error(f"Text input error: {e}", exc_info=True)
        await sio.emit('error', {'message': f'Processing error: {str(e)}'}, room=sid)

@sio.on('clear_history')
async def handle_clear_history(sid):
    session_conversations[sid].clear()
    logger.info(f'Cleared History : [{sid}]')
    await sio.emit('history_cleared', room=sid)

@app.get('/')
async def index():
    """Serve static HTML page"""
    return FileResponse('static/index.html')

@app.get('/statistics')
async def get_statistics():
    """Get system statistics including layer usage, TTS, and ASR"""
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

    return JSONResponse(organized)

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
    model_type = "Cloud" if LLM_MODEL.endswith('-cloud') else "Local"
    layers.append(f"LLM (Ollama-{model_type})")
pipeline = "Pipeline: " + " -> ".join(layers)
logger.info(pipeline)

# System Warmup
perform_system_warmup()