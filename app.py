from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sys
import os
import time
import base64
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
warnings.filterwarnings('ignore', message='.*is deprecated.*')

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'intentRecognizer'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ttsModule'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'asrModule'))

from intentRecognizer import IntentRecognizer
from ttsModule import TTSService
from asrModule import ASRService

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'hjbasfbue76t34g76wgv3bywyu47'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10 ** 7)
CORS(app)

conversation_history = []

# PIPELINE CONFIGURATION
ENABLE_ALGORITHMIC = True
ENABLE_SEMANTIC = True
ENABLE_LLM = True

ALGORITHMIC_THRESHOLD = 0.6
SEMANTIC_THRESHOLD = 0.5

SEMANTIC_MODEL = "all-MiniLM-L6-v2"

USE_LOCAL_LLM = True
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M" if USE_LOCAL_LLM else "gpt-5-nano"
OLLAMA_BASE_URL = "http://localhost:11434"

ENABLE_LOGGING = True
MIN_CONFIDENCE = 0.5

TEST_MODE = False

# TTS Configuration
ENABLE_TTS = True
TTS_MODEL = "tts_models/en/ljspeech/vits"
TTS_OUTPUT_DIR = "./static/audio"

# ASR Configuration
ENABLE_ASR = True
ASR_MODEL = "tiny.en"
ASR_DEVICE = "auto"

# Initialize intent recognizer
try:
    intent_recognizer = IntentRecognizer(
        enable_logging=ENABLE_LOGGING,
        enable_algorithmic=ENABLE_ALGORITHMIC,
        enable_semantic=ENABLE_SEMANTIC,
        enable_llm=ENABLE_LLM,
        algorithmic_threshold=ALGORITHMIC_THRESHOLD,
        semantic_threshold=SEMANTIC_THRESHOLD,
        semantic_model=SEMANTIC_MODEL,
        llm_model=LLM_MODEL,
        min_confidence=MIN_CONFIDENCE,
        test_mode=TEST_MODE,
        use_local_llm=USE_LOCAL_LLM,
        ollama_base_url=OLLAMA_BASE_URL,
    )

    print()
    print(f"  Pipeline: ", end="")
    layers = []
    if ENABLE_ALGORITHMIC:
        layers.append("Algorithmic")
    if ENABLE_SEMANTIC:
        layers.append("Semantic")
    if ENABLE_LLM:
        llm_type = "LLM (Ollama)" if USE_LOCAL_LLM else "LLM (OpenAI)"
        layers.append(llm_type)
    print(" -> ".join(layers))

    if ENABLE_SEMANTIC:
        print(f"  Semantic Model: {SEMANTIC_MODEL}")

    if USE_LOCAL_LLM and ENABLE_LLM:
        print(f"  LLM Provider: Ollama")
        print(f"  LLM Model: {LLM_MODEL}")
        print(f"  Ollama URL: {OLLAMA_BASE_URL}")
    elif ENABLE_LLM:
        print(f"  LLM Provider: OpenAI")
        print(f"  LLM Model: {LLM_MODEL}")

    if TEST_MODE:
        print(f"  Mode: TEST MODE (intent recognition only, no response generation)")
    else:
        print(f"  Test Mode: OFF")
    print()

except ValueError as e:
    print(f"\nConfiguration Error: {e}\n")
    sys.exit(1)
except Exception as e:
    print(f"\nInitialization Error: {e}\n")
    sys.exit(1)

# Initialize TTS Service
tts_service = None
if ENABLE_TTS:
    print(f"  TTS Model: {TTS_MODEL}")
    try:
        tts_service = TTSService(
            model_name=TTS_MODEL,
            enable_logging=ENABLE_LOGGING,
            output_dir=TTS_OUTPUT_DIR
        )
        print()
    except ImportError as e:
        print(f"\nTTS Import Error: {e}")
        print("Install with: pip install -r requirements.txt\n")
        ENABLE_TTS = False
    except Exception as e:
        print(f"\nTTS Initialization Error: {e}\n")
        ENABLE_TTS = False

# Initialize ASR Service
asr_service = None
if ENABLE_ASR:
    print(f"  ASR Model: whisper-{ASR_MODEL}")
    try:
        asr_service = ASRService(
            model_size=ASR_MODEL,
            device=ASR_DEVICE,
            enable_logging=ENABLE_LOGGING
        )
        print()
    except ImportError:
        print("\nASR Error: openai-whisper not installed")
        print("Install with: pip install openai-whisper\n")
        ENABLE_ASR = False
    except Exception as e:
        print(f"\nASR Initialization Error: {e}\n")
        ENABLE_ASR = False


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('voice_input')
def handle_voice_input(data):
    """Handle incoming voice audio for transcription and processing"""
    if not ENABLE_ASR or not asr_service:
        emit('error', {'message': 'ASR service not available'})
        return

    try:
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

        conversation_history.append({
            'type': 'user',
            'message': transcribed_text,
            'timestamp': int(time.time() * 1000),
            'audio': True
        })

        intent_info = intent_recognizer.recognize_intent(transcribed_text, conversation_history)
        response = intent_info.response

        audio_url = None
        if ENABLE_TTS and tts_service and response:
            try:
                timestamp = int(time.time() * 1000)
                audio_filename = f"response_{timestamp}.wav"
                audio_path = tts_service.generate_speech(
                    text=response,
                    output_filename=audio_filename
                )

                if audio_path:
                    audio_url = f"/static/audio/{audio_filename}"

            except Exception as e:
                print(f"TTS generation error: {e}")

        conversation_history.append({
            'type': 'assistant',
            'message': response,
            'timestamp': int(time.time() * 1000),
            'intent': intent_info.intent,
            'confidence': intent_info.confidence_level,
            'similarity': intent_info.confidence,
            'layer_used': intent_info.layer_used,
            'processing_method': intent_info.processing_method,
            'audio_url': audio_url,
            'audio': True
        })

        emit('voice_response', {
            'transcribed_text': transcribed_text,
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
        })

    except Exception as e:
        print(f"Voice input error: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': f'Processing error: {str(e)}'})


@socketio.on('clear_history')
def handle_clear_history():
    conversation_history.clear()
    print('Conversation history cleared')


@app.route('/')
def index():
    """Static HTML page"""
    return app.send_static_file('index.html')


@app.route('/statistics')
def get_statistics():
    """Get system statistics including layer usage, TTS, and ASR"""
    from flask import jsonify
    stats = intent_recognizer.get_statistics()

    if ENABLE_TTS and tts_service:
        stats['tts'] = tts_service.get_statistics()

    if ENABLE_ASR and asr_service:
        stats['asr'] = asr_service.get_statistics()

    return jsonify(stats)


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, log_output=True)