from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sys
import os
import time

# Utils directory path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'intentRecognizer'))

from intentRecognizer import IntentRecognizer

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'hjbasfbue76t34g76wgv3bywyu47'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

conversation_history = []

# PIPELINE CONFIGURATION
ENABLE_ALGORITHMIC = True  # Keyword pattern matching + Levenshtein distance
ENABLE_SEMANTIC = True  # Sentence Transformers (local ML model)
ENABLE_LLM = True  # LLM layer (OpenAI or Ollama)

# Layer Thresholds - confidence below which next active layer is tried
ALGORITHMIC_THRESHOLD = 0.6
SEMANTIC_THRESHOLD = 0.5

# Model Configuration
SEMANTIC_MODEL = "all-MiniLM-L6-v2"

# LLM Configuration
USE_LOCAL_LLM = True  # Set to True for Ollama, False for OpenAI
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M" if USE_LOCAL_LLM else "gpt-5-nano"
OLLAMA_BASE_URL = "http://localhost:11434"

# General Settings
ENABLE_LOGGING = True
MIN_CONFIDENCE = 0.5

# Mode Configuration
TEST_MODE = False  # When True, skips response generation for faster intent testing

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


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('send_message')
def handle_message(data):
    """Handle incoming text messages with intent recognition and response generation"""
    message = data.get('message', '')
    if message:
        # Add user message to conversation history
        conversation_history.append({
            'type': 'user',
            'message': message,
            'timestamp': int(time.time() * 1000)
        })

        intent_info = intent_recognizer.recognize_intent(message, conversation_history)
        response = intent_info.response

        # Add assistant response to conversation history
        conversation_history.append({
            'type': 'assistant',
            'message': response,
            'timestamp': int(time.time() * 1000) ,
            'intent': intent_info.intent,
            'confidence': intent_info.confidence_level,
            'similarity': intent_info.confidence,
            'layer_used': intent_info.layer_used,
            'processing_method': intent_info.processing_method
        })

        # Send response back to client
        emit('message_response', {
            'user_message': message,
            'assistant_response': response,
            'conversation_history': conversation_history,
            'intent_info': {
                'intent': intent_info.intent,
                'confidence': intent_info.confidence_level,
                'similarity': intent_info.confidence,
                'layer_used': intent_info.layer_used,
                'processing_method': intent_info.processing_method,
                'llm_explanation': intent_info.llm_explanation
            }
        })


@socketio.on('recognize_intent')
def handle_recognize_intent(data):
    """Recognize intent for a given message without adding to conversation history"""
    message = data.get('message', '')
    if message:
        intent_info = intent_recognizer.recognize_intent(message, conversation_history)
        emit('intent_recognition', {
            'message': message,
            'intent_info': {
                'intent': intent_info.intent,
                'confidence': intent_info.confidence_level,
                'similarity': intent_info.confidence,
                'layer_used': intent_info.layer_used,
                'processing_method': intent_info.processing_method,
                'llm_explanation': intent_info.llm_explanation,
                'response': intent_info.response
            }
        })


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
    """Get system statistics including layer usage"""
    from flask import jsonify
    stats = intent_recognizer.get_statistics()
    return jsonify(stats)


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)